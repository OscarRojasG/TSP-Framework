import os
import json
import copy
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import random_split
from training.sl import train
from training.common import *
from settings import DATA_FOLDER, HYPERPARAMETERS_FOLDER, MODELS_FOLDER
from TSP import TSPState
from solvers.model import ModelSolver
from instances.instances import read_instances
from data.generation import save_data
from data.preprocessing import H5Dataset

# Variables globales para los workers
worker_solver = None
worker_input_adapter = None
worker_output_adapter = None

def init_worker(model_class, hyperparams, model_state_dict, in_adapter_config, out_adapter_config, device_str):
    """
    Inicializa los adaptadores y recrea el modelo/solver globalmente en cada proceso worker,
    instanciando la clase del modelo con sus hiperparámetros.
    """
    global worker_solver
    global worker_input_adapter
    global worker_output_adapter

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # 1. Inicializar adaptadores
    in_class, *in_args = in_adapter_config
    out_class, *out_args = out_adapter_config
    worker_input_adapter = in_class(*in_args)
    worker_output_adapter = out_class(*out_args)

    # 2. Recrear el modelo y cargar pesos
    model = model_class(**hyperparams)
    model.load_state_dict(model_state_dict, strict=True)
    model.to(device_str)
    model.eval()

    # 3. Inicializar ModelSolver
    worker_solver = ModelSolver(model, worker_input_adapter)


def clone_state(state: TSPState) -> TSPState:
    """
    Función auxiliar ultrarrápida para clonar un TSPState sin usar copy.deepcopy,
    lo cual acelera enormemente la generación de datos en el Lookahead.
    """
    new_state = TSPState(state.instance)
    new_state.visited = list(state.visited)
    new_state.current_city = state.current_city
    new_state.tour = list(state.tour)
    new_state.cost = state.cost
    return new_state


def generate_data_from_instance(instance):
    """
    Fase de Generación: 1-Step Lookahead + Greedy Rollouts.
    Ejecutado por cada worker para una instancia específica.
    """
    input_vecs = []
    output_vecs = []

    state = TSPState(instance)
    
    with torch.no_grad():
        # Optimizacion: El Encoder se ejecuta UNA SOLA VEZ por instancia
        inputs = worker_solver._get_tensor_inputs(state)
        memory = worker_solver.model.encode(*inputs)

        while not state.is_finished():
            # Extraer índices de ciudades no visitadas iterando sobre state.visited
            unvisited = [i for i, v in enumerate(state.visited) if not v]
            
            best_city = None
            best_cost = float('inf')
            
            # 1-Step Lookahead: Evaluar cada ciudad hija posible
            for city in unvisited:
                child_state = clone_state(state)
                child_state.visit_city(city)
                
                # Greedy Rollout desde la ciudad hija
                rollout_state = clone_state(child_state)
                while not rollout_state.is_finished():
                    curr_inputs = worker_solver._get_tensor_inputs(rollout_state)
                    logits = worker_solver.model.decode(memory, *curr_inputs)
                    next_city = logits.argmax(dim=-1).item()
                    rollout_state.visit_city(next_city)
                
                # Evaluar costo TOTAL del rollout (incluyendo retorno a ciudad inicial)
                cost = rollout_state.get_total_cost()
                if cost < best_cost:
                    best_cost = cost
                    best_city = city
            
            # Guardar el estado actual y la mejor acción encontrada
            in_vec = worker_input_adapter.input_2_vec(state)
            out_vec = worker_output_adapter.output_2_vec(state, best_city)
            
            input_vecs.append(in_vec)
            output_vecs.append(out_vec)
            
            # Avanzar en la trayectoria real usando la mejor ciudad
            state.visit_city(best_city)

    return input_vecs, output_vecs


def save_model(model, model_name, verbose=True):
    os.makedirs(HYPERPARAMETERS_FOLDER, exist_ok=True)
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'w') as f:
        json.dump(model.hyperparams, f, indent=4)

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    weights = model.state_dict()
    torch.save(weights, str(MODELS_FOLDER / model_name) + ".pth")
    if verbose:
        print(f"** Modelo guardado en {MODELS_FOLDER / model_name}.pth")


def evaluate_policy(instances, model_solver):
    """
    Evalúa el costo promedio del modelo actual sobre un set de instancias.
    """
    total_cost = 0.0
    for instance in instances:
        final_state = model_solver.solve_instance(instance)
        # Usamos get_total_cost() para incluir el trayecto de vuelta
        total_cost += final_state.get_total_cost()
    return total_cost / len(instances)


def train_expert_iteration(
    model, 
    train_instances_file, 
    val_instances_file,   
    in_adapter_config, 
    out_adapter_config, 
    iterations, 
    epochs, 
    batch_size, 
    lr_config, 
    weight_decay, 
    loss_fn, 
    patience, 
    metrics,
    num_workers=4,
    base_model_name="tsp_exit_model",
    nn_train_split=0.8,
    seed=42
):
    model, device = config_training(model, seed)
    if num_workers is None:
        num_workers = os.cpu_count()

    print("** Cargando instancias desde los archivos...")
    train_instances = read_instances(train_instances_file)
    val_instances = read_instances(val_instances_file)
    print(f"** Instancias de entrenamiento: {len(train_instances)} | Instancias de validación: {len(val_instances)}")

    best_model = copy.deepcopy(model)
    best_solver = ModelSolver(best_model, in_adapter_config[0](*in_adapter_config[1:]))
    
    # Evaluar costo del modelo inicial sin entrenar (Greedy)
    best_val_cost = evaluate_policy(val_instances, best_solver)
    
    model_class = model.__class__
    hyperparams = model.hyperparams
    
    print(f"** Costo inicial de validación (Greedy): {best_val_cost:.4f}")

    for m in range(1, iterations + 1):
        print(f"\n{'='*40}")
        print(f"** ITERACIÓN EXIT {m}/{iterations}")
        print(f"{'='*40}")
        
        # ---------------------------------------------------------
        # FASE 1: GENERACIÓN DE DATOS (PARALELA)
        # ---------------------------------------------------------
        print(f"** Generando dataset paralelo (Lookahead + Rollouts) con {num_workers} workers...")
        current_state_dict = best_model.state_dict()

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(model_class, hyperparams, current_state_dict, in_adapter_config, out_adapter_config, device.type)
        ) as executor:
            results = list(executor.map(generate_data_from_instance, train_instances))

        main_in_adapter = in_adapter_config[0](*in_adapter_config[1:])
        main_out_adapter = out_adapter_config[0](*out_adapter_config[1:])

        for result in results:
            if result is None:
                continue
            input_vecs, output_vecs = result
            for in_vec, out_vec in zip(input_vecs, output_vecs):
                main_in_adapter.add(in_vec)
                main_out_adapter.add(out_vec)

        in_data = main_in_adapter.get()
        out_data = main_out_adapter.get()
        
        dataset_size = len(in_data[list(in_data.keys())[0]])
        print(f"** Generados {dataset_size} estados de entrenamiento.")

        # ---------------------------------------------------------
        # GUARDAR EN HDF5 Y CARGAR DATASET
        # ---------------------------------------------------------
        temp_h5_filename = f"exit_temp_iter_{m}.h5"
        
        save_data(in_data, out_data, temp_h5_filename, verbose=False)
        dataset_path = DATA_FOLDER / temp_h5_filename
        dataset = H5Dataset(dataset_path)

        # ---------------------------------------------------------
        # SPLIT PARA ENTRENAMIENTO DE LA RED
        # ---------------------------------------------------------
        train_size = int(nn_train_split * len(dataset))
        val_size = len(dataset) - train_size
        nn_train_set, nn_val_set = random_split(dataset, [train_size, val_size])

        # ---------------------------------------------------------
        # FASE 2: ENTRENAMIENTO DE LA RED
        # ---------------------------------------------------------
        print(f"** Entrenando red neuronal por {epochs} épocas...")
        current_model, _, _ = train(
            model=model, 
            epochs=epochs,
            train_set=nn_train_set, 
            val_set=nn_val_set,     
            batch_size=batch_size,
            lr_config=lr_config,
            weight_decay=weight_decay,
            loss_fn=loss_fn,
            patience=patience,
            metrics=metrics,
            metrics_filename=f"metrics_iter_{m}.json",
            device=device
        )

        # ---------------------------------------------------------
        # FASE 3: EVALUACIÓN Y ACTUALIZACIÓN (VALIDACIÓN DE POLÍTICA)
        # ---------------------------------------------------------
        current_solver = ModelSolver(current_model, main_in_adapter)
        val_cost = evaluate_policy(val_instances, current_solver)
        
        print(f"\n** Costo Validación Iteración {m}: {val_cost:.4f} (Anterior Mejor: {best_val_cost:.4f})")

        if val_cost < best_val_cost:
            print("** Nuevo mejor modelo encontrado.")
            best_val_cost = val_cost
            best_model = copy.deepcopy(current_model)
            save_model(best_model, f"{base_model_name}_best", verbose=True)
        else:
            print("** El modelo no superó al experto anterior. Mantenemos el experto histórico.")
            
        save_model(current_model, f"{base_model_name}_iter_{m}", verbose=False)

        if os.path.exists(dataset_path):
            os.remove(dataset_path)

    print("\n** ENTRENAMIENTO EXPERT ITERATION COMPLETADO.")
    return best_model