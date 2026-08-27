import os
import torch
from concurrent.futures import ProcessPoolExecutor
from settings import INSTANCE_FOLDER
from instances.instances import read_instances
from solvers.model import ModelSolver 

# Variable global para cada proceso worker
worker_solver = None

def init_worker(model_cls, model_hyperparams, model_weights, input_adapter_config):
    """
    Inicializa el modelo, el adaptador y el solver independientemente en cada worker.
    """
    global worker_solver
    
    # 1. Evitar que cada proceso acapare todos los hilos del CPU
    torch.set_num_threads(1)
    
    # 2. Reconstruir el adaptador
    adapter_cls, *adapter_args = input_adapter_config
    input_adapter = adapter_cls(*adapter_args)
    
    # 3. Reconstruir el modelo
    model = model_cls(**model_hyperparams)
    model.load_state_dict(model_weights)
    
    # 4. Crear el solver
    worker_solver = ModelSolver(model, input_adapter)

def evaluate_single_instance(instance):
    """
    Función que ejecuta cada worker de manera aislada.
    """
    return worker_solver.solve_instance(instance)

def evaluate(model, instance_file, input_adapter_config, num_workers=None):
    """
    Función principal para evaluar instancias en paralelo.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
        
    # Leer las instancias
    instance_path = INSTANCE_FOLDER / instance_file
    instances = read_instances(instance_path)

    # Extraer configuración y pesos del modelo para enviarlos a los workers
    model_cls = model.__class__
    model_hyperparams = model.hyperparams
    
    # Pasamos los pesos a CPU antes de enviarlos para evitar errores de Pickle con tensores CUDA
    model_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    print(f"Iniciando evaluación de {len(instances)} instancias con {num_workers} workers...")

    # Paralelismo usando ProcessPoolExecutor
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(model_cls, model_hyperparams, model_weights, input_adapter_config)
    ) as executor:
        # executor.map mantiene el mismo orden de la lista de instancias original
        solutions = list(executor.map(evaluate_single_instance, instances))
        
    # Calcular e imprimir el costo promedio
    total_cost = sum(sol.cost for sol in solutions)
    avg_cost = total_cost / len(solutions)
    
    print(f"** Evaluación completada. Costo promedio: {avg_cost:.4f}")
    
    return solutions