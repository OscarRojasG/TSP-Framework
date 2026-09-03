import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from settings import INSTANCE_FOLDER
from instances.instances import read_instances
from data.sparsification import sparse_instances
from solvers.model import ModelSolver 
from solvers.ortools import ORToolsSolver

# ==========================================
# Workers para el Modelo PyTorch
# ==========================================
worker_solver = None

def init_worker(model_cls, model_hyperparams, model_weights, input_adapter_config):
    """Inicializa el modelo y el adapter en cada worker para evitar bloqueos por Pickling."""
    global worker_solver
    torch.set_num_threads(1)
    
    adapter_cls, *adapter_args = input_adapter_config
    input_adapter = adapter_cls(*adapter_args)
    
    model = model_cls(**model_hyperparams)
    model.load_state_dict(model_weights)
    model.eval()
    
    worker_solver = ModelSolver(model, input_adapter)

def evaluate_single_instance(instance):
    return worker_solver.solve_instance(instance)


# ==========================================
# Workers para OR-Tools
# ==========================================
ort_worker_solver = None

def init_ortools_worker():
    """Inicializa el solver de OR-Tools en cada worker."""
    global ort_worker_solver
    ort_worker_solver = ORToolsSolver()

def evaluate_ortools_single_instance(instance):
    return ort_worker_solver.solve_instance(instance)


# ==========================================
# Función principal de Evaluación y Comparación
# ==========================================
def evaluate(model, instance_file, input_adapter_config, num_workers=None, sparse=False):
    """
    Evalúa instancias en paralelo usando el Modelo y OR-Tools, 
    e imprime los costos promedios, el gap y la desviación estándar.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
        
    instance_path = INSTANCE_FOLDER / instance_file
    instances = read_instances(instance_path)
    
    if sparse:
        instances = sparse_instances(instances)

    print(f"Iniciando evaluación conjunta de {len(instances)} instancias con {num_workers} workers...")

    # 1. Evaluación con el Modelo
    model_cls = model.__class__
    model_hyperparams = model.hyperparams
    model_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(model_cls, model_hyperparams, model_weights, input_adapter_config)
    ) as executor:
        model_sols = list(executor.map(evaluate_single_instance, instances))
        
    # 2. Evaluación con OR-Tools
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_ortools_worker
    ) as executor:
        ort_sols = list(executor.map(evaluate_ortools_single_instance, instances))

    # 3. Cálculo de métricas
    # Nota: Se utiliza sol.cost asumiendo la estructura del código original proporcionado
    model_costs = [sol.get_total_cost() for sol in model_sols]
    ort_costs = [sol.get_total_cost() for sol in ort_sols]
    
    model_avg = np.mean(model_costs)
    ort_avg = np.mean(ort_costs)
    
    gaps = []
    for m_cost, o_cost in zip(model_costs, ort_costs):
        gap = ((m_cost - o_cost) / o_cost) * 100
        gaps.append(gap)
        
    gap_mean = np.mean(gaps)
    gap_std = np.std(gaps)
    
    print("\n" + "="*40)
    print("RESULTADOS DE LA VALIDACIÓN")
    print("="*40)
    print(f"Costo promedio Modelo:   {model_avg:.2f}")
    print(f"Costo promedio OR-Tools: {ort_avg:.2f}")
    print(f"Gap de optimalidad:      {gap_mean:.2f}% ± {gap_std:.2f}%")
    print("="*40 + "\n")
    
    return model_sols, ort_sols