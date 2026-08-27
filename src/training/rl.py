import os
import time
import random
from typing import List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from TSP import TSPState
from solvers.model import ModelSolver
from training.common import *

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class POMOConfig:
    updates: int = 10000            # Total de iteraciones de entrenamiento
    k_rollouts: int = 16            # Trayectorias independientes por instancia (K)
    instances_per_update: int = 64  # Cuántas instancias distintas usar por update
    adv_clip: float = 4.0           # Límite máximo/mínimo para la ventaja
    grad_clip: float = 1.0          # Límite para la norma del gradiente
    minibatch_size: int = 2048      # Tamaño de batch en la fase de replay
    eval_interval: int = 100        # Frecuencia de evaluación en el val-set
    patience: int = 15              # Evaluaciones sin mejora antes de parar

@torch.no_grad()
def sample_pomo_rollouts(
    model: torch.nn.Module, 
    instances: List[Any], 
    k_rollouts: int, 
    input_adapter_config: Tuple[Any, ...], 
    device: torch.device
):
    """
    Juega K partidas en paralelo para cada instancia usando muestreo multinomial.
    """
    model.eval()
    la_class, *la_args = input_adapter_config
    
    total_envs = len(instances) * k_rollouts
    states = []
    
    # 1. Instanciar los K rollouts (Estados del TSP)
    for inst in instances: 
        for _ in range(k_rollouts):
            states.append(TSPState(inst))
            
    active_mask = np.ones(total_envs, dtype=bool)
    
    flat_inputs_steps = []
    flat_actions = []
    flat_row = []
    
    # Pre-codificar la memoria estática (Encoder) para eficiencia
    # Asumimos que la primera llamada extrae coords, num_cities, etc.
    input_adapter = la_class(*la_args)
    for state in states:
        input_adapter.add(input_adapter.input_2_vec(state))
    
    initial_batch_dict = input_adapter.get()
    initial_inputs = [torch.tensor(initial_batch_dict[key], device=device) for key in initial_batch_dict]
    
    # El encoder se llama UNA SOLA VEZ por todo el rollout
    memory = model.encode(*initial_inputs)
    
    # 2. Bucle de simulación paralela (Rollout autoregresivo)
    while np.any(active_mask):
        active_indices = np.where(active_mask)[0]
        
        # --- A. Vectorización ---
        input_adapter = la_class(*la_args)
        for idx in active_indices:
            vec = input_adapter.input_2_vec(states[idx]) 
            input_adapter.add(vec)
            
        batch_dict = input_adapter.get()
        batch_inputs = [torch.tensor(batch_dict[key], device=device) for key in batch_dict]
            
        # --- B. Inferencia de la Red (Decoder) ---
        # Filtramos la memoria solo para los entornos activos
        active_memory = memory[active_indices]
        logits = model.decode(active_memory, *batch_inputs)
        
        # --- C. Enmascaramiento y Probabilidades ---
        probs = torch.softmax(logits, dim=-1)
        
        # --- D. Muestreo Multinominal (Acción) ---
        sampled_actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        flat_inputs_steps.append([b.cpu() for b in batch_inputs])
        flat_actions.append(sampled_actions.cpu())
        flat_row.append(torch.tensor(active_indices, dtype=torch.long))
        
        # --- E. Aplicar Movimiento ---
        for i, idx in enumerate(active_indices):
            state = states[idx]
            action_idx = sampled_actions[i].item()
            
            state.visit_city(action_idx)
            
            if state.is_finished():
                active_mask[idx] = False

    # 3. Asignación de Recompensas Finales
    # Para el TSP, la recompensa es el costo negativo del tour (queremos minimizar la distancia)
    returns = torch.zeros(total_envs, dtype=torch.float32, device=device)
    for idx in range(total_envs):
        returns[idx] = -states[idx].cost  # Asume que tienes un atributo 'cost' o similar
            
    states_history = [torch.cat(tensors, dim=0) for tensors in zip(*flat_inputs_steps)]
    actions_history = torch.cat(flat_actions)
    rollout_idx_history = torch.cat(flat_row)
    
    return states_history, actions_history, returns, rollout_idx_history

def compute_pomo_advantages(
    returns: torch.Tensor, 
    rollout_idx_history: torch.Tensor,
    k_rollouts: int, 
    clip_val: float,
    device: torch.device
) -> torch.Tensor:
    """
    Calcula la ventaja POMO y la propaga al historial de pasos.
    """
    total_envs = returns.shape[0]
    num_instances = total_envs // k_rollouts
    
    # 1. Baseline por instancia (Auto-competencia POMO)
    # returns_matrix shape: (num_instances, k_rollouts)
    returns_matrix = returns.view(num_instances, k_rollouts)
    
    # baseline shape: (num_instances, 1)
    baseline = returns_matrix.mean(dim=1, keepdim=True)
    
    # advantages_matrix shape: (num_instances, k_rollouts)
    advantages_matrix = returns_matrix - baseline
    
    # 2. Normalización por instancia (Opcional pero recomendado para estabilidad)
    # std shape: (num_instances, 1)
    std = returns_matrix.std(dim=1, keepdim=True)
    
    # Dividimos matricialmente (se aplica el broadcasting a los K rollouts) 
    # y LUEGO aplanamos a (total_envs,)
    advantages_norm = (advantages_matrix / (std + 1e-6)).view(total_envs)
            
    # 3. Clipping
    advantages_norm = torch.clamp(advantages_norm, min=-clip_val, max=clip_val)
            
    # 4. Propagación temporal
    advantages_history = advantages_norm[rollout_idx_history.to(device)]
    
    return advantages_history

def update_model_pomo(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    states_history: List[torch.Tensor], 
    actions_history: torch.Tensor, 
    advantages_history: torch.Tensor, 
    minibatch_size: int, 
    grad_clip: float, 
    device: torch.device
):
    """
    Realiza el forward plano (Actor) acumulando gradientes en chunks de minibatch_size.
    """
    model.train()

    total_steps = actions_history.shape[0]
    num_inputs = len(states_history)
    
    total_pg_loss = 0.0
    total_entropy = 0.0
    
    optimizer.zero_grad(set_to_none=True)
    indices = torch.randperm(total_steps)
    
    for start_idx in range(0, total_steps, minibatch_size):
        end_idx = min(start_idx + minibatch_size, total_steps)
        mb_indices = indices[start_idx:end_idx]
        mb_size = end_idx - start_idx
        
        mb_states = [states_history[j][mb_indices].to(device) for j in range(num_inputs)]
        mb_actions = actions_history[mb_indices].to(device)
        mb_advantages = advantages_history[mb_indices.to(device)]
        
        # Volver a pasar por la red para obtener el grafo de gradientes
        memory = model.encode(*mb_states)
        logits = model.decode(memory, *mb_states)
        
        # Pérdida REINFORCE
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob_actions = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
        
        pg_loss = -(mb_advantages * log_prob_actions).mean()
        
        # Escalar la pérdida si se acumulan gradientes
        ratio = mb_size / total_steps
        scaled_loss = pg_loss * ratio
        
        scaled_loss.backward()
        
        total_pg_loss += pg_loss.item() * ratio
        
        # Entropía para métricas
        with torch.no_grad():
            probs = torch.exp(log_probs)
            ent = -(probs * log_probs).sum(dim=-1).mean()
            total_entropy += ent.item() * ratio
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    return total_pg_loss, total_entropy

@torch.no_grad()
def evaluate_val_set(
    model: torch.nn.Module, 
    val_instances: List[Any], 
    input_adapter_config: tuple,
) -> float:
    """
    Evalúa el modelo usando el ModelSolver para sacar soluciones deterministas (Greedy).
    """
    la_class, *la_args = input_adapter_config
    input_adapter = la_class(*la_args)
    
    solver = ModelSolver(model, input_adapter)
    
    total_cost = 0.0
    for inst in val_instances:
        solution_state = solver.solve_instance(inst)
        total_cost += solution_state.cost
        
    return float(total_cost / len(val_instances))

def train_pomo_rl(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    train_instances: list,
    val_instances: list,       
    pomo_config: POMOConfig,                
    input_adapter_config: tuple,                
    device: torch.device,
    save_dir: str,
    model_name: str
):
    print(f"Iniciando POMO RL por {pomo_config.updates} updates...")
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_score = float('inf')
    evals_without_improvement = 0
    
    for update in range(1, pomo_config.updates + 1):
        start_time = time.time()
        
        # Muestrear instancias para este update
        batch_instances = random.sample(train_instances, pomo_config.instances_per_update)
                
        states_hist, actions_hist, returns, rollout_idx_hist = sample_pomo_rollouts(
            model, batch_instances, 
            pomo_config.k_rollouts, 
            input_adapter_config,
            device
        )
        
        advantages_hist = compute_pomo_advantages(
            returns, rollout_idx_hist, 
            pomo_config.k_rollouts, 
            pomo_config.adv_clip, 
            device
        )
        
        pg_loss, entropy = update_model_pomo(
            model, optimizer,
            states_hist, actions_hist, advantages_hist,
            pomo_config.minibatch_size,
            pomo_config.grad_clip, device
        )
        
        # Telemetría 
        avg_cost = -returns.mean().item() # returns es -cost
        fps = len(actions_hist) / max(1e-4, time.time() - start_time)
        
        print(f"Update {update:05d} | "
              f"PG: {pg_loss:+.3f} | Ent: {entropy:.3f} | "
              f"AvgCost: {avg_cost:.2f} | {fps:.0f} steps/s")
        
        # Evaluación en Val-Set
        if update % pomo_config.eval_interval == 0:
            val_score = evaluate_val_set(
                model, val_instances, input_adapter_config
            )
            
            print(f"   >>> Evaluación Val Set: {val_score:.2f} | (Mejor: {min(val_score, best_val_score):.2f})")
            
            if val_score < best_val_score:
                best_val_score = val_score
                evals_without_improvement = 0
                
                save_model(model, model_name, verbose=False)
                print(f"   >>> Mejor modelo actualizado.")
            else:
                evals_without_improvement += 1
                print(f"   >>> Sin mejora ({evals_without_improvement}/{pomo_config.patience}).")
                
                if evals_without_improvement > pomo_config.patience:
                    print(f"   >>> Entrenamiento detenido por paciencia.")
                    break
        
    print("** Entrenamiento POMO finalizado.")