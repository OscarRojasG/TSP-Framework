from .env import *
import torch
from torch import optim
import numpy as np
import copy
from ..models.base.transformer import Transformer
from .reinforce import generate_rl_data

env = TSP_Environment()

@torch.no_grad()
def generate_greedy_cost(model: Transformer, instance):
    model.eval()

    state = env.initial_state(instance)

    X_src = np.array(instance.city_locations)
    X_src = torch.tensor(X_src, dtype=torch.float32).unsqueeze(0)
    memory = model.encode(X_src)

    while True:
        if state.is_finished():
            break

        visited = np.pad(
            np.array(state.tour),
            (0, instance.num_cities - len(state.tour)),
            'constant',
            constant_values=-1
        )
        visited = torch.tensor(np.array([visited]), dtype=torch.int32)

        output = model.decode(memory, visited)
        probs = output.squeeze(0)

        # Política greedy
        city = torch.argmax(probs)

        action = TSP_Action(city.item())
        env.state_transition(state, action)

    model.train()
    return state.cost


def evaluate(model, test_instances):
    cost_sum = 0
    for instance in test_instances:
        cost = generate_greedy_cost(model, instance)
        cost_sum += cost
    return cost_sum / len(test_instances)


def reinforce(model, train_instances, test_instances, eps=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    cost_arr = []
    loss_arr = []

    greedy_model = copy.deepcopy(model)  # inicializamos la política greedy
    best_avg_cost = None

    for i in range(eps):
        batch_log_probs = []
        batch_costs = []
        batch_baseline = []

        # generar datos RL para cada instancia en el batch
        for instance in train_instances:
            log_probs, rewards, cost = generate_rl_data(model, instance)
            log_probs = log_probs.sum()
            baseline_cost = generate_greedy_cost(greedy_model, instance)

            batch_log_probs.append(log_probs)
            batch_costs.append(cost)
            batch_baseline.append(baseline_cost)

        # concatenar los log_probs de todas las instancias
        log_probs_tensor = torch.stack(batch_log_probs)
        advantages_tensor = torch.tensor([b - c for b, c in zip(batch_baseline, batch_costs)], dtype=torch.float32)

        # calcular la pérdida batch-wise
        loss = -(log_probs_tensor * advantages_tensor / len(log_probs_tensor)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluar costo promedio en test
        avg_cost = evaluate(model, test_instances)
        if best_avg_cost is None:
            best_avg_cost = avg_cost
        elif avg_cost < best_avg_cost:
            greedy_model.load_state_dict(model.state_dict())
            best_avg_cost = avg_cost

        cost_arr.append(np.mean(batch_costs))
        loss_arr.append(loss.item())

        print(
            f'Episodio: {i}\t'
            f'Costo batch sampleado: {np.mean(batch_costs):.2f}\t'
            f'Costo batch greedy: {np.mean(batch_baseline):.2f}\t'
            f'Costo batch validación: {best_avg_cost:.2f}\t'
            f'Loss: {loss.item():.4f}'
        )

    return greedy_model, cost_arr, loss_arr