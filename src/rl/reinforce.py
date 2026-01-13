from .env import *
import torch
from torch import optim
import numpy as np
from ..models.base.transformer import Transformer

env = TSP_Environment()

def generate_rl_data(model: Transformer, instance):
    p = [] # Secuencia de probabilidades
    r = [] # Secuencia de recompensas

    # Generar estado inicial
    state = env.initial_state(instance)

    # Entrada encoder: coordenadas
    X_src = np.array(instance.city_locations)
    X_src = torch.tensor(X_src, dtype=torch.float32).unsqueeze(0)
    memory = model.encode(X_src)

    # Generar secuencia de acciones
    while True:
        if state.is_finished(): break # Estado completado

        # Entrada decoder: memoria + tour 
        visited = np.pad(np.array(state.tour), (0, instance.num_cities - len(state.tour)), 'constant', constant_values=-1)
        visited = torch.tensor(np.array([visited]), dtype=torch.int32)
        output = model.decode(memory, visited)
        probs = output.squeeze(0)

        # Predecir la siguiente acción
        dist = torch.distributions.Categorical(probs=probs)
        city = dist.sample()
        log_prob = dist.log_prob(city)

        # Aplicar la acción
        action = TSP_Action(city.item())
        reward = env.state_transition(state, action)

        # Acumular variables
        p.append(log_prob)
        r.append(reward)

    log_probs = torch.stack(p)
    rewards = torch.tensor(r, dtype=torch.float32) 

    return log_probs, rewards, state.cost


def compute_returns(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0
    for reward in reversed(rewards):
        accumulated_return = reward + gamma * accumulated_return
        discounted_returns.insert(0, accumulated_return)
    return torch.tensor(discounted_returns, dtype=torch.float32)


def reinforce(model, instance, eps=10, lr=1e-4, gamma=0.99):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    cost_arr = []
    loss_arr = []

    for i in range(eps):
        # Entrenar y actualizar parámetros por episodio
        log_probs, rewards, cost = generate_rl_data(model, instance)
        returns = compute_returns(rewards, gamma)
        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost_arr.append(cost)
        loss_arr.append(loss.detach().item())

        if i % 10 == 0:
            print(f'Episodio: {i}\tCosto del tour: {cost:.2f}\tLoss: {loss:.2f}')

    return cost_arr, loss_arr