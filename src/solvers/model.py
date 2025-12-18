from torch import nn
import torch
import numpy as np
from ..TSP import TSP_State
from ..data_gen import read_instances

class ModelSolver():
    def __init__(self, model: nn.Module):
        self.model = model

    def solve(self, instance_file, instance_number):
        instances = read_instances(instance_file)
        instance = instances[instance_number]

        state = TSP_State(instance)

        while not state.is_finished():
            # Datos
            x_src = torch.tensor(np.array([state.instance.city_locations]), dtype=torch.float32)
            visited = np.pad(np.array(state.tour), (0, instance.num_cities - len(state.tour)), 'constant', constant_values=-1)
            visited = torch.tensor(np.array([visited]), dtype=torch.int32)

            # Predecir la siguiente ciudad
            output = self.model.forward(x_src, visited)
            next_city = output.argmax()
            state.visit_city(next_city.item())

        return state