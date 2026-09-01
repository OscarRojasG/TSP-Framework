from data.adapters.input.input_adapter import InputAdapter
from TSP import TSPState
import numpy as np

class DefaultInputAdapter(InputAdapter):
    def __init__(self, max_cities=100):
        super().__init__({
            "coords": np.float32,
            "visited": np.int32,
            "num_cities": np.int32
        }, max_cities)

    def input_2_vec(self, state: TSPState):
        coords = np.array(state.instance.city_locations, dtype=np.float32)
        num_cities_actual = len(coords)
        
        # Padding para coordenadas (matriz N x 2)
        pad_len = self.max_cities - num_cities_actual
        if pad_len > 0:
            coords = np.pad(coords, ((0, pad_len), (0, 0)), 'constant', constant_values=0.0)
            
        # Padding para ciudades visitadas (vector N)
        visited = np.pad(np.array(state.tour), (0, self.max_cities - len(state.tour)), 'constant', constant_values=-1)

        return coords, visited, state.instance.num_cities