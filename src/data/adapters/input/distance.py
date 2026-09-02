from data.adapters.input.input_adapter import InputAdapter
from TSP import TSPState
import numpy as np

class DistanceInputAdapter(InputAdapter):
    def __init__(self, max_cities=100):
        super().__init__({
            "coords": np.float32,
            "distances": np.float32,
            "visited": np.int32,
            "num_cities": np.int32
        }, max_cities)

    def input_2_vec(self, state: TSPState):
        coords = np.array(state.instance.city_locations, dtype=np.float32)
        
        # Extraer y castear explícitamente a float32
        distances = np.array(state.instance.distance_matrix, dtype=np.float32)
        
        num_cities_actual = len(coords)
        pad_len = self.max_cities - num_cities_actual
        
        if pad_len > 0:
            # Padding para coordenadas (matriz N x 2)
            coords = np.pad(coords, ((0, pad_len), (0, 0)), 'constant', constant_values=0.0)
            
            # Padding para distancias (matriz N x N)
            distances = np.pad(distances, ((0, pad_len), (0, pad_len)), 'constant', constant_values=0.0)
            
        # Padding para ciudades visitadas (vector N)
        visited = np.pad(np.array(state.tour), (0, self.max_cities - len(state.tour)), 'constant', constant_values=-1)

        return coords, distances, visited, state.instance.num_cities