from data.adapters.input.input_adapter import InputAdapter
from TSP import TSPState
import numpy as np

class SparseInputAdapter(InputAdapter):
    def __init__(self, max_cities=100):
        super().__init__({
            "coords": np.float32,
            "visited": np.int32,
            "num_cities": np.int32,
            "adj_matrix": np.bool_  
        }, max_cities)

    def input_2_vec(self, state: TSPState):
        coords = np.array(state.instance.city_locations, dtype=np.float32)
        num_cities_actual = len(coords)
        
        # 1. --- MATRIZ DE ADYACENCIA PRECALCULADA ---
        adj = state.instance.sparse_adj_matrix
        
        pad_len = self.max_cities - num_cities_actual
        
        # 2. --- PADDING MATRIZ ADYACENCIA ---
        # Hacemos padding en ambas dimensiones (N x N -> Max x Max) con False
        if pad_len > 0:
            adj = np.pad(adj, ((0, pad_len), (0, pad_len)), mode='constant', constant_values=False)
            
        # 3. --- PADDING COORDENADAS ---
        if pad_len > 0:
            coords = np.pad(coords, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
            
        # 4. --- PADDING VISITADAS ---
        tour_array = np.array(state.tour, dtype=np.int32)
        visited = np.pad(tour_array, (0, self.max_cities - len(tour_array)), mode='constant', constant_values=-1)

        # Retornamos los 4 elementos en el orden exacto del __init__
        return coords, visited, state.instance.num_cities, adj