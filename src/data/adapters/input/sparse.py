from data.adapters.input.input_adapter import InputAdapter
from TSP import TSPState
import numpy as np
from scipy.spatial import Delaunay

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
        
        # 1. --- GRAFO DE DELAUNAY (Matriz de Adyacencia) ---
        adj = np.zeros((self.max_cities, self.max_cities), dtype=np.bool_)
        
        if num_cities_actual >= 3:
            tri = Delaunay(coords)
            for simplex in tri.simplices:
                # Un simplex en 2D es un triángulo (3 nodos). Conectamos todos con todos.
                for i in range(3):
                    for j in range(i+1, 3):
                        u, v = simplex[i], simplex[j]
                        adj[u, v] = True
                        adj[v, u] = True
        else:
            # Para menos de 3 ciudades, forzamos conexión total
            adj[:num_cities_actual, :num_cities_actual] = True
            
        # Conexión consigo mismo (fundamental para la atención del Transformer)
        np.fill_diagonal(adj, True)
        
        # 2. --- PADDING COORDENADAS ---
        pad_len = self.max_cities - num_cities_actual
        if pad_len > 0:
            coords = np.pad(coords, ((0, pad_len), (0, 0)), 'constant', constant_values=0.0)
            
        # 3. --- PADDING VISITADAS ---
        visited = np.pad(np.array(state.tour), (0, self.max_cities - len(state.tour)), 'constant', constant_values=-1)

        # Retornamos los 4 elementos en el orden exacto del __init__
        return coords, visited, state.instance.num_cities, adj