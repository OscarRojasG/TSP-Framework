import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

class TSPInstance:
    def __init__(self, city_locations):
        self.city_locations = city_locations
        self.num_cities = len(city_locations)
        self.distance_matrix = np.sqrt(np.sum((city_locations[:, np.newaxis, :] -  city_locations[np.newaxis, :, :]) ** 2, axis=-1))
        self.sparse_adj_matrix = None

    def compute_sparse_adj(self):
        adj = np.zeros((self.num_cities, self.num_cities), dtype=np.bool_)
        
        if self.num_cities >= 3:
            tri = Delaunay(self.city_locations)
            
            # Extracción vectorizada de aristas (mucho más rápido que los for loops)
            # Un simplex tiene forma [u, v, w]. Sacamos los pares (u,v), (v,w), (u,w)
            edges = np.vstack((
                tri.simplices[:, [0, 1]],
                tri.simplices[:, [1, 2]],
                tri.simplices[:, [0, 2]]
            ))
            
            # Asignamos True en ambas direcciones
            adj[edges[:, 0], edges[:, 1]] = True
            adj[edges[:, 1], edges[:, 0]] = True
        else:
            # Para menos de 3 ciudades, forzamos conexión total
            adj[:, :] = True
            
        # Conexión consigo mismo (importante para la atención)
        np.fill_diagonal(adj, True)
        
        self.sparse_adj_matrix = adj

    def plot(self):
        """Dibuja el grafo completamente conectado (Atención densa O(N^2))"""
        plt.figure(figsize=(7, 7))
        
        # Iteramos solo i < j para no dibujar la misma línea dos veces
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                plt.plot(
                    [self.city_locations[i, 0], self.city_locations[j, 0]],
                    [self.city_locations[i, 1], self.city_locations[j, 1]],
                    color='gray', linestyle='-', linewidth=0.3, alpha=0.3
                )
                
        plt.scatter(self.city_locations[:, 0], self.city_locations[:, 1], c='black', s=25, zorder=5)
        
        plt.title(f"Grafo Completo ({self.num_cities} ciudades)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_sparse_graph(self, show_circumcircles=False):
        """Dibuja el grafo esparsificado (Triangulación de Delaunay)"""
        if self.sparse_adj_matrix is None:
            self.compute_sparse_adj()
            
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        
        # 1. Dibujamos las aristas esparsificadas
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                if self.sparse_adj_matrix[i, j]:
                    ax.plot(
                        [self.city_locations[i, 0], self.city_locations[j, 0]],
                        [self.city_locations[i, 1], self.city_locations[j, 1]],
                        color='#1f77b4', linestyle='-', linewidth=1.0, alpha=0.8
                    )
                    
        # 2. Dibujamos las ciudades (nodos)
        ax.scatter(self.city_locations[:, 0], self.city_locations[:, 1], c='black', s=25, zorder=5)
        
        # Forzamos proporciones iguales
        ax.set_aspect('equal') 
        
        # Congelamos los límites actuales del gráfico para que los círculos no lo expandan
        ax.autoscale(False)
        
        # 3. Dibujamos las circunferencias de Delaunay si se solicita
        if show_circumcircles and self.num_cities >= 3:
            tri = Delaunay(self.city_locations)
            for simplex in tri.simplices:
                A, B, C = self.city_locations[simplex]
                
                # Fórmula geométrica para hallar el centro (Ux, Uy)
                D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
                if D != 0:
                    Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
                    Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
                    R = np.sqrt((A[0] - Ux)**2 + (A[1] - Uy)**2) # Radio
                    
                    circle = plt.Circle((Ux, Uy), R, color='red', fill=False, linestyle=':', alpha=0.3)
                    ax.add_patch(circle)
        
        plt.title(f"Grafo Esparsificado - Delaunay ({self.num_cities} ciudades)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

class TSPState():
    def __init__(self, instance: TSPInstance):
        self.instance = instance
        self.num_cities = instance.num_cities
        self.visited = [False] * self.num_cities
        self.current_city = 0
        self.tour = [self.current_city]
        self.visited[self.current_city] = True
        self.cost = 0.0

    def visit_city(self, city_index):
        if not self.visited[city_index]:
            last_city = self.current_city
            self.current_city = city_index
            self.tour.append(city_index)
            self.visited[city_index] = True
            self.cost += self.instance.distance_matrix[last_city][city_index]

    def is_finished(self):
        return all(self.visited)

    def get_total_cost(self):
        # Añadir la distancia de regreso a la ciudad inicial para completar el tour
        if self.is_finished():
            return self.cost + self.instance.distance_matrix[self.current_city][self.tour[0]]
        return self.cost
    
    def plot(self):
        # Separar las coordenadas x e y de los puntos
        x = [self.instance.city_locations[i][0] for i in self.tour]
        y = [self.instance.city_locations[i][1] for i in self.tour]

        # Agregar el primer punto al final para cerrar el tour
        if self.is_finished():
            x.append(x[0])
            y.append(y[0])

        # Graficar los puntos
        plt.scatter(x, y)

        # Graficar las líneas del tour
        plt.plot(x, y)

        # Agregar títulos y etiquetas si es necesario
        plt.title("Tour")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")

        # Mostrar el gráfico
        plt.show()