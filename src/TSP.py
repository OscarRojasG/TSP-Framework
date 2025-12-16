import numpy as np

class TSP_Instance:
    def __init__(self, city_locations):
        self.city_locations = city_locations
        self.num_cities = len(city_locations)
        self.distance_matrix = np.sqrt(np.sum((city_locations[:, np.newaxis, :] -  city_locations[np.newaxis, :, :]) ** 2, axis=-1))

class TSP_State():
    def __init__(self, instance: TSP_Instance):
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