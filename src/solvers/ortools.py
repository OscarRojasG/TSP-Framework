from TSP import TSPState, TSPInstance
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from solvers.solver import Solver

class ORToolsSolver(Solver):
    def __init__(self, scale_factor=10000, start_node=None, end_node=None, time_limit=0):
        super().__init__("ORToolsSolver")
        self.scale_factor = scale_factor
        self.start_node = start_node
        self.end_node = end_node
        self.time_limit = time_limit

    def _create_data_model(self, distance_matrix):
        """Almacena los datos del problema."""
        data = {}
        data["distance_matrix"] = distance_matrix
        
        # OR-Tools trabaja con enteros, escalamos las distancias flotantes
        data["scaled_distance_matrix"] = [
            [int(dist * self.scale_factor) for dist in row] 
            for row in data["distance_matrix"]
        ]
        
        data["num_vehicles"] = 1
        data["starts"] = [0]
        data["ends"] = [0]
        
        if self.start_node is not None:
            data["starts"] = [self.start_node]
        if self.end_node is not None:
            data["ends"] = [self.end_node]

        return data

    def solve_instance(self, instance: TSPInstance):
        """Resuelve una instancia de TSP usando Google OR-Tools."""
        data = self._create_data_model(instance.distance_matrix)

        # Crea el modelo de enrutamiento
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["starts"], data["ends"]
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Devuelve la distancia entre los dos nodos."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["scaled_distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Configura parámetros de búsqueda
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        if self.time_limit != 0:
            search_parameters.time_limit.seconds = self.time_limit

        # Resuelve el problema
        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            print("No se encontró una solución.")
            return None

        # Reconstruye la ruta visitada
        visited = []
        index = routing.Start(0)
        visited.append(manager.IndexToNode(index))
        
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            visited.append(manager.IndexToNode(index))

        # Reconstruimos el estado final
        sol_state = TSPState(instance)
        for city in visited:
            sol_state.visit_city(city)

        return sol_state