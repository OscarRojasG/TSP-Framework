from ..TSP import TSP_State, TSP_Instance

class TSP_Action:
    def __init__(self, city):
        self.city = city

class TSP_Environment:
    @staticmethod
    def initial_state(instance: TSP_Instance) -> TSP_State:
        return TSP_State(instance)

    @staticmethod
    def state_transition(state: TSP_State, action: TSP_Action):
        reward = state.instance.distance_matrix[state.current_city][action.city]
        state.visit_city(action.city)
        return -reward