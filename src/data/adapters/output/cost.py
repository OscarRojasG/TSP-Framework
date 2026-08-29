from data.adapters.output.output_adapter import OutputAdapter
from TSP import TSPState
import numpy as np

class CostOutputAdapter(OutputAdapter):
    def __init__(self, max_cities=100):
        super().__init__({
            "cost": np.float32
        }, max_cities)

    def output_2_vec(self, state: TSPState, best_city, final_cost):
        return final_cost