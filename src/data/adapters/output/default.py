from data.adapters.output.output_adapter import OutputAdapter
from TSP import TSPState
import numpy as np

class DefaultOutputAdapter(OutputAdapter):
    def __init__(self, max_cities=100):
        super().__init__({
            "Y": np.int32
        }, max_cities)

    def output_2_vec(self, state: TSPState, best_city, final_cost):
        Y = np.zeros(self.max_cities, dtype=np.int32)
        Y[best_city] = 1

        return Y