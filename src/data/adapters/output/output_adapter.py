from data.adapters.adapter import DataAdapter
from TSP import TSPState
from abc import abstractmethod
import numpy as np

class OutputAdapter(DataAdapter):
    def __init__(self, data_keys, max_cities):
        super().__init__(data_keys, max_cities)

    @abstractmethod
    def output_2_vec(self, state: TSPState, best_city, final_cost):
        pass