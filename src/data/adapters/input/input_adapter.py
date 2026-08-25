from data.adapters.adapter import DataAdapter
from TSP import TSPState
from abc import abstractmethod

class InputAdapter(DataAdapter):
    def __init__(self, data_keys, max_cities):
        super().__init__(data_keys, max_cities)

    @abstractmethod
    def input_2_vec(self, state: TSPState):
        pass