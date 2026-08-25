import numpy as np
from abc import ABC, abstractmethod

class DataAdapter(ABC):
    def __init__(self, data_keys):
        super().__init__()
        self.data = {
            k: [] for k in data_keys
        }
        self.data_keys = data_keys

    @abstractmethod
    def add(self, data):
        pass

    def get(self) -> dict:
        return {
            k: np.stack(v, dtype=self.data_keys[k]) for k, v in self.data.items()
        }

    def count(self):
        return len(self.data[list(self.data.keys())[0]])