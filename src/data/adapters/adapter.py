import numpy as np
from abc import ABC, abstractmethod

class DataAdapter(ABC):
    def __init__(self, data_keys, max_cities):
        self.data = {k: [] for k in data_keys}
        self.data_keys = data_keys
        self.max_cities = max_cities

    def add(self, values):
        """
        Recibe una tupla con los valores retornados por input_2_vec o output_2_vec
        y los asigna a sus llaves correspondientes.
        """
        # Si values no es una tupla (ej. retorna solo 1 arreglo), lo convertimos
        if not isinstance(values, tuple):
            values = (values,)
            
        for key, val in zip(self.data_keys.keys(), values):
            self.data[key].append(val)

    def get(self) -> dict:
        return {
            k: np.stack(v, dtype=self.data_keys[k]) for k, v in self.data.items()
        }

    def count(self):
        return len(self.data[list(self.data.keys())[0]])