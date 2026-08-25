from abc import ABC, abstractmethod
import torch.nn as nn

class Transformer(nn.Module, ABC):
    def __init__(self, **hyperparams):
        super(Transformer, self).__init__()
        self.hyperparams = hyperparams

    @abstractmethod
    def encode(self, *inputs):
        """
        inputs: todos los tensores crudos provenientes del dataset (input)
        returns:
            memory: representación latente (salida del encoder)
        """
        pass

    @abstractmethod
    def decode(self, memory, *inputs):
        """
        memory: salida del encoder
        inputs: todos los tensores crudos provenientes del dataset (input)
        """
        pass

    def forward(self, *inputs):
        """
        Forward genérico: 
        1. Pasa todos los inputs al encoder para obtener la memoria.
        2. Pasa la memoria y todos los inputs al decoder.
        """
        memory = self.encode(*inputs)
        return self.decode(memory, *inputs)