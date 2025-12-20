from abc import ABC, abstractmethod
import torch.nn as nn

class Transformer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, x_src):
        """
        x_src: entrada cruda del encoder
        returns:
            memory: representación latente
        """
        pass

    @abstractmethod
    def decode(self, memory, visited):
        """
        memory: salida del encoder
        visited: estado del decoder
        returns:
            probs / logits
        """
        pass

    def forward(self, x_src, visited):
        """
        Forward genérico: encoder una vez, decoder una vez
        """
        memory = self.encode(x_src)
        return self.decode(memory, visited)