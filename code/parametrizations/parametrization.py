import torch
from abc import abstractmethod

class Parametrization(torch.nn.Module):
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def deserialize(self, *args):
        pass
