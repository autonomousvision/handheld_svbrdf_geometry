import torch
from abc import abstractmethod

class Parametrization:
    @abstractmethod
    def parameter_info(self):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def deserialize(self, *args):
        pass

    @abstractmethod
    def enforce_parameter_bounds(self):
        pass

    def clear_cache(self):
        for entry in dir(self):
            entry = self.__getattribute__(entry)
            if hasattr(entry, "cache_clear"):
                entry.cache_clear()
