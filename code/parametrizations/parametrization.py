import torch
from abc import abstractmethod, ABC

class Parametrization(ABC):
    """
    Abstract base class for all parametrizations. A Parametrization should be able to:
        - be called for the underlying value,
        - be serialized to/deserialized from a torch.tensor list,
        - return relevant torch.nn.Parameter info,
        - enforce bounds on its underlying parameters.

    Additionally, it can use functools.lru_cache to cache function results. These
    caches can be assumed to be cleared whenever the underlying parameters have changed.
    """

    @abstractmethod
    def parameter_info(self):
        """
        Get information on the parameters of this module.
        This includes the underlying torch.nn.Parameters, their suggested learning rate,
        and optionally a visualization function reducing the parameter to a few scalars
        that can be plotted.

        Returns:
            parameter_info_dict         Dictionary containing, for each parameter:
                parameter_name: [torch.nn.Parameter, learning rate, visualization lambda]
        """
        pass

    @abstractmethod
    def serialize(self):
        """
        Get a list of torch.tensors from which the current state of this object can be
        restored.

        Returns:
            state                       List of torch.tensors
        """
        pass

    @abstractmethod
    def deserialize(self, *args):
        """
        Restores the state of this object from a tensor list, as returned by serialize().

        Inputs:
            state                       List of torch.tensors
        """
        pass

    @abstractmethod
    def enforce_parameter_bounds(self):
        """
        Enforce the bounds on the parameters of this Parametrization objects.
        These constraints should be applied to the underlying .data field.
        As such, this should only be called when the Parametrization is no longer
        part of a relevant computation graph.
        """
        pass

    def clear_cache(self):
        """
        Clear the functool caches on this object. This should be done whenever the
        parameter values change, but only when the Parametrization is no longer
        part of a relevant computation graph.
        """
        for entry in dir(self):
            entry = self.__getattribute__(entry)
            if hasattr(entry, "cache_clear"):
                entry.cache_clear()
