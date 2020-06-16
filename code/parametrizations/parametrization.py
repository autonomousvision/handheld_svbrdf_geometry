"""
Copyright (c) 2020 Autonomous Vision Group (AVG), Max Planck Institute for Intelligent Systems, Tuebingen, Germany

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
