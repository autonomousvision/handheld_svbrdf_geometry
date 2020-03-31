"""
Copyright (c) 2020 Simon Donné, Max Planck Institute for Intelligent Systems, Tuebingen, Germany

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

from abc import abstractmethod
import torch
import numpy as np

from utils.logging import error
from parametrizations.parametrization import Parametrization
from utils.vectors import normalize, cross_product

class NormalParametrization(Parametrization):
    """
    Parametrization subclass representing the normals for all scene points.
    It depends on the relevant LocationParametrization for initialization.
    """

    def __init__(self, location_parametrization=None):
        super().__init__()
        if location_parametrization is None:
            error("NormalParametrization should get created with a LocationParametrization")
        self.location_parametrization = location_parametrization

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def normals(self):
        pass

    @abstractmethod
    def enforce_parameter_bounds(self):
        pass


class PerPointNormals(NormalParametrization):
    """
    A single, completely independent normal for every point in the scene.
    Optimization over these normals is done by rotations around the
    Y and Z axis. To prevent gimbal lock, these rotations are applied to the
    underlying normals in enforce_parameter_bounds, and subsequently set to zero.
    """
    def initialize(self, base_normals):
        nr_points = self.location_parametrization.get_point_count()
        device = self.location_parametrization.device()
        self.base_normals = base_normals.detach()
        self.phis = torch.nn.Parameter(torch.zeros(nr_points, device=device))
        self.thetas = torch.nn.Parameter(torch.zeros(nr_points, device=device))

    def normals(self):
        sinphi = self.phis.sin()
        cosphi = self.phis.cos()
        sintheta = self.thetas.sin()
        costheta = self.thetas.cos()
        O = torch.zeros_like(sinphi)
        I = torch.ones_like(sinphi)
        Rz = torch.stack(
            [
                cosphi, -sinphi, O,
                sinphi, cosphi, O,
                O, O, I
            ],
            dim=1
        ).view(-1, 3, 3)
        Ry = torch.stack(
            [
                costheta, O, sintheta,
                O, I, O,
                -sintheta, O, costheta
            ],
            dim=1
        ).view(-1, 3, 3)
        normals = Ry @ Rz @ self.base_normals.view(-1,3,1)
        return normals.view(-1,3)
    
    def parameter_info(self):
        return {
            "normals": [[self.phis, self.thetas], np.pi/360, None],
        }

    def serialize(self):
        return [
            self.base_normals.clone(),
            self.phis.detach().clone(),
            self.thetas.detach().clone()
        ]

    def deserialize(self, *args):
        base_normals, phis, thetas = args
        self.base_normals = base_normals
        self.phis = torch.nn.Parameter(phis)
        self.thetas = torch.nn.Parameter(thetas)

    def enforce_parameter_bounds(self):
        self.base_normals = normalize(self.normals().detach())
        self.phis.data.zero_()
        self.thetas.data.zero_()

class HardLinkedNormals(NormalParametrization):
    """
    Normals for all scene points are those implied by the LocationParametrization.
    This module has no parameters. Instead, gradients are backpropagated to the
    depth parameters automatically.
    """

    def initialize(self):
        # we have no parameters
        pass

    def normals(self):
        return self.location_parametrization.implied_normal_vector()
    
    def parameter_info(self):
        # we have no parameters
        return {}

    def serialize(self):
        # we have no parameters
        pass

    def deserialize(self, *args):
        pass

    def enforce_parameter_bounds(self):
        # we have no parameters
        pass


def NormalParametrizationFactory(name):
    valid_dict = {
        "per point": PerPointNormals,
        "hard linked": HardLinkedNormals,
    }
    if name in valid_dict:
        return valid_dict[name]
    else:
        error("Normal parametrization '%s' is not supported." % name)
