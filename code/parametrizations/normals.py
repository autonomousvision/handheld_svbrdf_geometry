from abc import abstractmethod
import torch
import numpy as np

from utils.logging import error
from parametrizations.parametrization import Parametrization
from utils.vectors import normalize, cross_product

class NormalParametrization(Parametrization):
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
    It parametrizes over an extra rotation of the current normal estimates
    To avert gimbal lock, it constantly reparametrizes to zero delta angles
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
        return [self.base_normals, self.phis.detach(), self.thetas.detach()]

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
    Normals are calculated straight from the depth.
    This module has no parameters. Instead, normals are backpropagated
    to the depth parameters.
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
