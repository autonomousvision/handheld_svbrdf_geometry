from abc import ABC, abstractmethod
from parametrizations.parametrization import Parametrization

from utils.quaternions import R_to_quaternion, quaternion_to_R
from utils.vectors import normalize_
import torch

from functools import lru_cache

class PoseParametrization(Parametrization):
    """
    Parametrization subclass encoding a set of camera poses.
    """

    @abstractmethod
    def initialize(self, Rs, ts):
        pass

    @abstractmethod
    def Rts(self):
        pass

    @lru_cache(maxsize=1)
    def invRts(self, index=None):
        Rts = self.Rts()
        invRs = Rts[:,:3,:3].transpose(-1,-2)
        invts = -invRs @ Rts[:,:3,3:]
        invRts = torch.cat((invRs, invts), dim=-1)
        if index is None:
            return invRts
        else:
            return invRts.index_select(dim=0, index=index)

    def camlocs(self, index=None):
        return self.invRts(index)[...,:3,3:].contiguous()

    @abstractmethod
    def enforce_parameter_bounds(self):
        pass


class QuaternionPoseParametrization(PoseParametrization):
    """
    PoseParametrization that encodes orientations as quaternions, and locations as straight vectors.
    """
    def initialize(self, Rs, ts):
        self.qs = torch.nn.Parameter(R_to_quaternion(Rs).view(-1,4).detach().clone())
        self.ts = torch.nn.Parameter(ts.view(-1,3,1))

    @lru_cache(maxsize=1)
    def Rts(self, index=None):
        Rts = torch.cat((quaternion_to_R(self.qs), self.ts), dim=-1)
        if index is None:
            return Rts
        else:
            return Rts.index_select(dim=0, index=index)

    def parameter_info(self):
        return {
            "observation_poses": [[self.qs, self.ts], 1e-4, None]
        }

    def serialize(self):
        return [self.qs.detach(), self.ts.detach()]

    def deserialize(self, *args):
        self.qs = torch.nn.Parameter(args[0])
        self.ts = torch.nn.Parameter(args[1])

    def enforce_parameter_bounds(self):
        normalize_(self.qs)

def PoseParametrizationFactory(name):
    valid_dict = {
        "quaternion": QuaternionPoseParametrization,
    }
    if name in valid_dict:
        return valid_dict[name]
    else:
        error("Pose parametrization '%s' is not supported." % name)
