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

from abc import ABC, abstractmethod
import torch

from parametrizations.parametrization import Parametrization

from utils.depth_maps import depth_map_to_locations
from utils.logging import error
from utils.vectors import inner_product, cross_product, normalize, norm

from functools import lru_cache

class LocationParametrization(Parametrization):
    """
    Parametrization base class for scene point locations.
    It supports getting a set of representative scene points, as well as the
    implied depth and normal image.
    """

    @abstractmethod
    def initialize(self, depth, mask, invK, invRt):
        """
        Initialize this LocationParametrization in-place.

        Inputs:
            depth           HxW torch.tensor containing a depth image
            mask            HxW (bool) torch.tensor indicating which pixels are valid
            invK, invRt     torch.tensors containing the inverse intrinsics and extrinsics
        """
        pass
    
    @abstractmethod
    def location_vector(self):
        """
        Get the set of 3D point locations that comprise the scene.
        The size and ordering of this tensor should be consistent.

        Outputs:
            locations       Nx3 torch.tensor containing the locations.
        """
        pass

    @abstractmethod
    def create_image(self, measurements):
        """
        Yields an image with the relevant measurements filled into the correct pixels.

        Inputs:
            measurements    Nx3 torch.tensor containing the measurements
        
        Outputs:
            image           HxW torch.tensor containing the measurements in the relevant pixels
        """
        pass

    def create_vector(self, measurement_image):
        """
        Yields a vector with the relevant measurements extract from the correct pixels.

        Inputs:
            image           HxW torch.tensor containing the measurements in the relevant pixels
        
        Outputs:
            measurements    Nx3 torch.tensor containing the measurements
        """
        return 

    @lru_cache(maxsize=1)
    def implied_depth_image(self):
        """
        Returns the depth image implied by the current state of the LocationParametrization.
        
        Outputs:
            depth           HxW torch.tensor containing the depth values
        """

        location_image = self.create_image(self.location_vector())
        R = self.invRt[:3,:3].T
        t = -R @ self.invRt[:3,3:]
        depth_image = location_image @ R[2:3].T[None] + t[2:3].T[None]
        return depth_image

    @lru_cache(maxsize=1)
    def implied_depth_vector(self):
        """
        Returns a list of depth values for the current state.
        
        Outputs:
            depths          Nx1 torch.tensor containing the depth values
        """
        return self.create_vector(self.implied_depth_image())

    @lru_cache(maxsize=1)
    def implied_normal_image(self):
        """
        Returns the normal image implied by the current state of the LocationParametrization.
        
        Outputs:
            normals         HxWx3 torch.tensor containing the depth values
        """
        location_image = self.create_image(self.location_vector()) # HxWx3
        if self.mask[-1,:].sum() + self.mask[:,-1].sum() > 0:
            error("The mask should not reach the bottom and right image edges.")
        down_vectors = normalize(location_image[:-1,:-1] - location_image[1:,:-1])
        right_vectors = normalize(location_image[:-1,:-1] - location_image[:-1,1:])
        normal_image = normalize(cross_product(down_vectors, right_vectors))
        camloc = self.invRt[:3,3:]
        # make sure the normal is the one that points towards the camera
        reorientations = inner_product(
            normal_image,
            camloc.view(1,1,3) - location_image[:-1,:-1]
        ).sign()
        normal_image = normal_image * reorientations # H-1 x W-1 x 3
        # extend the normals to the edges of the mask
        local_normal_mean = normalize(torch.nn.functional.conv2d(
            normal_image.permute(2,0,1)[:,None],
            torch.ones((1,1,5,5),device=normal_image.device),
            padding=2
        )[:,0].permute(1,2,0))

        invalid_normals = norm(normal_image, keepdim=False) == 0
        replacement_mask = (self.mask[:-1,:-1] * (invalid_normals + ~self.mask[1:,:-1] + ~self.mask[:-1,1:] > 0))
        normal_image[replacement_mask] = local_normal_mean[replacement_mask]
        # for badly connected masks, some of the pixels are currently still not filled
        # fill them just with a normal pointing roughly in the right direction
        invalid_normals = norm(normal_image) == 0
        replacement_mask = invalid_normals[:,:,0] * self.mask[:-1,:-1]
        mean_normal = normalize(
            local_normal_mean.view(-1,3).sum(dim=0, keepdim=True)
        )
        normal_image[replacement_mask] = mean_normal
        normal_image = torch.cat(
            (
                normal_image,
                torch.zeros(
                    1, normal_image.shape[1], 3,
                    dtype=normal_image.dtype, device=normal_image.device
                )
            ), dim=0)
        normal_image = torch.cat(
            (
                normal_image,
                torch.zeros(
                    normal_image.shape[0], 1, 3,
                    dtype=normal_image.dtype, device=normal_image.device
                )
            ), dim=1)

        return normal_image

    @lru_cache(maxsize=1)
    def implied_normal_vector(self):
        """
        Returns a list of normal values for the current state.
        
        Outputs:
            normals         Nx3 torch.tensor containing the depth values
        """
        return self.create_vector(self.implied_normal_image())

    @abstractmethod
    def device(self):
        """
        Get the torch.device the parameters live on.

        Outputs:
            device          torch.device
        """
        return self.depth.device

    @abstractmethod
    def get_point_count(self):
        """
        Get the number of 3D points in the scene

        Outputs:
            N               python scalar
        """
        pass



class DepthMapParametrization(LocationParametrization):
    """
    LocationParametrization that represents the scene structure with a depth map.
    The intrinsics and extrinsics of this view cannot be optimized over; only the
    values of the depth map are returned from parameter_info.
    """

    def initialize(self, depth, mask, invK, invRt):
        self.depth = torch.nn.Parameter(depth)
        self.mask = (mask.view(depth.shape[:2]) > 0).to(depth.device)
        self.invK = invK.to(depth.device)
        self.invRt = invRt.to(depth.device)
        self.point_count = self.mask.sum().item()
    
    @lru_cache(maxsize=1)
    def location_vector(self):
        depth_points = depth_map_to_locations(self.depth, self.invK, self.invRt)
        return depth_points[self.mask].view(-1,3)

    def implied_depth_image(self):
        return self.depth

    def create_image(self, measurements, filler=None):
        image = torch.zeros(self.mask.shape[0], self.mask.shape[1], measurements.shape[1], device=measurements.device)
        if filler is not None:
            image.fill_(filler)
        image[self.mask] = measurements
        return image

    def create_vector(self, measurement_image):
        if measurement_image.shape[:2] == self.mask.shape[:2]:
            return measurement_image[self.mask]
        elif (
            self.mask.shape[0] == measurement_image.shape[0] + 1
            and
            self.mask.shape[1] == measurement_image.shape[1] + 1
        ):
            return measurement_image[self.mask[:-1,:-1]]
        else:
            error("Mask shape does not fit the measurement_image shape.")

    def get_point_count(self):
        return self.point_count
    
    def device(self):
        return self.depth.device

    def parameter_info(self):
        return {
            "locations": [self.depth, 1e-4, None],
        }

    def serialize(self):
        return self.depth.detach(), self.invK, self.invRt, self.mask

    def deserialize(self, *args):
        depth, invK, invRt, mask = args
        self.depth = torch.nn.Parameter(depth)
        self.invK = invK
        self.invRt = invRt
        self.mask = mask
        self.point_count = self.mask.sum().item()

    def enforce_parameter_bounds(self):
        self.depth.data.clamp_(min=0.)

class PlaneParametrization(LocationParametrization):
    """
    LocationParametrization that represents the scene structure with a plane equation.
    """

    def initialize(self, depth, mask, invK, invRt):
        self.invK = invK.to(depth.device)
        self.invRt = invRt.to(depth.device)
        self.mask = (mask.view(depth.shape[:2]) > 0).to(depth.device)

        world_points = depth_map_to_locations(depth, invK, invRt)[mask].view(-1,3)
        self.p_plane = torch.nn.Parameter(world_points.pinverse().sum(dim=1, keepdim=False))

        xs = torch.arange(0, W).float().reshape(1,W,1).to(depth.device).expand(H,W,1)
        ys = torch.arange(0, H).float().reshape(H,1,1).to(depth.device).expand(H,W,1)
        zs = torch.ones(1,1,1).to(depth.device).expand(H,W,1)
        self._camera_rays = torch.cat((xs, ys, zs), dim=2) @ self.invRt[:3,:3].T[None] @ self.invK[:3,:3].T
        self._camera_rays = self._camera_rays[self.mask].view(-1,3)
        self.point_count = self.mask.sum().item()

    @lru_cache(maxsize=1)
    def location_vector(self):
        camera_rays = self._camera_rays  # Nx3
        camera_loc = self.invRt[:3,3:].T # 1x3
        p_plane = self.p_plane.view(3,1)
        ray_lengths = (1-camera_loc @ p_plane) / (camera_rays @ p_plane) # Nx1
        locations = camera_loc + ray_lengths * camera_rays
        return locations

    def create_image(self, measurements):
        image = torch.zeros(self.mask.shape[0], self.mask.shape[1], measurements.shape[1], device=measurements.device)
        image[self.mask] = measurements
        return image

    def create_vector(self, measurement_image):
        if measurement_image.shape[:2] == self.mask.shape[:2]:
            return measurement_image[self.mask]
        elif (
            self.mask.shape[0] == measurement_image.shape[0] + 1
            and
            self.mask.shape[1] == measurement_image.shape[1] + 1
        ):
            return measurement_image[self.mask[:-1,:-1]]
        else:
            error("Mask shape does not fit the measurement_image shape.")

    def get_point_count(self):
        return self.point_count
    
    def device(self):
        return self.p_plane.device

    def parameters(self):
        return {
            "locations": [self.p_plane, 1e-4, None], 
        }

    def serialize(self):
        return self.p_plane.detach(), self.invK, self.invRt, self.mask

    def deserialize(self, *args):
        p_plane, invK, invRt, mask = args
        self.p_plane = torch.nn.Parameter(p_plane)
        self.invK = invK
        self.invRt = invRt
        self.mask = mask
        self.point_count = self.mask.sum().item()

    def enforce_parameter_bounds(self):
        pass

def LocationParametrizationFactory(name):
    valid_dict = {
        "depth map": DepthMapParametrization,
        "plane": PlaneParametrization,
    }
    if name in valid_dict:
        return valid_dict[name]
    else:
        error("Location parametrization '%s' is not supported." % name)
