
from abc import abstractmethod
import torch

from parametrizations.parametrization import Parametrization

import general_settings
from utils.vectors import normalize_
from utils.logging import error
from utils.TOME import depth_reprojection_bound

# Point light parametrization should include: position, intensity, attenuationfrom abc import ABC, abstractmethod

class LightParametrization(Parametrization):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_light_intensities(self, location_parametrization, rig_extrinsics):
        """
        Get the intensities this light model sends towards a given surface point.

        Inputs:
            location_parametrization    LocationParametrization instance
            rig_extrinsics              3x4 torch.tensor with the camera extrinsics
            calculate_shadows           Whether or not to calculate and return the shadow masks (default True)

        Outputs:
            ray_intensities             NxLx3 torch.tensor with the intensities arriving at the surface points
            ray_directions              NxLx3 torch.tensor with the directions each of these intensities is arriving from
                                        This is a unit vector pointing away from the surface point
        """
        pass

    @abstractmethod
    def enforce_parameter_bounds(self):
        pass


class PointLight(LightParametrization):
    def initialize(self, positions=None, intensities=None, attenuations=None, rig_light=True):
        if positions is None or intensities is None or attenuations is None:
            error("PointLight improperly initialized")
        self.positions = torch.nn.Parameter(positions)          # L x 3
        self.intensities = torch.nn.Parameter(intensities)      # L x 3
        self.attenuations = torch.nn.Parameter(attenuations)    # L x D
        self.rig_light = True 
        "Whether or not this point light is physically attached to the camera rig"

    def get_light_intensities(self, location_parametrization, rig_extrinsics, light_info, calculate_shadowing=True):
        surface_points = location_parametrization.location_vector()

        if not isinstance(light_info, list):
            light_info = [light_info]
        if min(light_info) < 0:
            error("Trying to simulate an image without a light source.")
        if max(light_info) >= self.positions.shape[0]:
            error("Trying to simulate an image with an out-of-bounds light index.")

        rig_R = rig_extrinsics[:3,:3]
        rig_t = rig_extrinsics[:3,3:]
        camloc = -rig_R.T @ rig_t
        light_positions = self.positions[light_info]
        if self.rig_light:
            # apply the inverse extrinsics
            light_positions = light_positions @ rig_R + camloc.T
        
        # the lighting rays
        ray_directions = light_positions.view(1,-1,3) - surface_points.view(-1,1,3)
        ray_lengths = ray_directions.norm(dim=-1, keepdim=True)
        ray_directions = ray_directions / ray_lengths

        # take into account the length of the light rays: we decay quadratically with distance
        ray_intensities = self.intensities[light_info].view(1,-1,3) / ray_lengths.pow(2.)

        # take into account the light dependency of the LEDs (modelled as a powered cosine)
        # for this, the LED orientation is equal to the camera rig's orientation
        cosines = - ray_directions @ rig_R.T[None,:3,2:3]
        attenuations = self.attenuations
        if attenuations.shape[0] > 1:
            attenuations = attenuations[light_info]
        ray_attenuations = cosines ** attenuations[None]

        # this is where we do shadowing, if required
        if calculate_shadowing:
            with torch.no_grad():
                ctr_Rt = location_parametrization.invRt.inverse()
                ctr_P = location_parametrization.invK.inverse() @ ctr_Rt[:3]
                depth_image = location_parametrization.create_image(surface_points) @ ctr_Rt[2:3,:3].T[None] + ctr_Rt[2:3,3:].T[None]

                light_Rs = rig_R[None]
                light_ts = -light_Rs @ light_positions.view(-1,3,1)
                L = light_ts.shape[0]
                virtualH = virtualW = 4096
                light_Ks = torch.tensor(
                    [
                        [virtualW,0,virtualW/2],
                        [0,virtualW,virtualH/2],
                        [0,0,1]
                    ],
                    dtype=torch.float32,
                    device=rig_R.device
                )[None]
                light_Ps = light_Ks @ torch.cat([light_Rs.repeat(L, 1, 1), light_ts.view(-1,3,1)], dim=2)
                light_invKRs = light_Rs.inverse() @ light_Ks.inverse()
            
                bounded = depth_reprojection_bound(
                    depth_image.view(1,depth_image.shape[0], depth_image.shape[1]).repeat(L,1,1),
                    ctr_P.view(1,3,4).repeat(L,1,1),
                    light_invKRs.view(1,1,3,3).repeat(L,1,1,1),
                    light_positions.view(-1,1,3,1),
                    4096,
                    4096,
                    0.250,
                    1.500,
                    0.010
                )

                # project the points onto the light cameras
                projected = light_Ps[...,None,:3,:3] @ surface_points.view(1,-1,3,1) + light_Ps[...,None,:3,3:]
                projected_depth = projected[:, :, 2:, 0]
                projected = projected[:, :, :2, 0]/projected_depth
                projected_p = projected.clone()
                projected += 0.5
                projected[...,0] /= virtualW/2
                projected[...,1] /= virtualH/2
                projected -= 1

                zbuffer = torch.nn.functional.grid_sample(
                    bounded,
                    projected[:, None],
                    mode="bilinear",
                    align_corners=False
                )[:,0]
                shadow_mask = projected_depth >= zbuffer.view(light_positions.shape[0], -1, 1) + 0.005
                shadow_fattening = general_settings.shadow_fattening
                if shadow_fattening is not None and shadow_fattening > 0:
                    # first, scatter the shadow mask into the 'bounded' plane
                    projected_pl = projected_p.round().long()
                    projected_pl.clamp_(min=0)
                    projected_pl[...,0].clamp_(max=virtualW-1)
                    projected_pl[...,1].clamp_(max=virtualH-1)
                    shadowers = torch.zeros_like(bounded)
                    lin_idces = projected_pl[:,:,0] + projected_pl[:,:,1] * virtualW
                    shadowers.view(L,-1).scatter_(dim=1,index=lin_idces, src=shadow_mask.view(L,-1).float())
                    # then dilate only those pixels that caused occlusion
                    weights = torch.ones(1,1,2*shadow_fattening+1,2*shadow_fattening+1).to(shadowers.device)
                    shadowers_dilated = torch.nn.functional.conv2d(
                        bounded * shadowers, weights, bias=None, padding=shadow_fattening
                    )
                    dilation_weights = torch.nn.functional.conv2d(
                        shadowers, weights, bias=None, padding=shadow_fattening
                    )
                    shadowers_dilated = shadowers_dilated / torch.clamp(dilation_weights, min=1)
                    fattened_bounded = torch.min(bounded, shadowers_dilated + 1e3*(dilation_weights == 0).float())
                    # then recalculate the occlusion
                    fattened_zbuffer = torch.nn.functional.grid_sample(
                        fattened_bounded,
                        projected[:, None],
                        mode="bilinear",
                        align_corners=False
                    )[:,0]
                    fattened_shadow_mask = projected_depth >= fattened_zbuffer.view(L, -1, 1) + 0.005
                    shadow_mask = fattened_shadow_mask
                calculated_shadowing = shadow_mask.view(L,-1).transpose(0,1)
        else:
            calculated_shadowing = None

        return ray_intensities * ray_attenuations, ray_directions, calculated_shadowing

    def parameters(self):
        return [self.positions, self.intensities, self.attenuations]
    
    def serialize(self):
        return [self.positions.detach(), self.intensities.detach(), self.attenuations.detach(), self.rig_light]

    def deserialize(self, *args):
        self.positions = torch.nn.Parameter(args[0])
        self.intensities = torch.nn.Parameter(args[1])
        self.attenuations = torch.nn.Parameter(args[2])
        self.rig_light = args[3]

    def enforce_parameter_bounds(self):
        self.intensities.data.clamp_(min=0.)


def LightParametrizationFactory(name):
    valid_dict = {
        "point light": PointLight,
    }
    if name in valid_dict:
        return valid_dict[name]
    else:
        error("Light parametrization '%s' is not supported." % name)
