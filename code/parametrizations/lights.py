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

from parametrizations.parametrization import Parametrization

import general_settings
from utils.vectors import normalize_
from utils.logging import error
from utils.TOME import depth_reprojection_bound

# Point light parametrization should include: position, intensity, attenuationfrom abc import ABC, abstractmethod

class LightParametrization(Parametrization):
    """
    Parametrization representing a light source. It should support the calculate of light intensities incident on a given
    scene point, including the shadow modeling that may or may not require.
    """

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_light_intensities(self, location_parametrization, rig_extrinsics, calculate_shadowing=True):
        """
        Get the intensities this light model sends towards a given surface point, optionally taking into account shadowing.

        Inputs:
            location_parametrization    LocationParametrization instance
            rig_extrinsics              3x4 torch.tensor with the camera extrinsics
            calculate_shadows           Whether or not to calculate and return the shadow masks (default True)

        Outputs:
            ray_intensities             CxNx3 torch.tensor with the intensities arriving at the surface points.
            ray_directions              CxNx3 torch.tensor with the directions each of these intensities is arriving from
                                            This is a unit vector pointing away from the surface point.
            calculated_shadowing        CxN torch.tensor indicating for each surface point whether it is obscured
                                            from this light source.
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

    def get_light_intensities(self, location_parametrization, rig_extrinsics, light_infos, calculate_shadowing=True):
        surface_points = location_parametrization.location_vector()

        rig_R = rig_extrinsics[...,:3,:3]
        rig_invR = rig_R.transpose(-2,-1)
        rig_t = rig_extrinsics[...,:3,3:]
        camlocs = - rig_invR @ rig_t
        light_positions = self.positions.index_select(dim=0, index=light_infos).view(-1,3,1)
        if self.rig_light:
            # apply the inverse extrinsics
            light_positions = rig_invR @ light_positions + camlocs
        
        # the lighting rays
        ray_directions = light_positions.view(-1,1,3) - surface_points.view(1,-1,3) # C x N x 3
        ray_lengths = ray_directions.norm(dim=-1, keepdim=True)
        ray_directions = ray_directions / ray_lengths

        # take into account the length of the light rays: we decay quadratically with distance
        ray_intensities = self.intensities.index_select(dim=0, index=light_infos).view(-1,1,3) / ray_lengths.pow(2.)

        # take into account the light dependency of the LEDs (modelled as a powered cosine)
        # for this, the LED orientation is equal to the camera rig's orientation
        cosines = - ray_directions @ rig_invR[...,:3,2:3]
        attenuations = self.attenuations
        if attenuations.shape[0] > 1:
            attenuations = attenuations.index_select(dim=0, index=light_infos)
        ray_attenuations = cosines ** attenuations.view(-1,1,3)

        # this is where we do shadowing, if required
        if calculate_shadowing:
            with torch.no_grad():
                ctr_Rt = location_parametrization.invRt.inverse()
                ctr_P = location_parametrization.invK.inverse() @ ctr_Rt[:3]
                depth_image = location_parametrization.create_image(surface_points) @ ctr_Rt[2:3,:3].T[None] + ctr_Rt[2:3,3:].T[None]
                depth_image = depth_image.view(1,*depth_image.shape[:2])

                light_Rs = rig_R
                light_ts = -light_Rs @ light_positions
                C = light_ts.shape[0]
                virtualH = virtualW = 2048
                light_Ks = torch.tensor(
                    [
                        [virtualW,0,virtualW/2],
                        [0,virtualW,virtualH/2],
                        [0,0,1]
                    ],
                    dtype=torch.float32,
                    device=rig_R.device
                ).repeat(light_Rs.shape[0], 1, 1)
                light_Ps = light_Ks @ torch.cat([light_Rs, light_ts], dim=2)

                # project the points onto the light cameras
                # to make sure they are somewhat well-centered: scale focal distances and translate optical centers
                projected = light_Ps[...,None,:3,:3] @ surface_points.view(1,-1,3,1) + light_Ps[...,None,:3,3:]
                projected_depth = projected[:, :, 2:, 0]
                projected = projected[:, :, :2, 0]/projected_depth
                scales = projected.max(dim=1,keepdim=True)[0] - projected.min(dim=1,keepdim=True)[0]
                scales[...,:1] /= virtualW * 0.98
                scales[...,1:] /= virtualH * 0.98
                scales = scales.max(dim=-1, keepdim=True)[0]
                light_Ks[:,:2,:2] /= scales
                light_Ps = light_Ks @ torch.cat([light_Rs, light_ts], dim=2)
                projected = light_Ps[...,None,:3,:3] @ surface_points.view(1,-1,3,1) + light_Ps[...,None,:3,3:]
                projected_depth = projected[:, :, 2:, 0]
                projected = projected[:, :, :2, 0]/projected_depth
                offsets = projected.min(dim=1,keepdim=True)[0]
                light_Ks[:,0,2] -= offsets[:,0,0] - virtualW / 100
                light_Ks[:,1,2] -= offsets[:,0,1] - virtualH / 100
                light_Ps = light_Ks @ torch.cat([light_Rs, light_ts], dim=2)

                light_invKRs = rig_invR @ light_Ks.inverse()
                bounded = depth_reprojection_bound(
                    depth_image,
                    ctr_P.view(1,3,4),
                    light_invKRs.view(1,-1,3,3),
                    light_positions.view(1,-1,3,1),
                    virtualH,
                    virtualW,
                    0.250,
                    1.500,
                    0.010
                ).view(C, 1, virtualH, virtualW)

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
                ).view(C, -1, 1)
                shadow_mask = projected_depth >= zbuffer + 0.005
                shadow_fattening = general_settings.shadow_fattening
                if shadow_fattening is not None and shadow_fattening > 0:
                    # first, scatter the shadow mask into the 'bounded' plane
                    projected_pl = projected_p.round().long()
                    projected_pl.clamp_(min=0)
                    projected_pl[...,0].clamp_(max=virtualW-1)
                    projected_pl[...,1].clamp_(max=virtualH-1)
                    shadowers = torch.zeros_like(bounded)
                    lin_idces = projected_pl[...,0] + projected_pl[...,1] * virtualW
                    shadowers.view(C,-1).scatter_(dim=1,index=lin_idces, src=shadow_mask.view(C,-1).float())
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
                    ).view(C, -1, 1)
                    fattened_shadow_mask = projected_depth >= fattened_zbuffer + 0.005
                    shadow_mask = fattened_shadow_mask
                calculated_shadowing = shadow_mask.view(C,-1)
        else:
            calculated_shadowing = torch.zeros_like(ray_lengths)[...,0] == 1

        return ray_intensities * ray_attenuations, ray_directions, calculated_shadowing

    def parameter_info(self):
        return {
            "light_positions": [self.positions, 0.01, lambda x: x.detach().clone()],
            "light_intensities": [self.intensities, 1.0, lambda x: x.detach().clone()],
            "light_attenuations": [self.attenuations, 0.01, lambda x:x.detach().clone()],
        }

    def serialize(self):
        return [
            self.positions.detach().clone(),
            self.intensities.detach().clone(),
            self.attenuations.detach().clone(),
            self.rig_light
        ]

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
