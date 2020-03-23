import torch
import os
import pickle
import numpy as np

import general_settings
from parametrizations.brdf import BrdfParametrizationFactory
from parametrizations.poses import PoseParametrizationFactory
from parametrizations.lights import LightParametrizationFactory
from parametrizations.normals import NormalParametrizationFactory
from parametrizations.materials import MaterialParametrizationFactory
from parametrizations.locations import LocationParametrizationFactory
from closed_form import closed_form_lambertian_solution

from utils.TOME import depth_reprojection_bound
from utils.vectors import normalize, inner_product
from utils.logging import error
from utils.sentinels import OBSERVATION_OUT_OF_BOUNDS, OBSERVATION_MASKED_OUT, OBSERVATION_OCCLUDED, SIMULATION_SHADOWED

class ExperimentState:
    def __init__(
        self,
        locations,
        normals,
        brdf,
        materials,
        observation_poses,
        light_parametrization,
    ):
        self.brdf = brdf
        self.locations = locations
        self.normals = normals
        self.materials = materials
        self.observation_poses = observation_poses
        self.light_parametrization = light_parametrization

    @staticmethod
    def create(parametrization_settings):
        locations = LocationParametrizationFactory(parametrization_settings['locations'])()
        normals = NormalParametrizationFactory(parametrization_settings['normals'])(locations)

        brdf = BrdfParametrizationFactory(parametrization_settings['brdf'])()
        materials = MaterialParametrizationFactory(parametrization_settings['materials'])(brdf)

        poses = PoseParametrizationFactory(parametrization_settings['observation_poses'])()
        lights = LightParametrizationFactory(parametrization_settings['lights'])()

        return ExperimentState(locations, normals, brdf, materials, poses, lights)

    def initialize(self, data_adapter, initialization_settings):
        device = torch.device(general_settings.device_name)

        self.locations.initialize(
            data_adapter.center_depth,
            data_adapter.center_depth > 0,
            data_adapter.center_invK,
            data_adapter.center_invRt
        )
        self.observation_poses.initialize(
            torch.stack([image.original_extrinsics[:3,:3] for image in data_adapter.images]).to(device),
            torch.stack([image.original_extrinsics[:3,3:] for image in data_adapter.images]).to(device),
        )

        if initialization_settings['lights'] == "precalibrated":
            with open(initialization_settings['light_calibration_files']['positions'], "rb") as fh:
                light_positions = pickle.load(fh)[0]
            light_intensities = np.load(initialization_settings['light_calibration_files']['intensities'])
            light_attenuations = np.load(initialization_settings['light_calibration_files']['attenuations'])
            self.light_parametrization.initialize(
                torch.tensor(light_positions, dtype=torch.float32, device=device),
                torch.tensor(light_intensities, dtype=torch.float32, device=device),
                torch.tensor(light_attenuations, dtype=torch.float32, device=device),
            )
        else:
            error("Only precalibrated lights are supported.")

        if any(["closed_form" in initialization_settings[entry] for entry in initialization_settings]):
            # we require the closed form solution given these locations
            closed_form_normals, closed_form_diffuse = closed_form_lambertian_solution(self, data_adapter)

        # below is some debug code for in-progress progress
        
        if initialization_settings['diffuse'] == "from_closed_form":
            self.materials.initialize(self.locations.get_point_count(), closed_form_diffuse, self.locations.device())
        else:
            error("Only closed-form diffuse initialization is supported.")

        if initialization_settings['specular'] == "hardcoded":
            self.materials.brdf_parameters['specular']['albedo'].data.fill_(0.5)
            self.materials.brdf_parameters['specular']['roughness'].data.fill_(0.1)
            if 'eta' in self.materials.brdf_parameters['specular']:
                self.materials.brdf_parameters['specular']['eta'].data.fill_(1.5)
        else:
            error("Only hardcoded specular initialization is supported.")

        if initialization_settings['normals'] == "from_closed_form":
            self.normals.initialize(closed_form_normals)
        elif initialization_settings['normals'] == "from_depth":
            self.normals.initialize(self.locations.implied_normal_vector())

    def simulate(self, observation_indices, light_infos, shadow_cache=None):
        """
        Simulate the result of the current experiment state for a given observation index, including the shadow mask if requested
        """
        if shadow_cache is None:
            shadow_cache = {}

        surface_points = self.locations.location_vector()
        obs_Rt = self.observation_poses.Rts(observation_indices)
        shadow_cache_miss = not light_infos in shadow_cache
        light_intensities, light_directions, calculated_shadowing = self.light_parametrization.get_light_intensities(
            self.locations, obs_Rt, light_infos=light_infos, calculate_shadowing=shadow_cache_miss
        )

        if shadow_cache_miss:
            shadow_mask = calculated_shadowing
            shadow_cache[light_infos] = shadow_mask
        else:
            shadow_mask = shadow_cache[light_infos]

        obs_camloc = self.observation_poses.camlocs(observation_indices)
        view_directions = normalize(obs_camloc.view(-1,1,3) - surface_points[None])
        surface_materials = self.materials.get_brdf_parameters()
        surface_normals = self.normals.normals().view(1,-1,3)
        surface_rhos, NdotHs = self.brdf.calculate_rhos(light_directions, view_directions, surface_normals, surface_materials)

        incident_light = light_intensities * inner_product(light_directions, surface_normals)

        incident_light.clamp_(min=0.)
        incident_light[shadow_mask] = 0.

        simulation = surface_rhos * incident_light

        # TODO: this is where vignetting would normally come in

        simulation[incident_light <= 0] = SIMULATION_SHADOWED

        return simulation

    def extract_observations(self, data_adapter, observation_indices, calculate_occlusions=True, occlusion_cache=None, smoothing_radius=None):
        """
        Get the relevant observed values for a given observation index, including the occlusion mask if requested
        """
        if occlusion_cache is None:
            occlusion_cache = {}

        obs_Rt = self.observation_poses.Rts(observation_indices)
        obs_camloc = self.observation_poses.camlocs(observation_indices)

        ctr_Rt = data_adapter.center_invRt.inverse()
        ctr_P = data_adapter.center_invK.inverse() @ ctr_Rt[:3]
        obs_K = torch.stack(
            [data_adapter.images[observation_index].get_intrinsics() for observation_index in observation_indices],
            dim=0
        )
        obs_P = obs_K @ obs_Rt

        surface_points = self.locations.location_vector().view(1,-1,3) # 1xNx3
        projected = surface_points @ obs_P[...,:3,:3].transpose(-1,-2) + obs_P[...,:3,3:].transpose(-1,-2)
        projected_depth = projected[..., 2:]
        projected = projected[..., :2] / projected_depth
        projected_p = projected.clone()

        if hasattr(data_adapter,"compound_image_tensor"):
            compound_image_tensor = data_adapter.compound_image_tensor
            if smoothing_radius is not None and smoothing_radius > 0:
                weights = torch.ones(1,1,2*smoothing_radius+1,2*smoothing_radius+1).to(compound_image_tensor.device) / (2*smoothing_radius+1)**2
                compound_image_tensor = torch.nn.functional.conv2d(
                    compound_image_tensor.view(-1,1,*compound_image_tensor.shape[-2:]),
                    weights, bias=None, padding=smoothing_radius
                ).view(compound_image_tensor.shape)
            compound_H, compound_W = compound_image_tensor.shape[-2:]
            compound_sizes = data_adapter.compound_image_tensor_sizes

            projected += 0.5
            projected[...,0] /= compound_image_tensor.shape[-1]/2
            projected[...,1] /= compound_image_tensor.shape[-2]/2
            projected -= 1

            observations = torch.nn.functional.grid_sample(
                compound_image_tensor,
                projected[:, None],
                mode="bilinear",
                align_corners=False,
            )[:,:,0]
        else:
            observations = []
            compound_H = 0
            compound_W = 0
            compound_sizes = torch.zeros(C, 2, dtype=torch.long, device=device)
            for idx, observation_index in enumerate(observation_indices):
                image_data = data_adapter.images[observation_index].get_image() # 3xHxW
                if smoothing_radius is not None and smoothing_radius > 0:
                    weights = torch.ones(1,1,2*smoothing_radius+1,2*smoothing_radius+1).to(image_data.device) / (2*smoothing_radius+1)**2
                    image_data = torch.nn.functional.conv2d(image_data[:,None], weights, bias=None, padding=smoothing_radius)[:,0]
                compound_H = max(compound_H, image_data.shape[-2])
                compound_W = max(compound_W, image_data.shape[-1])
                compound_sizes[idx,0] = image_data.shape[-1]
                compound_sizes[idx,1] = image_data.shape[-2]

                subprojected = projected[idx]
                subprojected += 0.5
                subprojected[...,0] /= image_data.shape[-1]/2
                subprojected[...,1] /= image_data.shape[-2]/2
                subprojected -= 1

                observation = torch.nn.functional.grid_sample(
                    image_data[None],
                    subprojected[None, None],
                    mode="bilinear",
                    align_corners=False,
                )[:,:,0]

                observations.append(observation)
            observations = torch.cat(observations, dim=0)
            
            projected += 0.5
            projected[...,0] /= compound_W/2
            projected[...,1] /= compound_H/2
            projected -= 1

        observations = observations.transpose(-1, -2) # CxNx3
        out_of_bounds = (
            (projected_p[...,0] < 1) +
            (projected_p[...,1] < 1) +
            (projected_p[...,0] >= data_adapter.compound_image_tensor_sizes[:,:1]) + 
            (projected_p[...,1] >= data_adapter.compound_image_tensor_sizes[:,1:])
        )[:,:,None] > 0
        observations.masked_fill_(out_of_bounds, OBSERVATION_OUT_OF_BOUNDS)
        observations.masked_fill_(out_of_bounds, OBSERVATION_OUT_OF_BOUNDS)
        # saturations = saturations.transpose(-1, -2) # CxNx3
        # saturations.masked_fill_(projected[...,:1].abs() >= 0.99, OBSERVATION_OUT_OF_BOUNDS)
        # saturations.masked_fill_(projected[...,1:].abs() >= 0.99, OBSERVATION_OUT_OF_BOUNDS)

        if observation_indices in occlusion_cache:
            occlusion_masks = occlusion_cache[observation_indices]
        else:
            occlusion_fattening = general_settings.occlusion_fattening
            depth_image = self.locations.create_image(surface_points.view(-1,3)) @ ctr_Rt[2:3,:3].T[None] + ctr_Rt[2:3,3:].T[None]
            with torch.no_grad():
                obs_invKR = obs_Rt[...,:3,:3].transpose(-1,-2) @ obs_K.inverse()
                bounded = depth_reprojection_bound(
                    depth_image.view(1,*depth_image.shape[:2]),
                    ctr_P.view(1,3,4),
                    obs_invKR.view(1,-1,3,3),
                    obs_camloc.view(1,-1,3,1),
                    compound_H,
                    compound_W,
                    0.250,
                    1.500,
                    0.010
                ).view(-1,1,compound_H, compound_W)
                zbuffer = torch.nn.functional.grid_sample(
                    bounded,
                    projected[:, None],
                    mode="bilinear",
                    align_corners=False,
                )[:,0,0,:,None]
                occlusion_mask = projected_depth >= zbuffer + 0.005
                if occlusion_fattening is not None and occlusion_fattening > 0:
                    # first, scatter the occlusion mask into the 'bounded' plane
                    projected_pl = projected_p.round().long()
                    projected_pl.clamp_(min=0)
                    projected_pl[...,0].clamp_(max=compound_W-1)
                    projected_pl[...,1].clamp_(max=compound_H-1)
                    occluders = torch.zeros_like(bounded)
                    lin_idces = projected_pl[...,0] + projected_pl[...,1] * compound_W
                    C = bounded.shape[0]
                    occluders.view(C,-1).scatter_(dim=1,index=lin_idces, src=occlusion_mask.view(C,-1).float())
                    # then dilate only those pixels that caused occlusion
                    weights = torch.ones(1,1,2*occlusion_fattening+1,2*occlusion_fattening+1).to(occluders.device)
                    occluders.masked_fill_(bounded == 0, 0.)
                    occluders_dilated = torch.nn.functional.conv2d(
                        bounded * occluders, weights, bias=None, padding=occlusion_fattening
                    )
                    dilation_weights = torch.nn.functional.conv2d(
                        occluders, weights, bias=None, padding=occlusion_fattening
                    )
                    occluders_dilated = occluders_dilated / torch.clamp(dilation_weights, min=1)
                    fattened_bounded = torch.min(bounded, occluders_dilated + 1e3*(dilation_weights == 0).float())
                    # then recalculate the occlusion
                    fattened_zbuffer = torch.nn.functional.grid_sample(
                        fattened_bounded,
                        projected[:, None],
                        mode="bilinear",
                        align_corners=False,
                    )[:,0,0,:,None]
                    fattened_occlusion_mask = projected_depth >= fattened_zbuffer + 0.005
                    occlusion_mask = fattened_occlusion_mask
                occlusion_masks = occlusion_mask.view(C, -1)
                occlusion_cache[observation_indices] = occlusion_masks

        return observations, occlusion_masks

    def _get_named_fields(self):
        return {
            "locations": self.locations,
            "normals": self.normals,
            "brdf": self.brdf,
            "materials": self.materials,
            "observation_poses": self.observation_poses,
            "lights": self.light_parametrization,
        }

    def enforce_parameter_bounds(self):
        [field.enforce_parameter_bounds() for field in self._get_named_fields().values()]

    def clear_parametrization_caches(self):
        [field.clear_cache() for field in self._get_named_fields().values()]

    def get_parameter_dictionaries(self):
        parameter_dictionary = {}
        learning_rate_dictionary = {}
        visualizer_dictionary = {}
        for field in self._get_named_fields().values():
            parameter_info = field.parameter_info()
            for parameter in parameter_info:
                parameter_dictionary[parameter] = parameter_info[parameter][0]
                learning_rate_dictionary[parameter] = parameter_info[parameter][1]
                visualizer_dictionary[parameter] = parameter_info[parameter][2]
            
        return parameter_dictionary, learning_rate_dictionary, visualizer_dictionary

    def load(self, folder):
        for name, field in self._get_named_fields().items():
            saved_tensors = torch.load(os.path.join(folder, name+".torch"))
            field.deserialize(*saved_tensors if saved_tensors is not None else [None])

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        for name, field in self._get_named_fields().items():
            torch.save(field.serialize(), os.path.join(folder, name+".torch"))
