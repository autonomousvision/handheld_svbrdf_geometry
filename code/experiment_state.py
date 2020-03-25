import torch
import os
import pickle
import numpy as np
import cv2
from matplotlib import pyplot as plt

import general_settings
from parametrizations.brdf import BrdfParametrizationFactory
from parametrizations.poses import PoseParametrizationFactory
from parametrizations.lights import LightParametrizationFactory
from parametrizations.normals import NormalParametrizationFactory
from parametrizations.materials import MaterialParametrizationFactory
from parametrizations.locations import LocationParametrizationFactory
from closed_form import closed_form_lambertian_solution
from losses import LossFunctionFactory

from utils.TOME import depth_reprojection_bound
from utils.vectors import normalize, inner_product
from utils.logging import error
from utils.sentinels import OBSERVATION_OUT_OF_BOUNDS, OBSERVATION_MASKED_OUT, OBSERVATION_OCCLUDED, SIMULATION_SHADOWED
from utils.visualization import draw_scene_preview, color_depths, color_depth_offsets, color_normals, curvature
from utils.visualization import color_errors, color_base_weights
from utils.export import export_pointcloud_as_ply

def to_numpy(x):
    return x.detach().cpu().numpy()

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
            roughness = self.materials.brdf_parameters['specular']['roughness']
            roughness.data[:] = (
                0.1 + 0.3 * torch.arange(start=0, end=roughness.shape[0], step=1).view(-1,1) / roughness.shape[0]
            )
            if 'eta' in self.materials.brdf_parameters['specular']:
                self.materials.brdf_parameters['specular']['eta'].data.fill_(1.5)
        else:
            error("Only hardcoded specular initialization is supported.")

        if initialization_settings['normals'] == "from_closed_form":
            self.normals.initialize(closed_form_normals)
        elif initialization_settings['normals'] == "from_depth":
            self.normals.initialize(self.locations.implied_normal_vector())

    def simulate(self, observation_indices, light_infos, shadow_cache=None, override=None):
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
        if override is not None and "geometry" in override:
            surface_normals = NormalParametrizationFactory("hard linked")(self.locations).normals()
            brdf = BrdfParametrizationFactory("cook torrance F1")()
            N, device = surface_normals.shape[0], surface_normals.device
            surface_materials = {
                "diffuse": torch.empty(N, 3, device=device).fill_(0.8),
                "specular": {
                    "albedo": torch.empty(N, 3, device=device).fill_(0.2),
                    "roughness": torch.empty(N, 1, device=device).fill_(0.15 - 0.10*("1" in override)),
                }
            }
        else:
            surface_materials = self.materials.get_brdf_parameters()
            if override is not None:
                surface_materials = dict(surface_materials)
                if override == "diffuse":
                    surface_materials['specular'] = dict(surface_materials['specular'])
                    surface_materials['specular']['albedo'] = torch.zeros_like(
                        surface_materials['specular']['albedo']
                    )
                elif override == "specular":
                    surface_materials['diffuse'] = torch.zeros_like(
                        surface_materials['diffuse']
                    )
            surface_normals = self.normals.normals().view(1,-1,3)
            brdf = self.brdf

        surface_rhos, NdotHs = brdf.calculate_rhos(light_directions, view_directions, surface_normals, surface_materials)

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

    def visualize_statics(self, output_path, data_adapter):
        with torch.no_grad():
            surface_points = self.locations.location_vector()

            cv2.imwrite(
                os.path.join(output_path, "mask.png"),
                to_numpy(
                    self.locations.create_image(
                        torch.ones_like(surface_points)
                    )
                ) * 255
            )
        
            observation_Rts = self.observation_poses.Rts()
            for i, (elevation, azimuth) in enumerate(zip([90., 20.], [0., 70.])):
                fig = draw_scene_preview(
                    surface_points,
                    data_adapter.center_invRt,
                    observation_Rts,
                    elevation=elevation,
                    azimuth=azimuth,
                )
                plt.savefig(
                    os.path.join(output_path, "camera_locations_%d.png" % i),
                    bbox_inches='tight'
                )
                plt.close(fig)


    def visualize(self, output_path, set_name, data_adapter, losses):
        with torch.no_grad():
            # depth image
            original_depths = self.locations.create_vector(data_adapter.center_depth)
            offset = original_depths.min() * 0.95
            scale = original_depths.max() * 1.05 - offset
            depth_image_file = os.path.join(output_path, "depth_%s.png" % set_name)
            depths = self.locations.implied_depth_vector()
            cv2.imwrite(
                depth_image_file,
                to_numpy(self.locations.create_image(
                    color_depths(depths, offset, scale),
                    filler=255
                ))
            )

            # depth change w.r.t. original
            depth_change = depths - original_depths
            depth_change_file = os.path.join(output_path, "depth_change_%s.png" % set_name)
            cv2.imwrite(
                depth_change_file,
                to_numpy(self.locations.create_image(
                    color_depth_offsets(depth_change) * 255,
                    filler=255
                ))
            )

            # geometry: save a rendered image with normals from depth and a uniform brdf
            normals = self.normals.normals()
            center_view_idx = [
                i for i, view_info in enumerate(data_adapter.observation_views)
                if view_info[0] == data_adapter.center_view
            ][0]
            training_indices, training_light_infos = data_adapter.get_training_info()
            for i in range(1,3):
                geometry_simulation = self.simulate(
                    training_indices[center_view_idx,None],
                    training_light_infos[center_view_idx,None],
                    override="geometry_%d" % i
                )[0]
                geometry_image_file = os.path.join(
                    output_path,
                    "geometry%d_%s.png" % (i, set_name)
                )
                cv2.imwrite(
                    geometry_image_file,
                    to_numpy(self.locations.create_image(
                        geometry_simulation * general_settings.intensity_scale,
                        filler=255
                    ))
                )

            if general_settings.save_clouds:
                # depth cloud
                export_pointcloud_as_ply(
                    os.path.join(output_path, "pointcloud_%s.ply" % set_name),
                    self.locations.location_vector(),
                    normals=normals,
                )

            normals_file = os.path.join(output_path, "normals_%s.png" % set_name)
            cv2.imwrite(
                normals_file,
                to_numpy(self.locations.create_image(
                    color_normals(normals),
                    filler=255
                ))
            )

            normal_curvature = self.locations.create_vector(
                curvature(self.locations.create_image(normals))
            )
            normal_curvature_file = os.path.join(output_path, "normals_curvature_%s.png" % set_name)
            cv2.imwrite(
                normal_curvature_file,
                to_numpy(self.locations.create_image(
                    color_errors(normal_curvature.sqrt_()),
                    filler=255
                ))
            )

            if hasattr(self.materials, 'base_weights'):
                weight_file = os.path.join(output_path, "base_weights_%s.png" % set_name)
                cv2.imwrite(
                    weight_file,
                    to_numpy(self.locations.create_image(
                        color_base_weights(self.materials.base_weights),
                        filler=255
                    ))
                )

            simulations = self.simulate(
                training_indices,
                training_light_infos,
                shadow_cache=None
            )
            observations, observation_occlusions = self.extract_observations(
                data_adapter,
                training_indices,
                occlusion_cache=None
            )
            photo_loss = [x for x in losses if "photoconsistency" in x]
            if len(photo_loss) > 0:
                photo_loss = photo_loss[0]
                photoconsistency_errors = LossFunctionFactory(
                    photo_loss
                )().evaluate(
                    simulations,
                    [observations, observation_occlusions],
                    self,
                    data_adapter
                )
            else:
                photoconsistency_errors = None

            os.makedirs(os.path.join(output_path, "fitting_set"), exist_ok=True)
            for training_index in training_indices:
                camera_index = data_adapter.observation_views[training_index][0]

                cv2.imwrite(
                   os.path.join(output_path, "fitting_set", "cam%05d_%s_predicted.png" % (camera_index, set_name)),
                    to_numpy(self.locations.create_image(
                        simulations[training_index] * general_settings.intensity_scale,
                        filler=255
                    ))
                )
                cv2.imwrite(
                   os.path.join(output_path, "fitting_set", "cam%05d_%s_observed.png" % (camera_index, set_name)),
                    to_numpy(self.locations.create_image(
                        observations[training_index] * general_settings.intensity_scale,
                        filler=255
                    ))
                )
                
                if photoconsistency_errors is not None:
                    cv2.imwrite(
                        os.path.join(output_path, "fitting_set", "cam%05d_%s_error.png" % (camera_index, set_name)),
                        to_numpy(self.locations.create_image(
                            color_errors(
                                photoconsistency_errors[training_index] * general_settings.intensity_scale,
                                scale=1,
                            ),
                            filler=255
                        ))
                    )

            diffuse_simulation = self.simulate(
                training_indices[center_view_idx,None],
                training_light_infos[center_view_idx,None],
                override="diffuse"
            )[0]
            diffuse_image_file = os.path.join(
                output_path,
                "diffuse_component_%s.png" % (set_name)
            )
            cv2.imwrite(
                diffuse_image_file,
                to_numpy(self.locations.create_image(
                    diffuse_simulation * general_settings.intensity_scale,
                    filler=255
                ))
            )
            specular_simulation = self.simulate(
                training_indices[center_view_idx,None],
                training_light_infos[center_view_idx,None],
                override="specular"
            )[0]
            specular_image_file = os.path.join(
                output_path,
                "specular_component_%s.png" % (set_name)
            )
            cv2.imwrite(
                specular_image_file,
                to_numpy(self.locations.create_image(
                    specular_simulation * general_settings.intensity_scale,
                    filler=255
                ))
            )

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
