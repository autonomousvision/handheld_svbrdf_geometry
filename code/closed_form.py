import torch
import numpy as np
from tqdm import tqdm
from utils.vectors import normalize, inner_product
from utils.sentinels import OBSERVATION_OUT_OF_BOUNDS
import general_settings

def closed_form_lambertian_solution(experiment_state, data_adapter, sample_radius=7):
    device = torch.device(general_settings.device_name)
    # calculates the closed-form solution given the current state and the measurements
    with torch.no_grad():
        light_intensities = []
        light_directions = []
        shadows = []
        observations = []
        occlusions = []

        training_indices_batches, training_light_infos_batches = data_adapter.get_training_info()

        for batch_indices, batch_light_infos in zip(training_indices_batches, training_light_infos_batches):
            batch_light_intensities, batch_light_directions, batch_shadows = experiment_state.light_parametrization.get_light_intensities(
                experiment_state.locations, experiment_state.observation_poses.Rts(batch_indices), light_infos=batch_light_infos
            )
            light_intensities.append(batch_light_intensities.transpose(0,1))
            light_directions.append(batch_light_directions.transpose(0,1))
            shadows.append(batch_shadows.transpose(0,1))

            batch_observations, batch_occlusions = experiment_state.extract_observations(
                data_adapter,
                batch_indices,
                smoothing_radius=sample_radius,
            )
            batch_occlusions = (batch_occlusions + (batch_observations[...,1] == OBSERVATION_OUT_OF_BOUNDS)) > 0
            observations.append(batch_observations.transpose(0,1))
            occlusions.append(batch_occlusions.transpose(0,1))

        light_intensities = torch.cat(light_intensities, dim=1)
        light_directions = torch.cat(light_directions, dim=1)
        shadows = torch.cat(shadows, dim=1)
        observations = torch.cat(observations, dim=1)
        occlusions = torch.cat(occlusions, dim=1)

        incident_light = ( # points x views x color channels x direction_ray
            light_intensities[:,:,:,None] * light_directions[:,:,None,:]
        )
        
        shadowing_mask = (shadows == 0).float()[:,:,None]
        occlusion_mask = (occlusions == 0).float()[:,:,None]
        valid_mask = (shadowing_mask * occlusion_mask)
        invalids = (valid_mask.sum(dim=1) <= 2).float().view(-1,1,1)
        eye = torch.eye(3, device=invalids.device).view(1,3,3)*1e8
        
        # normals are estimated from a gray-scale version. However, here we are aiming for maximum SNR rather than
        # a visually pleasing grayscale. As such, no fancy color-dependent scaling
        gray_light_intensities = incident_light.mean(dim=-2, keepdim=False) * valid_mask
        gray_observations = observations.mean(dim=-1, keepdim=True) * valid_mask

        # a RANSAC approach
        # at every iteration, we take a small subset to calculate the solution from
        # and count the number of inliers
        subset_size = min(observations.shape[1], 6)
        threshold = 1e4
        # we aim for 10 succesfull RANSAC samples (i.e. without outliers)
        # not taking into account the sample set size
        estimated_outlier_ratio = 0.20
        targeted_success_chance = 1-1e-4
        nr_its = 10*np.log(1-targeted_success_chance) / np.log(1 - (1-estimated_outlier_ratio) ** subset_size) if subset_size < observations.shape[1] else 1
        best_inlier_counts = torch.zeros(observations.shape[0], 1, 1).to(observations.device)-1
        best_inliers = torch.zeros(*observations.shape[:2], 1).to(observations.device)
        ransac_loop = tqdm(range(int(nr_its)))
        for iteration in ransac_loop:
            subset = np.random.choice(observations.shape[1], size=[subset_size], replace=False)
            subset = torch.tensor(subset, dtype=torch.long, device=observations.device)
            intensities_subset = gray_light_intensities.index_select(dim=1, index=subset)
            observations_subset = gray_observations.index_select(dim=1, index=subset)
            shadowing_subset = shadowing_mask.index_select(dim=1, index=subset)
            occlusion_subset = occlusion_mask.index_select(dim=1, index=subset)
            valid_subset = shadowing_subset * occlusion_subset
            invalids_subset = (valid_subset.sum(dim=1) <= 2).float().view(-1,1,1)
            intensities_subset_pinv = (intensities_subset.transpose(1,2) @ intensities_subset + eye * invalids_subset).inverse() @ intensities_subset.transpose(1,2)
            solution_subset = intensities_subset_pinv @ observations_subset

            residuals = gray_light_intensities @ solution_subset - gray_observations
            inliers = ((residuals.abs() < threshold) * valid_mask).float()
            inlier_counts = inliers.sum(dim=1, keepdim=True)
            better_subset = (inlier_counts > best_inlier_counts).float()

            best_inlier_counts = best_inlier_counts * (1-better_subset) + inlier_counts * better_subset
            best_inliers = best_inliers * (1-better_subset) + inliers * better_subset

            ransac_loop.set_description(desc="Lambertian closed form | RANSAC total energy | avg inliers %8.5f" % best_inlier_counts.mean())

        # now calculate the normal estimate for the best inlier group
        inlier_intensities = gray_light_intensities * best_inliers
        inlier_observations = gray_observations * best_inliers
        inlier_invalids = ((valid_mask * best_inliers).sum(dim=1) <= 2).float().view(-1,1,1)
        inlier_intensities_pinv = (inlier_intensities.transpose(1,2) @ inlier_intensities + eye * inlier_invalids).inverse() @ inlier_intensities.transpose(1,2)
        inlier_normals = normalize((inlier_intensities_pinv @ inlier_observations).view(-1,3))

        albedos = []
        for c in range(3):
            c_light_intensities = inner_product(inlier_normals.view(-1,1,3), incident_light[:,:,c]) * valid_mask
            c_observations = observations[:,:,c,None] * valid_mask
            c_light_intensities_pinv = (c_light_intensities.transpose(1,2) @ c_light_intensities + invalids).inverse() @ c_light_intensities.transpose(1,2)
            c_albedo = np.pi * (c_light_intensities_pinv @ c_observations).view(-1,1)
            albedos.append(c_albedo)
        albedos = torch.cat(albedos, dim=1)
    return inlier_normals, albedos
