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

import os
import torch
from tqdm import tqdm
import numpy as np
import time
from matplotlib import pyplot as plt

from closed_form import closed_form_lambertian_solution
from experiment_state import ExperimentState

from parametrizations.locations import DepthMapParametrization
from utils.logging import error, log
from utils.conversion import to_numpy, to_torch
from utils.depth_maps import depth_map_to_locations
import general_settings

def higo_baseline(experiment_state, data_adapter, cache_path, higo_settings):
    """
    Reimplementation of the Higo et al. 2009 paper
        Tomoaki Higo, Yasuyuki Matsushita, Neel Joshi, and Katsushi Ikeuchi
        A hand-held photometric stereo camera for 3-d modeling
        ICCV2009

    Uses the PyMaxFlow library for graphcut problem: http://pmneila.github.io/PyMaxflow/.
    This library is installed as a python package by the installEnv.sh script.
    """

    if not isinstance(experiment_state.locations, DepthMapParametrization):
        error("Higo et al. 2009 requires a depth map parametrization.")
    
    os.makedirs(cache_path, exist_ok=True)
    device = torch.device(general_settings.device_name)

    with torch.no_grad():
        step_size = higo_settings['step_size']
        step_radius = higo_settings['step_radius']
        depth_range = step_size * step_radius # 2.5cm
        nr_steps = 2*step_radius + 1
        eta = higo_settings['eta'] * general_settings.intensity_scale
        lambda_n = higo_settings['lambda_n']
        lambda_s = higo_settings['lambda_s'] * step_size * 1000
        lambda_1 = higo_settings['lambda_1']
        lambda_2 = higo_settings['lambda_2']

        surface_constraint_threshold = 0.005 # 5mm
        surface_constraint_threshold = surface_constraint_threshold / (depth_range / nr_steps)
        surface_constraint_penalization = 0.010 # 1cm

        ## 1) calculate the photometric loss volume, and the depth/normal hypothesis volume
        N_pixels = experiment_state.locations.get_point_count()
        photo_loss_volume = torch.zeros(N_pixels, nr_steps).to(device)
        depth_volume = torch.zeros(N_pixels, nr_steps, 1).to(device)
        normal_volume = torch.zeros(N_pixels, nr_steps, 3).to(device)
        diffuse_volume = torch.zeros(N_pixels, nr_steps, 3).to(device)

        # we need multiples of the step size for the graph cut later on
        initial_depth = experiment_state.locations.implied_depth_image().clone()
        initial_depth.div_(step_size).round_().mul_(step_size)

        for offset_idx in tqdm(range(nr_steps), desc="Solving all RANSAC problems (photometric loss volume)"):
            depth_offset = -depth_range + offset_idx * step_size
            cache_file = os.path.join(cache_path, "%8.6f.npz" % depth_offset)
            depth = initial_depth + depth_offset
            if os.path.exists(cache_file):
                cached = np.load(cache_file)
                normals = to_torch(cached['normals'])
                inliers_N = to_torch(cached['inliers_N'])
                inlier_photometric_error = to_torch(cached['inlier_photometric_error'])
                albedo = to_torch(cached['diffuse'])
            else:
                spoofed_experiment_state = ExperimentState.copy(experiment_state)
                spoofed_experiment_state.locations = DepthMapParametrization()
                spoofed_experiment_state.locations.initialize(
                    depth, experiment_state.locations.mask,
                    experiment_state.locations.invK, experiment_state.locations.invRt,
                )
                normals, albedo, inliers, residuals = closed_form_lambertian_solution(
                    spoofed_experiment_state,
                    data_adapter,
                    sample_radius=0,
                    shadows_occlusions=False,
                    verbose=False
                )
                inlier_photometric_error = (residuals * inliers).abs().sum(dim=1).sum(dim=1)
                inliers_N = inliers.squeeze().sum(dim=1)
                np.savez_compressed(
                    cache_file,
                    normals=to_numpy(normals),
                    diffuse=to_numpy(albedo),
                    inliers_N=to_numpy(inliers_N),
                    inlier_photometric_error=to_numpy(inlier_photometric_error),
                )
            
            depth_volume[:, offset_idx, 0] = experiment_state.locations.create_vector(depth)
            normal_volume[:, offset_idx] = normals
            diffuse_volume[:, offset_idx] = albedo.squeeze()
            photo_loss_volume[:, offset_idx] = eta * inlier_photometric_error / inliers_N - inliers_N

        # precalculation of neighbour relationships
        mask = experiment_state.locations.mask
        py, px = torch.meshgrid([torch.arange(0,mask.shape[0]), torch.arange(0,mask.shape[1])])
        pixels = torch.stack((
            px[mask.squeeze()],
            py[mask.squeeze()],
        ), dim=0).to(device)
        
        indices = torch.zeros(*mask.shape[:2],1).long().to(device) - 1
        indices[mask] = torch.arange(N_pixels).to(device)[:,None]
        indices = torch.nn.functional.pad(indices, pad=(0,0,1,1,1,1), value=-1)
        neighbours = []
        for offset in [[-1,0],[1,0],[0,-1],[0,1]]:
            offset_pixels = pixels + 1 # because of the padding
            for c in range(2):
                offset_pixels[c,:] += offset[c]
            offset_linidces = offset_pixels[0,:] + offset_pixels[1,:] * indices.shape[1]
            neighbours.append(indices.flatten()[offset_linidces])
        neighbours = torch.stack(neighbours, dim=1)

        surface_constrain_cachefile = os.path.join(cache_path, "surface_normal_constraint.npz")
        ## 2) calculate the surface normal constraint loss volume:
        if not os.path.exists(surface_constrain_cachefile):
            # we add in a nonsense 'neighbour' that will never be able to win, for implementational cleanliness
            surface_constraint_volume = torch.zeros(N_pixels, nr_steps).to(device)
            neighbours_n = neighbours.clone()
            neighbours_n[neighbours_n < 0] = N_pixels
            depth_volume_n = torch.cat((depth_volume, torch.zeros(1, nr_steps, 1).to(device)))
            normal_volume_n = torch.cat((normal_volume, torch.ones(1, nr_steps, 3).to(device)))

            pixel_locs = torch.cat((
                pixels.float(), torch.ones(1,pixels.shape[1]).to(device)
            ), dim=0)
            pixel_locs_n = torch.cat((pixel_locs, torch.zeros(pixel_locs.shape[0], 1).to(device)), dim=1)
            for offset_idx in tqdm(range(nr_steps), desc="Generating the surface constraint loss volume"):
                hypothesis_points = experiment_state.locations.invK @ (pixel_locs * depth_volume[None, :, offset_idx, 0])
                hypothesis_normals = normal_volume[:, offset_idx].transpose(0,1)
                for n_idx in range(4):
                    these_neighbours = neighbours_n[:,n_idx]
                    n_pixel_locs = pixel_locs_n[:,these_neighbours]
                    best_label_points = torch.zeros(3, N_pixels).to(device)
                    best_label_normals = torch.zeros(3, N_pixels).to(device)
                    best_label_offsets = torch.zeros(N_pixels).to(device)
                    best_label_pdists = torch.zeros(N_pixels).to(device) + np.inf
                    for n_offset_idx in range(nr_steps):
                        n_hypothesis_points = experiment_state.locations.invK @ (n_pixel_locs * depth_volume_n[None, these_neighbours, n_offset_idx, 0])
                        n_hypothesis_normals = normal_volume_n[these_neighbours, n_offset_idx].transpose(0,1)
                        n_hypothesis_pdists = (hypothesis_normals*(hypothesis_points - n_hypothesis_points)).abs().sum(dim=0)
                        better_matches = n_hypothesis_pdists < best_label_pdists

                        best_label_offsets[better_matches] = n_offset_idx
                        best_label_pdists[better_matches] = n_hypothesis_pdists[better_matches]
                        best_label_points[:,better_matches] = n_hypothesis_points[:,better_matches]
                        best_label_normals[:,better_matches]= n_hypothesis_normals[:,better_matches]
                    hypothesis_ldists = (best_label_offsets - offset_idx).abs()
                    valid_best_labels = hypothesis_ldists < surface_constraint_threshold

                    hypothesis_pdists = (best_label_normals*(hypothesis_points - best_label_points)).abs().sum(dim=0)
                    # we don't have parallel depth planes, however!
                    surface_constraint_volume[valid_best_labels, offset_idx] += hypothesis_pdists[valid_best_labels]
                    surface_constraint_volume[valid_best_labels == False, offset_idx] = surface_constraint_penalization
            np.savez_compressed(
                surface_constrain_cachefile,
                surface_constraint_volume=to_numpy(surface_constraint_volume),
            )
        else:
            cached = np.load(surface_constrain_cachefile)
            surface_constraint_volume = to_torch(cached['surface_constraint_volume'])

        # at this point we can calculate the unary result, i.e. without TV-1 depth smoothness
        unary_loss_volume = photo_loss_volume + lambda_n * surface_constraint_volume

        winners_unary = torch.argmin(unary_loss_volume, dim=1)

        graphcut_cache_file = os.path.join(cache_path, "higo_graphcut.npz")
        if not os.path.exists(graphcut_cache_file):
            # 3) Graph-cut magic. TV-1 optimization on the depth
            #       because we now have discretized depth values, there is only a finite number of labels to optimize over.
            #       As such, we can use the graph construction from "Stereo Without Epipolar Lines: A Maximum-Flow Formulation"
            depth_min = depth_volume.min()
            depth_max = depth_volume.max()
            n_hyps = round((depth_max - depth_min).item() / step_size) + 1
            depth_hypotheses = depth_min + (depth_max - depth_min) * torch.arange(n_hyps).float().to(device) / (n_hyps - 1)
            depth_hypotheses.div_(step_size).round_().mul_(step_size)
            
            # make it amenable to graphcut optimization, i.e. all positive values
            safe_unary_loss_volume = unary_loss_volume.clone()
            safe_unary_loss_volume = safe_unary_loss_volume - (safe_unary_loss_volume.min() - 1)
            # a value definitely higher than the optimal solution's loss
            cost_upper_bound = safe_unary_loss_volume.sum(dim=0).sum().item() + 1
            # create the bigger volume of unary weights
            # because of the way graphcut imposes smoothness cost this is the easiest to implement
            full_unary_loss_volume = torch.zeros(len(unary_loss_volume), n_hyps).to(device) + cost_upper_bound
            for step in range(nr_steps):
                # fill in these unary losses in the correct position in the full volume
                full_idces = ((depth_volume[:, step] - depth_min) / step_size).round().long()
                full_values = safe_unary_loss_volume[:, step, None]
                full_unary_loss_volume.scatter_(dim=1, index=full_idces, src=full_values)
            full_offsets = (depth_volume[:,0,0] - depth_min).div_(step_size).round_()

            import maxflow
            graph = maxflow.GraphFloat()
            node_ids = graph.add_grid_nodes((N_pixels, n_hyps+1))
            for hyp in tqdm(range(1,n_hyps+1), desc="Building optimization graph - unary weights"):
                nodepairs = node_ids[:,hyp-1:hyp+1]
                edgeweights = to_numpy(full_unary_loss_volume[:, hyp-1:hyp])
                graph.add_grid_edges(
                    nodepairs,
                    weights=edgeweights,
                    structure=np.array([[0,0,0],[0,0,1],[0,0,0]]),
                    symmetric=1
                )
            # build terminal edges
            for x in tqdm(range(N_pixels), desc="Building optimization graph - terminal edges"):
                graph.add_tedge(node_ids[x,0],      cost_upper_bound, 0)
                graph.add_tedge(node_ids[x,n_hyps], 0, cost_upper_bound)
            
            # debug test: not including the smoothness loss *should* mean that we get exactly the unary winners
            # print("Starting unary maxflow calculation...")
            # tic = time.time()
            # unary_max_flow = graph.maxflow()
            # print("Finished in %ss" % (time.time() - tic))
            # unary_min_cut = graph.get_grid_segments(node_ids)
            # winners_unary_test = np.nonzero(unary_min_cut[:,1:] != unary_min_cut[:,:-1])[1]
            # assert np.all(winners_unary_test == to_numpy(winners_unary) + to_numpy(full_offsets)), "Issue building the graph: unary solution does not match unary WTA"

            no_neighbour_node = graph.add_nodes(1)
            neighbours_g = to_numpy(neighbours)
            neighbours_g[neighbours_g < 0] = len(neighbours)
            node_ids_g = np.concatenate((
                node_ids, np.ones((1,n_hyps+1), dtype=node_ids.dtype)*no_neighbour_node
            ), axis=0)
            for n_idx in range(4):
                neighbour_ids = np.take(node_ids_g[:,:-1], indices=neighbours_g[:,n_idx], axis=0)
                nodepairs = np.stack((
                    node_ids[:,:-1], neighbour_ids
                ), axis=2)
                edgeweights = lambda_s
                candidates = nodepairs[:,:,1] != no_neighbour_node
                candidates = to_numpy(
                    (depth_volume[:,0] - step_size * 3 <= depth_hypotheses[None])
                    *
                    (depth_volume[:,-1] + step_size * 3 >= depth_hypotheses[None])
                ) * candidates
                graph.add_grid_edges(
                    nodepairs[candidates].reshape(-1,2),
                    weights=edgeweights,
                    structure=np.array([[0,0,0],[0,0,1],[0,0,0]]),
                    symmetric=0
                )

            print("Starting full maxflow calculation...")
            tic = time.time()
            max_flow = graph.maxflow()
            print("Finished in %ss" % (time.time() - tic))
            min_cut = graph.get_grid_segments(node_ids)
            nonzeroes = np.nonzero(min_cut[:,1:] != min_cut[:,:-1])
            unique_nonzeroes = np.unique(nonzeroes[0], return_index=True)
            winners_graphcut = nonzeroes[1][unique_nonzeroes[1]]
            winners_graphcut = winners_graphcut - to_numpy(full_offsets)
            np.savez_compressed(
                graphcut_cache_file,
                winners_graphcut=winners_graphcut,
            )
        else:
            cached = np.load(graphcut_cache_file)
            winners_graphcut = cached['winners_graphcut']

        winners_graphcut = to_torch(winners_graphcut).long()


    #4) Depth refinement step
    def tangent_vectors(depth_image, invK, inv_extrinsics=None):
        """
        Given HxWx1 depth map, return quick-and-dirty the HxWx3 LR and UP tangents
        Takes the weighted left-right and up-down neighbour points as spanning the local plane.
        """
        assert len(depth_image.shape) == 3, "Depth map should be H x W x 1"
        assert depth_image.shape[2] == 1, "Depth map should be H x W x 1"
        H = depth_image.shape[0]
        W = depth_image.shape[1]
        data_shape = list(depth_image.shape)
        world_coords = depth_map_to_locations(depth_image, invK, inv_extrinsics)
        depth1 = depth_image[:-2,1:-1] * depth_image[1:-1,1:-1] * depth_image[2:,1:-1] * depth_image[1:-1,:-2] * depth_image[1:-1,2:]
        depth2 = depth_image[:-2,:-2] * depth_image[:-2,2:] * depth_image[2:,:-2] * depth_image[2:,2:]
        depth_mask = (depth1 * depth2 == 0).float()
        ud_vectors = (world_coords[:-2,1:-1,:] * 2 + world_coords[:-2,0:-2,:] + world_coords[:-2,2:,:]) \
                    - (world_coords[2:,1:-1,:] * 2 + world_coords[2:,0:-2,:] + world_coords[2:,2:,:])
        ud_vectors = ud_vectors / (depth_mask + ud_vectors.norm(dim=2, keepdim=True))
        lr_vectors = (world_coords[1:-1,:-2,:] * 2 + world_coords[0:-2,:-2,:] + world_coords[2:,:-2,:]) \
                    - (world_coords[1:-1,2:,:] * 2 + world_coords[0:-2,2:,:] + world_coords[2:,2:,:])
        lr_vectors = lr_vectors / (depth_mask + lr_vectors.norm(dim=2, keepdim=True))

        repad = lambda x: torch.nn.functional.pad(x, pad=(0,0,1,1,1,1))

        return repad(lr_vectors), repad(ud_vectors), repad((depth_mask == 0).float())

    def get_laplacian(depth_image, mask):
        kernel = to_torch(np.array([
            [-0.25, -0.50, -0.25],
            [-0.50,  3.00, -0.50],
            [-0.25, -0.50, -0.25],
        ])).float()
        laplacian = torch.nn.functional.conv2d(depth_image[None,None,:,:,0], kernel[None,None])[0,0,:,:,None]
        laplacian_mask = torch.nn.functional.conv2d(mask[None,None,:,:,0].float(), torch.ones_like(kernel)[None,None])[0,0,:,:,None] == 9
        repad = lambda x: torch.nn.functional.pad(x, pad=(0,0,1,1,1,1))
        return repad(laplacian * laplacian_mask.float())

    depth_estimate = torch.gather(depth_volume, dim=1, index=winners_graphcut[:,None,None])
    depth_estimate.requires_grad_(True)
    center_mask = mask.view(*mask.shape[:2], 1)
    with torch.no_grad():
        normal_estimate = torch.gather(normal_volume, dim=1, index=winners_graphcut[:,None,None].expand(-1,-1,3))
        normal_image = torch.zeros(*center_mask.shape[:2],3).to(device)
        normal_image.masked_scatter_(center_mask, normal_estimate)
        depth_estimate_initial = depth_estimate.clone()
        depth_image_initial = torch.zeros(*center_mask.shape[:2],1).to(device)
        depth_image_initial.masked_scatter_(center_mask, depth_estimate_initial)

    pixel_locs = torch.cat((
        pixels.float(), torch.ones(1,pixels.shape[1]).to(device)
    ), dim=0)
    loop = tqdm(range(1000), desc="Depth refinement")
    loss_evolution = []
    
    optimizer = torch.optim.Adam([depth_estimate], eps=1e-5, lr=0.0001, betas=[0.9,0.99])
    for iteration in loop:
        # term 1: position error
        initial_points = experiment_state.locations.invK @ (pixel_locs * depth_estimate_initial.view(1,-1))
        current_points = experiment_state.locations.invK @ (pixel_locs * depth_estimate.view(1,-1))
        position_diff = (initial_points - current_points).abs()
        position_error = (position_diff**2).sum()

        # term 2: the normal error
        depth_image = torch.zeros(*center_mask.shape[:2],1).to(device)
        depth_image.masked_scatter_(center_mask, depth_estimate)
        lr_img, ud_img, mask_img = tangent_vectors(depth_image, experiment_state.locations.invK, experiment_state.locations.invRt)
        lr = lr_img[center_mask.expand_as(lr_img)].view(-1,1,3)
        ud = ud_img[center_mask.expand_as(ud_img)].view(-1,1,3)
        mask = mask_img[center_mask].view(-1,1)
        normal_error = (mask * (
            ((lr*normal_estimate).sum(dim=2)**2)
          + ((ud*normal_estimate).sum(dim=2)**2)
        )).sum()

        # term 3: smoothness constraint
        laplacian = get_laplacian(depth_image, center_mask)
        laplacian_masked = laplacian # * (laplacian.abs() < 5 * step_size).float()
        smoothness_constraint = laplacian_masked.abs().sum()

        # the backprop
        total_loss = 1e5 * position_error + 10*normal_error + 3000*smoothness_constraint
        loss_evolution.append(total_loss.item())
        loop.set_description("Depth refinement | loss %8.6f" % loss_evolution[-1])

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    plt.clf()
    plt.plot(loss_evolution)
    plt.savefig(os.path.join(cache_path, "higo_refinement_loss.png"))
    plt.close()

    # now return a new experiment_state
    new_experiment_state = ExperimentState.copy(experiment_state)
    new_experiment_state.locations = DepthMapParametrization()
    new_experiment_state.locations.initialize(
        experiment_state.locations.create_image(depth_estimate[:,0]).squeeze(),
        experiment_state.locations.mask,
        experiment_state.locations.invK,
        experiment_state.locations.invRt,
    )
    
    winner_normals = torch.gather(normal_volume, dim=1, index=winners_graphcut[:,None,None].expand(-1,-1,3)).squeeze()
    winner_diffuse = torch.gather(diffuse_volume, dim=1, index=winners_graphcut[:,None,None].expand(-1,-1,3)).squeeze()

    new_experiment_state.normals = experiment_state.normals.__class__(new_experiment_state.locations)
    new_experiment_state.normals.initialize(winner_normals)
    new_experiment_state.materials = experiment_state.materials.__class__(new_experiment_state.brdf)
    new_experiment_state.materials.initialize(
        winner_diffuse.shape[0],
        winner_diffuse,
        winner_diffuse.device
    )
    new_experiment_state.materials.brdf_parameters['specular']['albedo'].data.zero_()
    new_experiment_state.materials.brdf_parameters['specular']['roughness'].data.fill_(0.1)
    return new_experiment_state
