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
import numpy as np
import open3d as o3d
import copy
import cv2
import trimesh
import thirdparty.pyrender.pyrender as pyrender
from tqdm import tqdm
import torch

from parametrizations.poses import PoseParametrizationFactory
from utils.logging import error, log
from utils.conversion import to_numpy, to_o3d, to_torch

gt_scan_folder = "/is/rg/avg/projects/mobile_lightstage/simon_scans"

def refine_registration(target, source, distance_threshold, Tinit=np.eye(4)):
    """
    Using ICP, register two nearly-registered point clouds onto eachother.

    Inputs:
        source, target          o3d.geometry.Pointclouds
        distance_threshold      Distance threshold for the ICP algorithm
        [Tinit]                 4x4 np.ndarray containing an initial estimate. Defaults to identity.
    
    Outputs:
        transformation          4x4 np.ndarray that transforms points in the source to the target domain
    """
    result = o3d.registration.registration_icp(
        target, source, distance_threshold, Tinit,
        o3d.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=1000)
    )
    return result.transformation

class CustomShaderCache():
    """
    Helper class for custom pyrender shader functions.
    """
    def __init__(self):
        self.program = None
    
    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(
                "thirdparty/pyrender_shaders/evaluation_mesh.vert",
                "thirdparty/pyrender_shaders/evaluation_mesh.frag",
                defines=defines
            )
        return self.program

def image_based_alignment(mesh_points, mesh_normals, Rt_img, K_img, depth, normals, verbose=True):
    """
    Refining a mesh's position and rotation based on depth and normal maps
    """
    # optimize Rt so that we get a good reprojection comparison
    # anything that doesn't project within 2 cm is wrong
    # average the mean of the lowest 50% distances after that
    torch_pts = to_torch(mesh_points)
    torch_mesh_normals = to_torch(mesh_normals)
    torch_K = to_torch(K_img).double()
    torch_Rt = to_torch(Rt_img).double()
    torch_pose_change = PoseParametrizationFactory("quaternion")()
    torch_pose_change.initialize(to_torch(np.eye(3)), to_torch(np.zeros((3,1))))
    torch_depth = to_torch(depth).double()
    torch_normals = to_torch(normals).double()

    # now we optimize a pose for the mesh
    dilated_depth = cv2.dilate(depth, np.ones((7,7)))
    eroded_depth = cv2.erode(depth, np.ones((7,7)))
    edges = (dilated_depth - eroded_depth) > 0.02
    edges = to_torch(edges.astype(int)).double()
    
    iterations = 100
    parameter_info = torch_pose_change.parameter_info()['observation_poses']
    optimizer = torch.optim.Adam(parameter_info[0], lr=parameter_info[1], betas=(0.9,0.9), eps=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[iterations/2, iterations/4*3],
        gamma=0.1
    )
    loss_evolution = []
    loop = range(iterations)
    if verbose:
        loop = tqdm(loop, desc="Optimizing pose for the reprojection")
    for it in loop:
        optimizer.zero_grad()
        Rts = torch_pose_change.Rts()[0]

        transformed_points = torch_pts @ Rts[:3,:3].T + Rts[:3,3:].T
        transformed_normals = torch_mesh_normals @ Rts[:3,:3].T
        
        projected_points = (transformed_points @ torch_Rt[:3,:3].T + torch_Rt[:3,3:].T) @ torch_K.T
        #find the pixels
        projected_pixels = (projected_points[:,:2] / projected_points[:,2:])
        projected_coords = projected_pixels.clone()
        # transform to gridsample units: -1 to 1
        projected_coords += 0.5
        projected_coords[:,0] /= depth.shape[1] / 2
        projected_coords[:,1] /= depth.shape[0] / 2
        projected_coords -= 1

        # sample depth
        sampled_depth = torch.nn.functional.grid_sample(
            torch_depth[None,None],
            projected_coords[None,None],
            align_corners=False
        )[0,0,0]
        sampled_edges = torch.nn.functional.grid_sample(
            edges[None,None],
            projected_coords[None,None],
            align_corners=False
        )[0,0,0]
        sampled_normals = torch.nn.functional.grid_sample(
            torch_normals.permute(2,0,1)[None],
            projected_coords[None,None],
            align_corners=False
        ).permute(0,2,3,1)[0,0]
        # depth differences
        depth_diffs = (sampled_depth - projected_points[:,2]).abs()
        normal_diffs = (transformed_normals * sampled_normals).sum(dim=1).clamp_(min=-1.,max=1.).acos()
        # anything too large is just set to <don't care>
        valid_depths = (depth_diffs <= 0.01) * (sampled_edges == 0) * (normal_diffs < np.pi/2)

        loss = (10000*depth_diffs[valid_depths] + normal_diffs[valid_depths]).mean()
        loss.backward()
        torch_pose_change.clear_cache()
        loss_evolution.append(loss.detach().item())
        optimizer.step()
        scheduler.step()
        if verbose:
            loop.set_description("Performing alignment refinement: loss %12.6f" % loss.item())

    return to_numpy(torch_pose_change.Rts()[0])

def render_depth_normals(vertices, faces, normals, K, camera_pose, image_shape):
    """
    Using pyrender, render a mesh into depth and normal images.

    Inputs:
        vertices, faces, normals        Nx3 np.ndarrays
        K                               3x3 np.ndarray with the intrinsic image
        camera_pose                     3x4 np.ndarray with the camera pose
        image_shape                     (height, width) python tuple
    
    Outputs:
        depth                           HxW np.ndarray with the depth map
        normals                         HxWx3 np.ndarray with the respective normals
                                            Invalid (or absent) normals are zero.
    """

    vertices = vertices @ camera_pose[:3,:3].T + camera_pose[:3,3:].T
    vertices[:, 1:] *= -1
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(128 - 128*np.array(normals)).astype(np.uint8))
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    render_camera_pose = np.eye(4)

    scene = pyrender.Scene()
    scene.add(mesh)
    scale = 1
    cam = pyrender.IntrinsicsCamera(K[0,0]/scale, K[1,1]/scale, K[0,2]/scale, K[1,2]/scale)
    scene.add(cam, pose=render_camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                        innerConeAngle=np.pi/16.0,
                        outerConeAngle=np.pi/6.0)
    scene.add(light, pose=render_camera_pose)
    r = pyrender.OffscreenRenderer(image_shape[1],image_shape[0])
    r._renderer._program_cache = CustomShaderCache()

    normals_rgb, depth = r.render(scene)
    normals = (128 - normals_rgb.astype(float)*255)/128.

    invalids = np.all(normals_rgb == 1., axis=2, keepdims=True)
    normals[invalids.repeat(3, axis=2)] = 0.
    normals = normals / (np.linalg.norm(normals, axis=2, keepdims=True) + invalids)
    return normals, depth

def evaluate_state(evaluation_name, object_name, experiment_state):
    """
    Evaluate the current experiment_state with the relevant ground truth.
    Performs registration refinement on the point clouds, and then calculates
    average geometric accuracy and normal angle (in the image domain).
    """
    gt_scan_file = os.path.join(gt_scan_folder, "manual_scan_alignment", "%s_manual.ply" % object_name)
    if not os.path.exists(gt_scan_file):
        error("WARNING> GT scan for %s not available" % object_name)

    gt_scan_mesh = o3d.io.read_triangle_mesh(gt_scan_file) # stored in m
    gt_scan = gt_scan_mesh.sample_points_uniformly(int(1e6)) # objects covering 1m^2, that's 1mm^2 per point
    gt_scan.estimate_normals()

    estimated_cloud = o3d.geometry.PointCloud()
    estimated_cloud.points = to_o3d(to_numpy(experiment_state.locations.location_vector()))
    estimated_cloud.normals = to_o3d(to_numpy(experiment_state.normals.normals()))
    estimated_cloud.colors = to_o3d(to_numpy(experiment_state.materials.get_brdf_parameters()['diffuse']))

    registration_transform = refine_registration(
        gt_scan,
        estimated_cloud,
        distance_threshold=0.005 #m
    )

    gt_mesh_aligned = copy.deepcopy(gt_scan_mesh)
    gt_mesh_aligned.transform(registration_transform)
    # o3d.visualization.draw_geometries([gt_mesh_aligned, estimated_cloud])

    # now project the gt_scan onto our image plane and calculate depth and normal errors there
    estimated_depth = to_numpy(experiment_state.locations.implied_depth_image())
    estimated_normals = to_numpy(experiment_state.locations.create_image(experiment_state.normals.normals()))
    
    K_proj  = to_numpy(experiment_state.locations.invK.inverse())
    Rt = to_numpy(experiment_state.locations.invRt.inverse())

    image_based_transform = image_based_alignment(
        np.array(gt_mesh_aligned.vertices),
        np.array(gt_mesh_aligned.vertex_normals),
        Rt,
        K_proj,
        estimated_depth,
        estimated_normals,
        verbose=False
    )
    gt_mesh_aligned.transform(np.concatenate((image_based_transform, np.array([0,0,0,1]).reshape(1,4)), axis=0))

    gt_normals_aligned, gt_depth_aligned = render_depth_normals(
        np.array(gt_mesh_aligned.vertices),
        np.array(gt_mesh_aligned.triangles),
        np.array(gt_mesh_aligned.vertex_normals),
        K_proj, Rt,
        estimated_depth.shape[:2],
    )

    depth_diff = np.abs(estimated_depth - gt_depth_aligned)

    dilated_depth = cv2.dilate(gt_depth_aligned, np.ones((7,7)))
    eroded_depth = cv2.erode(gt_depth_aligned, np.ones((7,7)))
    edges = (dilated_depth - eroded_depth) > 0.02

    valid_pixels = (estimated_depth > 0) * (gt_depth_aligned > 0) * (depth_diff < 0.02)
    edgevalid_pixels = (estimated_depth > 0) * (gt_depth_aligned > 0) * (depth_diff < 0.02) * (edges == 0)

    normal_dotprods = (gt_normals_aligned * estimated_normals).sum(axis=2).clip(min=-1., max=1.)
    normal_anglediff = np.arccos(normal_dotprods) / np.pi * 180.0

    average_accuracy = (depth_diff * edgevalid_pixels).sum() / edgevalid_pixels.sum()
    average_angle_error = (edgevalid_pixels*normal_anglediff).sum() / edgevalid_pixels.sum()

    log("Evaluating %s - %s: depth accuracy %6.4fmm        normal accuracy %10.4f degrees" % (evaluation_name, object_name, average_accuracy * 1000, average_angle_error))
