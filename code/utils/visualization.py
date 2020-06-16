"""
Copyright (c) 2020 Autonomous Vision Group (AVG), Max Planck Institute for Intelligent Systems, Tuebingen, Germany

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

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def color_normals(normals):
    """
    Transform a set of normals to a colorized representation that can be visualized.

    Inputs:
        normals         torch.tensor or np.ndarray

    Outputs:
        colorized       A colorized version of these normals for visualization, in [0,256[
    """
    return 128-128*normals

def curvature(normals_image):
    """
    Given HxWx3 vector map, return an angle map containing the angles
    to the below and the right neighbours.

    Inputs:
        normals_image       HxWx3 torch.tensor containing the normal vectors
    
    Outputs:
        curvatures          HxWx1 torch.tensor containing indicators for the local curvature
    """
    ud_angles = (normals_image[:-1,:-1, None, :] @ normals_image[1:, :-1, :, None]).clamp(min=-1.0,max=1.0)[:,:,0].acos()
    lr_angles = (normals_image[:-1,:-1, None, :] @ normals_image[:-1, 1:, :, None]).clamp(min=-1.0,max=1.0)[:,:,0].acos()

    return ud_angles ** 2 + lr_angles ** 2

_color_map_depths = torch.tensor(np.array([
    [0, 0, 0],          # 0.000
    [0, 0, 255],        # 0.114
    [255, 0, 0],        # 0.299
    [255, 0, 255],      # 0.413
    [0, 255, 0],        # 0.587
    [0, 255, 255],      # 0.701
    [255, 255,  0],     # 0.886
    [255, 255,  255],   # 1.000
    [255, 255,  255],   # 1.000
])).float()
_color_map_bincenters = torch.tensor(np.array([
    0.0,
    0.114,
    0.299,
    0.413,
    0.587,
    0.701,
    0.886,
    1.000,
    2.000, # doesn't make a difference, just strictly higher than 1
])).float()

def color_depths(depth, offset=None, scale=None):
    """
    Color a depth image.

    Inputs:
        depth       Nx1 torch.tensor containing depth values
        [offset]    Offset to subtract from the depth values before visualization. Defaults to depth.min()
        [scale]     Scale to divide the depth values by before visualization. Defaults to (depth.max() - offset)
    
    Outputs:
        colorized   Nx3 torch.tensor with the colorized version of the depth values
    """
    if offset is None:
        offset = depths.min()
    if scale is None:
        scale = depths.max() - offset

    values = ((depth.flatten() - offset) / scale).clamp_(min=0, max=1).to(_color_map_depths.device)
    # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
    lower_bin = (values.view(-1,1) >= _color_map_bincenters.view(1,-1)).max(dim=1)[1]
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1-alphas).view(-1,1) + _color_map_depths[lower_bin + 1] * alphas.view(-1,1)
    return colors

def color_depth_offsets(offsets):
    """
    Color a depth difference image. Blue means a positive difference, red negative.
    The bigger the difference, the more intense the color is.

    Inputs:
        offsets     Nx1 torch.tensor with depth differences
    
    Outputs:
        colorized   Nx3 torch.tensor with the colorized version of the depth differences.
    """
    offset_colors = torch.tensor([[0,0,255],[255,0,0]]).to(offsets.device)
    error_colors = offset_colors[(offsets > 0).long()]
    return error_colors * offsets.abs().view(-1,1)


_color_map_errors = torch.tensor(np.array([
    [149,  54, 49],     #0: log2(x) = -infinity
    [180, 117, 69],     #0.0625: log2(x) = -4
    [209, 173, 116],    #0.125: log2(x) = -3
    [233, 217, 171],    #0.25: log2(x) = -2
    [248, 243, 224],    #0.5: log2(x) = -1
    [144, 224, 254],    #1.0: log2(x) = 0
    [97, 174,  253],    #2.0: log2(x) = 1
    [67, 109,  244],    #4.0: log2(x) = 2
    [39,  48,  215],    #8.0: log2(x) = 3
    [38,   0,  165],    #16.0: log2(x) = 4
    [38,   0,  165]    #inf: log2(x) = inf
])).float()

def color_errors(errors, scale=1):
    """
    Color an error image.

    Inputs:
        errors          torch.tensor containing error values, of arbitrary shape
        [scale]         Scale to divide the error values by before visualization. Defaults to 1.
    
    Outputs:
        colorized       torch.tensor with the colorized version of the error values.
                        Has an extra dimension of size 3 appended.
    """
    errors_color_indices = ((errors / scale + 1e-5).log2() + 5).clamp_(min=0, max=9)
    i0 = errors_color_indices.long()
    f1 = errors_color_indices - i0.float()
    i0 = i0[:,0]

    global _color_map_errors
    _color_map_errors = _color_map_errors.to(errors.device)
    colored_errors_flat = _color_map_errors[i0] * (1-f1) + _color_map_errors[i0+1] * f1

    return colored_errors_flat.reshape(*errors.shape[:-1], 3)

_base_colors = torch.Tensor(np.array([
    [255, 0, 0],     # blue
    [0, 0, 255],       # red
    [0, 255, 0],       # green
])).float()

def color_base_weights(base_weights):
    """
    Color a set of base weights according to a pre-defined color scheme.

    Inputs:
        base_weights        NxB torch.tensor containing the weights, where B<=3.
    
    Outputs:
        colors              Nx3 torch.tensor containing the colorized version.
    """
    base_colors = _base_colors[:base_weights.shape[1]].to(base_weights.device)
    return base_weights @ base_colors

def draw_scene_preview(object_points, center_invRt, observation_cameras, elevation=0, azimuth=0):
    """
    Draws the scene (object points, reference view position and all observation cameras) in a matplotlib.pyplot.Figure.
    The caller is responsible for showing, saving, and/or closing the figure.

    Inputs:
        object_points           Nx3 torch.tensor containing the scene point locations
        center_invRt            3x4 torch.tensor containing the (inverse) extrinsics of the reference view
        observation_cameras     Cx3x4 torch.tensor containing the extrinsics of the observation views
        [elevation]             The 3d elevation of the matplotlib visualization.
        [azimuth]               The 3d azimuth of the matplotlib visualization.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    camera_frustum_0, camera_frustum_1 = _get_camera_frustum()

    observation_Rts = observation_cameras.detach().cpu().numpy()
    observation_invRs = observation_Rts[:,:3,:3].transpose(0,2,1)
    observation_camlocs = - observation_invRs @ observation_Rts[:,:3,3:]
    observation_frustums_0 = (observation_invRs @ camera_frustum_0[None] + observation_camlocs).transpose(0,2,1)
    observation_frustums_1 = (observation_invRs @ camera_frustum_1[None] + observation_camlocs).transpose(0,2,1)

    ax.scatter(
        observation_camlocs[:, 0, 0], observation_camlocs[:, 1, 0], observation_camlocs[:, 2, 0],
        marker='o',
        c='tab:orange',
        s=10.0,
        label='neighboring views',
        alpha=1.0
    )
    ax.quiver(
        *[observation_frustums_0[...,i].flatten() for i in range(3)],
        *[(observation_frustums_1[...,i] - observation_frustums_0[...,i]).flatten() for i in range(3)],
        arrow_length_ratio=0,
        color='tab:orange'
    )

    if center_invRt is not None:
        center_invRt = center_invRt.detach().cpu().numpy()
        center_camloc = center_invRt[:3,3:]
        center_frustum_0 = center_invRt[:3,:3] @ camera_frustum_0 + center_camloc
        center_frustum_1 = center_invRt[:3,:3] @ camera_frustum_1 + center_camloc

        ax.scatter(
            center_camloc[0,0], center_camloc[1,0], center_camloc[2,0],
            marker='o',
            c='tab:blue',
            s=50.0,
            label='center view',
            alpha=1.0
        )
        ax.quiver(
            *[center_frustum_0[i] for i in range(3)],
            *[(center_frustum_1[i] - center_frustum_0[i]) for i in range(3)],
            arrow_length_ratio=0,
            color='tab:blue'
        )

    if object_points is not None:
        max_points = 2000
        pts_stride = int(object_points.shape[0] // max_points)
        object_points = object_points[::pts_stride].detach().cpu().numpy()
        ax.scatter(
            object_points[:, 0],
            object_points[:, 1],
            object_points[:, 2],
            marker='o',
            c='tab:blue',
            s=100.0,
            label='object'
        )

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z (depth)')
    set_axes_equal(ax)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.invert_zaxis()

    return fig

def _get_camera_frustum(scale=0.05):
    """
    Internal helper function, generating a series of coordinates that visualize a camera's frustum in 3D.
    """
    # camera coordinate axes
    line_starts = [np.zeros((3,3))]
    line_ends = [np.eye(3) / 2]
    
    ijs = np.array([[1,1,1], [1,-1,1], [-1,-1,1], [-1,1,1], [1,1,1]]).reshape(5,3).T
    # the camera frustum: connection to origin
    line_starts.extend([
        np.zeros((3,4))
    ])
    line_ends.extend([
        ijs[:,:4]
    ])
    # the camera frustum: rectangle
    line_starts.extend([
        ijs[:,:4]
    ])
    line_ends.extend([
        ijs[:,-4:]
    ])
    return scale*np.concatenate(line_starts, axis=1), scale*np.concatenate(line_ends, axis=1)


def set_axes_equal(ax):
    """
    Implement Matplotlib's ax.set_aspect('equal') and ax.axis('equal') for 3D.

    Inputs:
        ax      matplotlib.pyplot.Axis
    """
    def set_axes_radius(ax, origin, radius):
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    limits = np.array([ax.get_xlim3d(),
                       ax.get_ylim3d(),
                       ax.get_zlim3d()])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)
