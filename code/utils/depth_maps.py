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

import torch

def depth_map_to_locations(depth, invK, invRt):
    """
    Create a point cloud from a depth map

    Inputs:
        depth           HxW torch.tensor with the depth values
        invK            3x4 torch.tensor with the inverse intrinsics matrix
        invRt           3x4 torch.tensor with the inverse extrinsics matrix

    Outputs:
        points          HxWx3 torch.tensor with the 3D points
    """
    H,W = depth.shape[:2]
    depth = depth.view(H,W,1)

    xs = torch.arange(0, W).float().reshape(1,W,1).to(depth.device).expand(H,W,1)
    ys = torch.arange(0, H).float().reshape(H,1,1).to(depth.device).expand(H,W,1)
    zs = torch.ones(1,1,1).to(depth.device).expand(H,W,1)
    world_coords = depth * (torch.cat((xs, ys, zs), dim=2) @ invK.T[None]) @ invRt[:3,:3].T[None] + invRt[:3,3:].T[None]
    return world_coords
