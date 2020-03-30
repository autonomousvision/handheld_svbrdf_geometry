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
import open3d as o3d

from general_settings import device_name

device = torch.device(device_name)

def to_numpy(x):
    """
    Converts a torch.tensor to a np.ndarray
    """
    return x.detach().cpu().numpy()

def to_torch(x):
    """
    Converts a np.ndarray to a torch.tensor
    """
    return torch.tensor(x).to(device)

def to_o3d(x):
    """
    Converts a np.ndarray to an open3d-compatible vector.

    Inputs:
        x           Nx3 torch.tensor

    Outputs:
        y           o3d.utility.Vector3dVector
    """
    return o3d.utility.Vector3dVector(x)
