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

def inner_product(this, that, keepdim=True):
    """
    Calculate the inner product of two tensors along the last dimension

    Inputs:
        this, that          ...xD torch.tensors containing the vectors
    
    Outputs:
        result              ... torch.tensor containing the inner products
    """
    return torch.sum(this*that, dim=-1, keepdim=keepdim)

def cross_product(this, that, keepdim=True):
    """
    Calculate the cross product of two tensors along the last dimension

    Inputs:
        this, that          ...xD torch.tensors containing the vectors
    
    Outputs:
        result              ... torch.tensor containing the cross products
    """
    return torch.cross(this, that, dim=-1)

def normalize(this):
    """
    Normalize a tensor along the last dimension.

    Inputs:
        this                ...xD torch.tensor containing the vector

    Outputs:
        result              ...xD torch.tensor containing the normalized version
    """
    this_norm = norm(this)
    result = this / (this_norm + (this_norm == 0).float())
    return result

def normalize_(this):
    """
    Normalize a tensor along the last dimension, acting in-place on its data.
    No backpropagation is supported.

    Inputs:
        this                ...xD torch.tensor containing the vector
    """
    this_norm = norm(this.data)
    this_norm[this_norm == 0] = 1.0
    this.data.div_(this_norm + (this_norm == 0).float())

def norm(this, keepdim=True):
    """
    Calculate the tensor norm along the last dimension.

    Inputs:
        this                ...xD torch.tensor containing the vector

    Outputs:
        result              ...x1 torch.tensor containing the normalized version
    """
    return this.norm(dim=-1, keepdim=keepdim)
