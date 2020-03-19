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
    this_norm[this_norm == 0] = 1.0
    result = this / this_norm
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
    this.data.cdiv_(this_norm)

def norm(this, keepdim=True):
    """
    Calculate the tensor norm along the last dimension.

    Inputs:
        this                ...xD torch.tensor containing the vector

    Outputs:
        result              ...x1 torch.tensor containing the normalized version
    """
    return this.norm(dim=-1, keepdim=keepdim)
