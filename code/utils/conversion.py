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
