import torch
import open3d as o3d

from general_settings import device_name

device = torch.device(device_name)

def to_numpy(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.tensor(x).to(device)

def to_o3d(x):
    return o3d.utility.Vector3dVector(x)
