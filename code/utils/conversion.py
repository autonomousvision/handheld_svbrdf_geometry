import torch

from general_settings import device_name

device = torch.device(device_name)

def to_numpy(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.tensor(x).to(device)
