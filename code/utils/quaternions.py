import torch
from utils.vectors import normalize

def quaternion_to_R(qs, normalize_qs=True):
    """
    Convert a set of quaternions to rotation matrices

    Inputs:
        qs              Ix4 torch.tensor containing the quaternions
        normalize_qs    Whether or not to normalize quaternions before continuing (default True)

    Outputs:
        Rs              Ix3x3 torch.tensor containing the rotation matrices
    """
    qs = qs.view(-1,4)
    if normalize_qs:
        qs = normalize(qs)
    ws,xs,ys,zs = qs[:,0], qs[:,1], qs[:,2], qs[:,3]
    Rs = qs.new_empty(qs.shape[0], 3, 3)
    Rs[:,0,0] = 1 - 2*ys*ys - 2*zs*zs
    Rs[:,0,1] = 2*xs*ys - 2*zs*ws
    Rs[:,0,2] = 2*xs*zs + 2*ys*ws
    Rs[:,1,0] = 2*xs*ys + 2*zs*ws
    Rs[:,1,1] = 1 - 2*xs*xs - 2*zs*zs
    Rs[:,1,2] = 2*ys*zs - 2*xs*ws
    Rs[:,2,0] = 2*xs*zs - 2*ys*ws
    Rs[:,2,1] = 2*ys*zs + 2*xs*ws
    Rs[:,2,2] = 1 - 2*xs*xs - 2*ys*ys
    return Rs

def R_to_quaternion(Rs):
    """
    Convert a set of quaternions to rotation matrices

    Inputs:
        Rs              Ix3x3 torch.tensor containing the rotation matrices
    
    Outputs:
        qs              Ix4 torch.tensor containing the quaternions
    """
    Rs = Rs.view(-1,3,3)
    ws = (1 + Rs[:,0,0] + Rs[:,1,1] + Rs[:,2,2]).clamp_(min=0.).sqrt()
    xs = (1 + Rs[:,0,0] - Rs[:,1,1] - Rs[:,2,2]).clamp_(min=0.).sqrt()
    ys = (1 - Rs[:,0,0] + Rs[:,1,1] - Rs[:,2,2]).clamp_(min=0.).sqrt()
    zs = (1 - Rs[:,0,0] - Rs[:,1,1] + Rs[:,2,2]).clamp_(min=0.).sqrt()

    # we need four alternatives to be completely robust against weird rotations

    qs0 = Rs.new_empty(Rs.shape[0], 4)
    qs0[:,0] = ws
    qs0[:,1] = xs * torch.sign(xs * (Rs[:,2,1] - Rs[:,1,2]))
    qs0[:,2] = ys * torch.sign(ys * (Rs[:,0,2] - Rs[:,2,0]))
    qs0[:,3] = zs * torch.sign(zs * (Rs[:,1,0] - Rs[:,0,1]))
    qs1 = Rs.new_empty(Rs.shape[0], 4)
    qs1[:,0] = ws * torch.sign(ws * (Rs[:,2,1] - Rs[:,1,2]))
    qs1[:,1] = xs
    qs1[:,2] = ys * torch.sign(ys * (Rs[:,1,0] + Rs[:,0,1]))
    qs1[:,3] = zs * torch.sign(zs * (Rs[:,0,2] + Rs[:,2,0]))
    qs2 = Rs.new_empty(Rs.shape[0], 4)
    qs2[:,0] = ws * torch.sign(ws * (Rs[:,0,2] - Rs[:,2,0]))
    qs2[:,1] = xs * torch.sign(xs * (Rs[:,0,1] + Rs[:,1,0]))
    qs2[:,2] = ys
    qs2[:,3] = zs * torch.sign(zs * (Rs[:,1,2] + Rs[:,2,1]))
    qs3 = Rs.new_empty(Rs.shape[0], 4)
    qs3[:,0] = ws * torch.sign(ws * (Rs[:,1,0] - Rs[:,0,1]))
    qs3[:,1] = xs * torch.sign(xs * (Rs[:,0,2] + Rs[:,2,0]))
    qs3[:,2] = ys * torch.sign(ys * (Rs[:,1,2] + Rs[:,2,1]))
    qs3[:,3] = zs

    qs0 = normalize(qs0)
    qs1 = normalize(qs1)
    qs2 = normalize(qs2)
    qs3 = normalize(qs3)
    qs =    qs0 * (ws[:,None] > 0).float() + (ws[:,None] == 0).float() * (
            qs1 * (xs[:,None] > 0).float() + (xs[:,None] == 0).float() * (
            qs2 * (ys[:,None] > 0).float() + (ys[:,None] == 0).float() * (
            qs3)))
    qs = normalize(qs)

    return qs

