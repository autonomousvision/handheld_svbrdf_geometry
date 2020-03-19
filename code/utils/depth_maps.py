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
