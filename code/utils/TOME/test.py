import torch
import TOME
import numpy as np
import cv2
from settings import get_settings_ximea as get_settings
from data_adapters.adapter import load_data_adapter


def permutohedral_test():
    lenna_cpu = cv2.imread("/is/sg/sdonne/Pictures/wallpapers/fairy-grove_2560x1440.jpg")[400:800, 1000:1500]
    lenna_torch = torch.tensor(lenna_cpu, device=torch.device('cuda:0'), dtype=torch.float32)
    H, W = lenna_torch.shape[:2]

    xs = torch.ones(H, 1) @ torch.arange(0, W).view(1,-1).float()
    ys = torch.arange(0, H).view(-1,1).float() @ torch.ones(1, W)

    spatial_support = 16
    rgb_support = 16

    cv2.imwrite("/tmp/sdonne/debug_images/ph_in.png", lenna_cpu)

    positions = torch.cat((
        xs[:,:,None].cuda() / spatial_support,
        ys[:,:,None].cuda() / spatial_support,
        lenna_torch / rgb_support,
    ), dim=2)

    # testing the weighting

    
    weights = torch.zeros_like(lenna_torch).uniform_()
    result = TOME.permutohedral_filter(
        lenna_torch,
        positions,
        weights,
        False
    )
    cv2.imwrite("/tmp/sdonne/debug_images/ph_out_bilateral_random_weighted.png", result.cpu().numpy())

    weights = 1./ (192 - lenna_torch).clamp(min = 1/255.).min(dim=2, keepdim=True)[0]
    result = TOME.permutohedral_filter(
        lenna_torch,
        positions,
        weights,
        False
    )
    cv2.imwrite("/tmp/sdonne/debug_images/ph_out_bilateral_weighted.png", result.cpu().numpy())
    cv2.imwrite("/tmp/sdonne/debug_images/weights.png", weights.cpu().numpy())

    # continuing with uniform weights

    weights = torch.ones_like(positions)[:,:,0]
    result = TOME.permutohedral_filter(
        lenna_torch,
        positions,
        weights,
        False
    )
    cv2.imwrite("/tmp/sdonne/debug_images/ph_out_bilateral.png", result.cpu().numpy())

    positions = torch.cat((
        xs[:,:,None].cuda() / spatial_support,
        ys[:,:,None].cuda() / spatial_support,
    ), dim=2)
    result = TOME.permutohedral_filter(
        lenna_torch,
        positions,
        weights,
        False
    )
    cv2.imwrite("/tmp/sdonne/debug_images/ph_out_spatial.png", result.cpu().numpy())
    cv2.imwrite("/tmp/sdonne/debug_images/ph_out_spatial_red_x3.png", result[:,:,2].cpu().numpy())

    result = TOME.permutohedral_filter(
        lenna_torch[:,:,1:].contiguous(),
        positions,
        weights,
        False
    )

    cv2.imwrite("/tmp/sdonne/debug_images/ph_out_spatial_red_x2.png", result[:,:,1].cpu().numpy())
    result = TOME.permutohedral_filter(
        lenna_torch[:,:,2:3].contiguous(),
        positions,
        weights,
        False
    )
    cv2.imwrite("/tmp/sdonne/debug_images/ph_out_spatial_red_x1.png", result[:,:,0].cpu().numpy())

def bound_test():
    ctr_view = 0
    side_view = 4

    input_scene_name = 'cups_stack'

    # load the scene settings
    settings_ctr = get_settings(input_scene_name=input_scene_name, view=ctr_view)

    adapter_ctr = load_data_adapter(settings_ctr, view=ctr_view)

    settings_side = get_settings(input_scene_name=input_scene_name, view=side_view)
    adapter_side = load_data_adapter(settings_side, view=side_view)

    ctr_invextr = adapter_ctr.images[0].inv_extrinsics
    ctr_invintr = adapter_ctr.images[0].inv_intrinsics
    sid_invextr = adapter_side.images[0].inv_extrinsics
    sid_invintr = adapter_side.images[0].inv_intrinsics

    ctr_camera = np.matmul(
        np.linalg.inv(ctr_invintr),
        np.linalg.inv(np.concatenate(
            (ctr_invextr, np.array([0,0,0,1]).reshape(1,4)),
            axis=0
        ))[:3]
    )
    side_camera = np.matmul(
        np.linalg.inv(sid_invintr),
        np.linalg.inv(np.concatenate(
            (sid_invextr, np.array([0,0,0,1]).reshape(1,4)),
            axis=0
        ))[:3]
    )

    ctr_invKR = np.matmul(ctr_invextr[:,:3], ctr_invintr)
    ctr_camloc = ctr_invextr[:,3:]
    side_invKR = np.matmul(sid_invextr[:,:3], sid_invintr)
    side_camloc = sid_invextr[:,3:]

    to_tensor = lambda x: torch.from_numpy(x.astype(np.float32)).cuda()
    to_np = lambda x: x.detach().cpu().numpy()

    outH = 3008
    outW = 4112
    depth = to_tensor(adapter_ctr.images[0].depth_image)[None]
    target_depth = to_tensor(adapter_side.images[0].depth_image)

    def filter_depth(x):
        x[x<0.5] = 0
        maxpool = torch.nn.MaxPool2d(5,1,2,1)
        mask = maxpool((x == 0).float()) > 0
        x[mask] = 0
        return x

    depth = filter_depth(depth)
    target_depth = filter_depth(target_depth[None])

    cameras = to_tensor(side_camera).reshape(1,1,3,4)
    invKR = to_tensor(ctr_invKR).reshape(1,3,3)
    camloc = to_tensor(ctr_camloc).reshape(1,3,1)
    reprojected = TOME.depth_reprojection(depth, cameras, invKR, camloc, outH, outW)

    camera = to_tensor(ctr_camera).reshape(1,3,4)
    invKRs = to_tensor(side_invKR).reshape(1,1,3,3)
    camlocs = to_tensor(side_camloc).reshape(1,1,3,1)
    dmin = 0.5
    dmax = 2.0
    dstep = 0.001
    bounded = TOME.depth_reprojection_bound(depth, camera, invKRs, camlocs, outH, outW, dmin, dmax, dstep)

    filtered = reprojected * (reprojected < bounded + 0.001).float()

    cv2.imwrite("/tmp/sdonne/debug_images/00000.png", to_np(reprojected).squeeze()*100)
    cv2.imwrite("/tmp/sdonne/debug_images/00001.png", to_np(bounded).squeeze()*100)
    cv2.imwrite("/tmp/sdonne/debug_images/00002.png", to_np(filtered).squeeze()*100)
    cv2.imwrite("/tmp/sdonne/debug_images/00003.png", to_np(target_depth).squeeze()*100)

    print("Finished reproj/bound test")

def splat_test():
    ctr_view = 0

    settings_ctr = get_settings()
    adapter_ctr = load_data_adapter(settings_ctr, view=ctr_view)

    ctr_invextr = adapter_ctr.images[0].inv_extrinsics
    ctr_invintr = adapter_ctr.images[0].inv_intrinsics
    sid_invextr = ctr_invextr.copy()
    sid_invintr = ctr_invintr.copy()

    sid_invextr[0,3] += 0.5
    sid_superres = 2
    sid_invintr[:2,:2] /= sid_superres

    ctr_camera = np.matmul(
        np.linalg.inv(ctr_invintr),
        np.linalg.inv(np.concatenate(
            (ctr_invextr, np.array([0,0,0,1]).reshape(1,4)),
            axis=0
        ))[:3]
    )
    side_camera = np.matmul(
        np.linalg.inv(sid_invintr),
        np.linalg.inv(np.concatenate(
            (sid_invextr, np.array([0,0,0,1]).reshape(1,4)),
            axis=0
        ))[:3]
    )

    ctr_invKR = np.matmul(ctr_invextr[:,:3], ctr_invintr)
    ctr_camloc = ctr_invextr[:,3:]
    side_invKR = np.matmul(sid_invextr[:,:3], sid_invintr)
    side_camloc = sid_invextr[:,3:]

    to_tensor = lambda x: torch.from_numpy(x.astype(np.float32)).cuda()
    to_np = lambda x: x.detach().cpu().numpy()

    outH = 1080 * sid_superres
    outW = 1920 * sid_superres
    depth = to_tensor(adapter_ctr.images[0].depth_image)[None]

    def filter_depth(x):
        x[x<0.5] = 0
        maxpool = torch.nn.MaxPool2d(5,1,2,1)
        mask = maxpool((x == 0).float()) > 0
        x[mask] = 0
        return x

    depth = filter_depth(depth)

    cameras = to_tensor(side_camera).reshape(1,1,3,4)
    invKR = to_tensor(ctr_invKR).reshape(1,3,3)
    camloc = to_tensor(ctr_camloc).reshape(1,3,1)

    splatted, weights, visibilities = TOME.depth_reprojection_splat(depth, cameras, invKR[None], camloc[None], 1.0, 1.0, 1.0, outH, outW)
    splatted[splatted!=splatted]=0
    cv2.imwrite("/tmp/sdonne/debug_images/00004.png", to_np(splatted).squeeze()*100)
    i = 4
    for radius in [0.5, 1.0, 2.0]:
        for visibility_scale in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
            splatted, weights, visibilities = TOME.depth_reprojection_splat(depth, cameras, invKR[None], camloc[None], radius, 1.0, visibility_scale, outH, outW)
            splatted[splatted!=splatted]=0
            i+=1
            cv2.imwrite("/tmp/sdonne/debug_images/%05d.png"%i, to_np(visibilities).squeeze()*255)
    i+=1
    cv2.imwrite("/tmp/sdonne/debug_images/%05d.png"%i, to_np(depth).squeeze()*100)

    print("Finished splat test")

if __name__ == "__main__":
    permutohedral_test()
    # bound_test()
    # splat_test()