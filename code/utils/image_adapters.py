from abc import ABC, abstractmethod
import os
import numpy as np
import cv2
import torch

from utils.logging import log, error
from utils.image_wrapper import ImageWrapper

class ImageWrapper:
    @staticmethod
    def load_metadata(filename):
        with open(filename, "rt") as fh:
            metadata = json.load(fh)
        return metadata

    def __init__(
        self,
        view_number,
        image_file,
        crop,
        intrinsics,
        extrinsics,
        vignetting,
        down_sample=1,
        reup_sample=1,
        lazy=False,
        device="cuda:0",
    ):
        self.view_number = view_number
        self.image_file = image_file
        self._image = None
        self.crop = crop
        self.original_intrinsics = intrinsics
        self.original_extrinsics = extrinsics
        self.original_vignetting = vignetting
        self.down_sample = down_sample
        self.reup_sample = reup_sample
        self.lazy = lazy
        self.device = torch.device(device)

    def get_intrinsics():
        if self._K is None:
            K = self.original_intrinsics.clone()
            if self.crop is not None:
                K[:2,2] -= self.crop[:,0]
            K[:2] *= self.reup_sample / self.down_sample
            self._K = K
        return self._K

    def get_image(self):
        if self._image is None:
            image_data = np.load(image_file)
            if not isinstance(data, np.ndarray):
                data = data['arr_0']
            image_info = load_metadata(image_file+".meta")
            exposure_time = self.metadata['exposure_time_us'] * 1e-6
            dark_level = float(self.metadata['black_level'])
            saturation_mask = image_data.max(axis=2) >= 4094
            image_data = np.clip((image_data.astype(np.float32) - dark_level),
                                a_min=0.0, a_max=None) / exposure_time
            if self.original_vignetting is not None:
                image_data = image_data / self.original_vignetting
            if self.crop is not None:
                self.image = self.image[
                    self.crop[1,0]:self.crop[1,1],
                    self.crop[0,0]:self.crop[0,1]
                ]
                if self.down_sample is not None:
                    color_image_data = cv2.resize(
                        color_image_data,
                        dsize=None,
                        fx=1./down_scale,
                        fy=1./down_scale,
                        interpolation=cv2.INTER_AREA
                    )
                if self.up_sample is not None:
                    color_image_data = cv2.resize(
                        color_image_data,
                        dsize=None,
                        fx=reup_scale,
                        fy=reup_scale,
                        interpolation=cv2.INTER_CUBIC
                    )
            image = torch.Tensor(color_image_data, device=self.device)
            saturation_mask = torch.Tensor(saturation_mask, device=self.device)
            if not self.lazy:
                self._image = image
                self._saturation_mask = saturation_mask
        else:
            image = self._image
            saturation_mask = self._saturation_mask

        return image, saturation_mask

class ImageAdapter(ABC):
    @staticmethod
    def load_depth_and_bbox(depth_dir, view, depth_scale, device):
        base_filename = os.path.join(depth_dir, "%05d" % center_view)
        if os.path.exists(base_filename + ".npz"):
            npz_dict = np.load(base_filename + ".npz")
            depth = npz_dict['depth']
            crop = npz_dict['bbox']
        elif os.path.exists(base_filename + ".npy"):
            depth = np.load(base_filename + ".npy")
            crop = None
        else:
            depth = None
            crop = None
        if depth is not None:
            depth = torch.Tensor(depth * depth_scale, device=device)
        return depth, crop

    @staticmethod
    def load_bbox(depth_dir, view):
        base_filename = os.path.join(depth_dir, "%05d" % center_view)
        if os.path.exists(base_filename + ".npz"):
            npz_dict = np.load(base_filename + ".npz")
            crop = npz_dict['bbox']
        else:
            crop = None
        return crop

    @staticmethod
    def load_pose(color_dir, view, depth_scale, device):
        pose_file = os.path.join(color_dir, '%05d.pose' % view)
        if os.path.exists(pose_file):
            with open(pose_file, "rb") as fh:
                pose = pickle.load(fh)
            pose[:3,3:] *= depth_scale
            pose = torch.Tensor(
                np.concatenate(
                    (pose, np.array([0, 0, 0, 1]).reshape(1, 4)),
                    axis=0
                ),
                device=device,
            )
        else:
            error("Pose file '%s' does not exist." % pose_file)
        return pose

    @abstractmethod
    def __init__(self, center_view, observation_views):
        self.center_view = center_view
        self.center_invpose = None
        self.center_depth = None
        self.center_invK = None
        self.images = [None for view in observation_views]


class XimeaAdapter(ImageAdapter):
    def __init__(self, center_view, observation_views, data_settings):
        self.center_view = center_view
        
        color_dir = os.path.join(
            self.data_settings['base_directory'],
            "RGB_undistorted",
        )

        depth_dir = os.path.join(
            self.data_settings['base_directory'],
            self.data_settings['depth_directory'],
        )
        device = torch.device(data_settings['device'])

        K_color = torch.Tensor(np.load(os.path.join(
            data_settings['geometric_calibration_folder'], "stereo_calibration.npz"
        ))['intrinsic_color'], device=device)

        self.center_crop = center_crop
        
        self.center_invpose = Adapter.load_pose(
            color_dir,
            center_view,
            data_settings['depth_scale'],
            device
        ).inverse()

        center_K = K_color.clone()
        center_depth, center_crop = Adapter.load_depth_and_bbox(
            depth_dir,
            center_view,
            data_settings['depth_scale'],
            device
        )
        if self.data_settings['center_view_crop'] is not None:
            log("Manually cropping the center view")
            center_crop = self.settings['center_view_crop']

        if center_depth is None:
            log(
                "Warning> missing depth for view %05d. Filled in to the Z=0 plane"
                % center_view
            )
            pixels_y, pixels_x = torch.meshgrid([
                torch.arange(0, image_size[0]),
                torch.arange(0, image_size[1]),
            ])
            pixels = torch.stack(
                [pixels_x, pixels_y, torch.ones_like(pixels_y)],
                dim=2
            ).float().to(device)

            # pixel rays in camera space
            pixels_normed = pixels @ K_color.inverse().T[None]
            center_depth = - self.center_invpose[2,3] / (
                pixels_normed @ self.center_invpose[2:3,:3].T[None]
            ).squeeze()

        if center_crop is not None:
            center_depth = center_depth[
                center_crop[1,0]:center_crop[1,1],
                center_crop[0,0]:center_crop[0,1],
            ]
            center_K[0,2] -= center_crop[0]
            center_K[1,2] -= center_crop[0]

        self.center_depth = center_depth
        self.center_invK = center_K.inverse()

        self.images = [
            ImageWrapper(
                view,
                os.path.join(color_dir, "%05d.npy" % view),
                Adapter.load_bbox(
                    depth_dir, view, data_settings['depth_scale'], device
                ),
                K_color,
                Adapter.load_pose(
                    color_dir, view, data_settings['depth_scale'], device
                ),
                vignetting,
                data_settings['input_down_sample'],
                data_settings['input_reup_sample'],
                data_settings['lazy_image_loading'],
                data_settings['device'],
            )
            for view in observation_views
        ]


def data_adapter_factory(name):
    if name == "XimeaAdapter":
        return XimeaAdapter
    else:
        error("Data adapter class '%s' is not known.")
