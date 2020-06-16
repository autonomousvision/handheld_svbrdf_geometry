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

import json
import os
from tqdm import tqdm
import cv2
import torch
import itertools

from data import DataAdapterFactory
from experiment_state import ExperimentState
from experiment_settings import ExperimentSettings, recursive_dict_update
from optimization import optimize
from higo import higo_baseline
import general_settings
from evaluation import evaluate_state

from utils.logging import error

settings_dict_base = {
    'data_settings': {
        'data_type': "XIMEA",
        'center_view': None,
        'nr_neighbours': 40, 
        'base_input_path': "<input_data_base_folder>/",
        'object_name': None,
        'base_output_path': "<output_base_folder>/",
        'gt_scan_folder': "<gt_scan_folder>/",
        'calibration_path_geometric': "<calibration_base_folder>/geometric/",
        'vignetting_file': '<calibration_base_folder>/photometric/vignetting.npz',
        'depth_folder': 'tsdf-fusion-depth_oldCV_40_views',
        'center_stride': 2,
        'depth_scale': 1e-3,
        'light_scale': 1e0,
        'lazy_image_loading': False,
        'input_reup_sample': 1.,
        'input_down_sample': 1.,
        'manual_center_view_crop': None,
    },
    'parametrization_settings': {
        'locations': 'depth map',
        'normals': 'per point',
        'materials': 'base specular materials',
        'brdf': 'cook torrance F1',
        'observation_poses': 'quaternion',
        'lights': 'point light',
    },
    'initialization_settings': {
        'normals': 'from_closed_form',
        'diffuse': 'from_closed_form',
        'specular': 'hardcoded',
        'lights': 'precalibrated',
        'light_calibration_files': {
            "positions": "<calibration_base_folder>/photometric/lights_array.pkl",
            "intensities": "<calibration_base_folder>/photometric/LED_light_intensities.npy",
            "attenuations": "<calibration_base_folder>/photometric/LED_angular_dependency.npy",
        }
    },
    'default_optimization_settings': {
        'parameters': [
            'locations',
            'poses',
            'normals',
            'vignetting',
            'diffuse_materials',
            'specular_weights',
            'specular_materials',
            'light_positions',
            'light_intensities',
            'light_attenuations',
        ],
        'losses': {
            "photoconsistency L1": 1e-4,
            "geometric consistency": 1e1,
            "depth compatibility": 1e10,
            "normal smoothness": 1e-1,
            "material sparsity": 1e-1,
            "material smoothness": 1e0,
        },
        "iterations": 1000,
        'visualize_initial': False,
        'visualize_results': True,
        'target_set': "training",
    },
    'optimization_steps': [],
}

settings_dict_higo = recursive_dict_update(
    settings_dict_base,
    {
        'data_settings': {
            'output_path_suffix': 'higo',
        },
        'higo_baseline': {
            'step_size': 0.001, #1 mm
            'step_radius': 25,
            'eta': 10,
            'lambda_n': 7.5,
            'lambda_s': 3.0,
            'lambda_1': 0.1,
            'lambda_2': 0.5,
        },
    }
)

disjoint_iteration = [
    {
        'parameters': ['specular_weights'],
        "iterations": 40,
    },
    {
        'parameters': ['diffuse_materials', 'specular_materials'],
        "iterations": 40,
    },
    {
        'parameters': ['normals'],
        "iterations": 60,
    },
    {
        'parameters': ['locations', 'observation_poses'],
        "iterations": 60,
    },
]
settings_dict_disjoint = recursive_dict_update(
    settings_dict_base,
    {
        'data_settings': {
            'output_path_suffix': 'disjoint',
        },
        'default_optimization_settings': {
        },
        'optimization_steps': [
            *(disjoint_iteration * 5),
            # {
            #     'iterations': 1000,
            #     'parameters': [
            #         'observation_poses',
            #     ],
            #     'visualize_initial': True,
            #     'target_set': "testing",
            # },
        ]
    }
)

settings_dict_proposed = recursive_dict_update(
    settings_dict_base,
    {
        'data_settings': {
            'output_path_suffix': 'proposed',
        },
        'optimization_steps': [
            {
                'parameters': [
                    'diffuse_materials',
                    'specular_weights',
                    'specular_materials',
                    'normals',
                    'observation_poses',
                    'locations',
                ],
                'visualize_initial': True,
            },
            # {
            #     'parameters': [
            #         'observation_poses',
            #     ],
            #     'visualize_initial': True,
            #     'target_set': "testing",
            # },
        ]
    }
)


def run_experiment(settings_dict):
    experiment_settings = ExperimentSettings(settings_dict)
    # localize to the current computer as required
    experiment_settings.localize()

    # create an empty experiment object, with the correct parametrizations
    experiment_settings.check_stored("parametrization_settings")
    experiment_state = ExperimentState.create(experiment_settings.get('parametrization_settings'))
    experiment_settings.save("parametrization_settings")

    # create the data adapter
    experiment_settings.check_stored("data_settings")
    data_adapter = DataAdapterFactory(
        experiment_settings.get('data_settings')['data_type']
    )(
        experiment_settings.get('local_data_settings')
    )

    if not experiment_settings.get('data_settings')['lazy_image_loading']:
        device = torch.device(general_settings.device_name)
        image_tensors = [
            observation.get_image()
            for observation in tqdm(data_adapter.images, desc="Preloading training images")
        ]
        # now compact all observations into a few big tensors and remove the old tensors
        # this makes for much faster access/operations
        data_adapter.compound_image_tensors = {}
        data_adapter.compound_image_tensor_sizes = {}
        training_indices_batches, _ = data_adapter.get_training_info()
        testing_indices_batches, _ = data_adapter.get_testing_info()
        for batch in itertools.chain(training_indices_batches, testing_indices_batches):
            compound_H = max([image_tensors[i].shape[-2] for i in batch])
            compound_W = max([image_tensors[i].shape[-1] for i in batch])
            C = len(batch)
            compound_images = torch.zeros(C, 3, compound_H, compound_W, dtype=torch.float, device=device)
            compound_sizes = torch.zeros(C, 2, dtype=torch.long, device=device)
            for idx, image_idx in enumerate(batch):
                src_tensor = image_tensors[image_idx]
                compound_images[idx,:,:src_tensor.shape[-2], :src_tensor.shape[-1]] = src_tensor
                compound_sizes[idx,0] = src_tensor.shape[-1]
                compound_sizes[idx,1] = src_tensor.shape[-2]
                del data_adapter.images[image_idx]._image
            data_adapter.compound_image_tensors[batch] = compound_images
            data_adapter.compound_image_tensor_sizes[batch] = compound_sizes
        del image_tensors

    experiment_settings.save("data_settings")

    # initialize the parametrizations with the requested values, if the initialization is not available on disk
    initialization_state_folder = experiment_settings.get_state_folder("initialization")
    if experiment_settings.check_stored("initialization_settings"):
        experiment_state.load(initialization_state_folder)
    else:
        experiment_state.initialize(data_adapter, experiment_settings.get('local_initialization_settings'))
        experiment_state.save(initialization_state_folder)
        experiment_settings.save("initialization_settings")

    # evaluate_state("initialization",
    #                experiment_settings.get('data_settings')['object_name'],
    #                experiment_settings.get('local_data_settings')['gt_scan_folder'],
    #                experiment_state)
    experiment_state.visualize_statics(
        experiment_settings.get_output_path(),
        data_adapter
    )

    if experiment_settings.get("higo_baseline") is not None:
        higo_state_folder = experiment_settings.get_state_folder("higo")
        if not experiment_settings.check_stored("higo_baseline"):
            higo_experiment_state = higo_baseline(
                experiment_state,
                data_adapter,
                higo_state_folder,
                experiment_settings.get('higo_baseline')
            )
            higo_experiment_state.visualize(
                experiment_settings.get_output_path(),
                "higo_baseline",
                data_adapter,
                losses = [],
                shadows_occlusions=False
            )
            higo_experiment_state.save(higo_state_folder)
            experiment_settings.save("higo_baseline")
        else:
            higo_experiment_state = ExperimentState.copy(experiment_state)
            higo_experiment_state.load(higo_state_folder)
        evaluate_state("higo baseline",
                       experiment_settings.get('data_settings')['object_name'],
                       experiment_settings.get('local_data_settings')['gt_scan_folder'],
                       higo_experiment_state)

    optimization_step_settings = experiment_settings.get('default_optimization_settings')
    experiment_settings.check_stored("default_optimization_settings")
    experiment_settings.save("default_optimization_settings")

    for step_index in range(len(experiment_settings.get('optimization_steps'))):
        step_state_folder = experiment_settings.get_state_folder("optimization_steps", step_index)

        optimization_settings = experiment_settings.get("optimization_steps", step_index)

        shorthand = experiment_settings.get_shorthand("optimization_steps", step_index)
        set_name = "%02d_%s" % (step_index, shorthand)

        if optimization_settings['visualize_initial']:
            experiment_state.visualize(
                experiment_settings.get_output_path(),
                "%02d__initial" % step_index,
                data_adapter,
                optimization_settings['losses']
            )

        if experiment_settings.check_stored("optimization_steps", step_index):
            experiment_state.load(step_state_folder)
        else:
            optimize(
                experiment_state,
                data_adapter,
                optimization_settings,
                output_path_structure=os.path.join(
                    experiment_settings.get_output_path(),
                    "evolution_%%s_%s.png" % set_name
                )
            )
            experiment_state.save(step_state_folder)
            experiment_settings.save("optimization_steps", step_index)

        if optimization_settings['visualize_results']:
            experiment_state.visualize(
                experiment_settings.get_output_path(),
                set_name,
                data_adapter,
                optimization_settings['losses']
            )

    evaluate_state(experiment_settings.get('data_settings').get('output_path_suffix', 'proposed'),
                   experiment_settings.get('data_settings')['object_name'],
                   experiment_settings.get('local_data_settings')['gt_scan_folder'],
                   experiment_state)

if __name__ == "__main__":
    for object_name, center_view in [
        # ("peter",         1171),
        # ("peter",          566),
        # ("teapot",         451),
        # ("teapot",         999),
        # ("gnome",          308),
        # ("gnome",          488),
        # ("girl",           882),
        # ("girl",          1059),
        ("duck",           826),
        # ("duck",            49),
        # ("fire_hydrant",   582),
        # ("fire_hydrant",   704),
        # ("pineapple",      401),
        # ("pineapple",      536),
        # ("bunny",          670),
        # ("bunny",          180),
    ]:
        for settings_dict in [settings_dict_proposed, settings_dict_higo, settings_dict_disjoint]:
            settings_dict['data_settings']['object_name'] = object_name
            settings_dict['data_settings']['center_view'] = center_view
            run_experiment(settings_dict)

