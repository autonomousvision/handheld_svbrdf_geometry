import json
import os
from tqdm import tqdm
import cv2

from data import DataAdapterFactory
from experiment_state import ExperimentState
from experiment_settings import ExperimentSettings
from optimization import optimize

from utils.logging import error

# any key in here is replaced in actual path settings by its value
experiment_settings = ExperimentSettings({
    'data_settings': {
        'data_type': "XIMEA",
        'center_view': 180,
        'nr_neighbours': 20, 
        'input_path': "<input_data_base_folder>/bunny/",
        'output_path': "<output_base_folder>/development_test/",
        'calibration_path_geometric': "<calibration_base_folder>/geometry/calib-20191002/",
        'vignetting_file': '<calibration_base_folder>/photometric/20190822_vignettes_light_intensities_attenuation/vignetting.npz',
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
            "positions": "<calibration_base_folder>/photometric/20190717_light_locations/lights_array.pkl",
            "intensities": "<calibration_base_folder>/photometric/20190822_vignettes_light_intensities_attenuation/LED_light_intensities.npy",
            "attenuations": "<calibration_base_folder>/photometric/20190822_vignettes_light_intensities_attenuation/LED_angular_dependency.npy",
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
            "normal smoothness": 1e0,
            "material sparsity": 1e-1,
            "material smoothness": 1e0
        },
        "iterations": 1000,
        "plotting_frequency": 1,
        'visualize_initial': False,
        'visualize_results': True,
    },
    'optimization_steps': [
        {
            'parameters': ['diffuse_materials', 'specular_weights', 'specular_materials']
        },
    ],
})

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
observations = data_adapter.images
if not experiment_settings.get('data_settings')['lazy_image_loading']:
    [observation.get_image() for observation in tqdm(observations, desc="Preloading images")]
experiment_settings.save("data_settings")

# initialize the parametrizations with the requested values, if the initialization is not available on disk
initialization_state_folder = experiment_settings.get_state_folder("initialization")
if experiment_settings.check_stored("initialization_settings"):
    experiment_state.load(initialization_state_folder)
else:
    experiment_state.initialize(data_adapter, experiment_settings.get('local_initialization_settings'))
    experiment_state.save(initialization_state_folder)
    experiment_settings.save("initialization_settings")

optimization_step_settings = experiment_settings.get('default_optimization_settings')
experiment_settings.check_stored("default_optimization_settings")
experiment_settings.save("default_optimization_settings")

for step_index in range(len(experiment_settings.get('optimization_steps'))):
    step_state_folder = experiment_settings.get_state_folder("optimization_steps", step_index)

    optimization_settings = experiment_settings.get("optimization_steps", step_index)

    if optimization_settings['visualize_initial']:
        experiment_state.visualize(
            experiment_settings.get('local_data_settings')['output_path'],
            step_index,
            "initial",
        )

    if experiment_settings.check_stored("optimization_steps", step_index):
        experiment_state.load(step_state_folder)
    else:
        optimize(
            experiment_state,
            data_adapter,
            optimization_settings,
        )
        experiment_state.save(step_state_folder)
    experiment_settings.save("optimization_steps", step_index)

    if optimization_settings['visualize_results']:
        experiment_state.visualize(
            experiment_settings.get('local_data_settings')['output_path'],
            step_index,
            experiment_settings.get_shorthand("optimization_steps", step_index),
        )
