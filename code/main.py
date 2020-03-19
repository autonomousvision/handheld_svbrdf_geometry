import json
import os
from tqdm import tqdm
import cv2

from data import DataAdapterFactory
from experiment_state import ExperimentState

from utils.logging import error

# any key in here is replaced in actual path settings by its value
path_localization = {
    "<input_data_base_folder>": "/is/rg/avg/projects/mobile_lightstage/captures/20191029_benchmark_objects_dense",
    "<output_base_folder>": "/tmp/sdonne/BRDF_refactoring",
    "<calibration_base_folder>": "/is/rg/avg/projects/mobile_lightstage/calibration"
}
experiment_settings = {
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
        'brdf': 'cook torrance',
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
        'optimization_settings': {
            'targets': [],
        },
        'visualize_initial': False,
        'visualize_results': True,
    },
    'optimization_steps': [
        {
            'optimization_settings': {
                'targets': ['locations']
            }
        }
    ],
}

def localize_settings(data_settings, local_paths):
    local_data_settings = dict(data_settings)
    for path_key in local_paths:
        for data_key in local_data_settings:
            if isinstance(local_data_settings[data_key], str):
                local_data_settings[data_key] = local_data_settings[data_key].replace(path_key, local_paths[path_key])
            elif isinstance(local_data_settings[data_key], dict):
                local_data_settings[data_key] = localize_settings(local_data_settings[data_key], local_paths)
    return local_data_settings

def save_settings(experiment_settings, name, index=None):
    full_name = name + ("" if index is None else "_%d" % index)
    os.makedirs(experiment_settings['local_data_settings']['output_path'], exist_ok=True)
    settings_file = os.path.join(experiment_settings['local_data_settings']['output_path'], "%s.json" % full_name)
    subsettings = experiment_settings[name]
    if index is not None:
        subsettings = subsettings[index]
    with open(settings_file, "wt") as fh:
        return json.dump(subsettings, fh, indent=4)

def get_state_folder(experiment_settings, name, index=None):
    """
    Returns the folder name for the stored state for "name".
    """
    full_name = name + ("" if index is None else "_%d" % index)
    return os.path.join(
        experiment_settings['local_data_settings']['output_path'],
        "stored_states",
        full_name
    )

def check_stored_result(experiment_settings, name, index=None, non_critical=[]):
    """
    Checks if a stored settings file is available, reflecting that that step has been previously run.
    Returns False if no such settings are available, or True if they are available and match the provided settings.
    Raises an error if the stored settings are incompatible with the current settings.
    One can pass non-critical settings that don't necessarily need to match.
    """
    full_name = name + ("" if index is None else "_%d" % index)
    settings_file = os.path.join(experiment_settings['local_data_settings']['output_path'], "%s.json" % full_name)
    if os.path.exists(settings_file):
        with open(settings_file, "rt") as fh:
            stored_settings = json.load(fh)
        subsettings = experiment_settings[name]
        if index is not None:
            subsettings = subsettings[index]
        if not (
            {k: v for k, v in subsettings.items() if k not in non_critical}
            ==
            {k: v for k, v in stored_settings.items() if k not in non_critical}
        ):
            error("The stored settings in '%s' are not compatible with the defined '%s'" % (
                settings_file,
                name + ("" if index is None else "[%d]" % index)
            ))
        else:
            return True
    else:
        return False

# localize the experiment settings
experiment_settings['local_data_settings'] = localize_settings(experiment_settings['data_settings'], path_localization)
experiment_settings['local_initialization_settings'] = localize_settings(experiment_settings['initialization_settings'], path_localization)

# create an empty experiment object, with the correct parametrizations
check_stored_result(experiment_settings, "parametrization_settings")
experiment_state = ExperimentState.create(experiment_settings['parametrization_settings'])
save_settings(experiment_settings, "parametrization_settings")

# create the data adapter
check_stored_result(experiment_settings, "data_settings")
data_adapter = DataAdapterFactory(
    experiment_settings['data_settings']['data_type']
)(
    experiment_settings['local_data_settings']
)
observations = data_adapter.images
if not experiment_settings['data_settings']['lazy_image_loading']:
    [observation.get_image() for observation in tqdm(observations, desc="Preloading images")]
save_settings(experiment_settings, "data_settings")

# initialize the parametrizations with the requested values, if the initialization is not available on disk
initialization_state_folder = get_state_folder(experiment_settings, "initialization")
if check_stored_result(experiment_settings, "initialization_settings"):
    experiment_state.load(initialization_state_folder)
else:
    experiment_state.initialize(data_adapter, experiment_settings['local_initialization_settings'])
    experiment_state.save(initialization_state_folder)
    save_settings(experiment_settings, "initialization_settings")


observation0, occlusions0 = experiment_state.extract_observations(data_adapter, 0)
observation0 = experiment_state.locations.create_image(observation0 * (occlusions0 == 0).float())
observation1, occlusions1 = experiment_state.extract_observations(data_adapter, 1)
observation1 = experiment_state.locations.create_image(observation1 * (occlusions1 == 0).float())

simulation0 = experiment_state.simulate(observation_index=0, light_info=data_adapter.images[0].light_info)
simulation0 = experiment_state.locations.create_image(simulation0)
simulation1 = experiment_state.simulate(observation_index=1, light_info=data_adapter.images[1].light_info)
simulation1 = experiment_state.locations.create_image(simulation1)

cv2.imwrite("/tmp/sdonne/debug_images/test0_observation.png", observation0.detach().cpu().numpy()/256)
cv2.imwrite("/tmp/sdonne/debug_images/test1_observation.png", observation1.detach().cpu().numpy()/256)
cv2.imwrite("/tmp/sdonne/debug_images/test0_simulation.png", simulation0.detach().cpu().numpy()/256)
cv2.imwrite("/tmp/sdonne/debug_images/test1_simulation.png", simulation1.detach().cpu().numpy()/256)

print("hi")

optimization_step_settings = experiment_settings['default_optimization_settings']
check_stored_result(experiment_settings, "default_optimization_settings")

for step_index in range(len(experiment_settings['optimization_steps'])):
    step_state_folder = get_state_folder(experiment_settings, "optimization_steps", step_index)
    if check_stored_result(
        experiment_state,
        "optimization_steps",
        step_index,
        non_critical=[],
    ):
        experiment_state.load(step_state_folder)
    else:
        optimize(
            experiment_state,
            observations,
            experiment_settings["optimization_steps"][step_index],
        )
        experiment_state.save(step_state_folder)
    visualize(
        experiment_state,
        experiment_settings,
        get_optimization_step_shorthand(
            experiment_settings["optimization_steps"],
            step_index
        )
    )
    save_settings(experiment_settings, "optimization_steps", step_index)
