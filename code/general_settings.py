"""
This file collects settings that mostly should not be affecting the results,
but only the run-time behaviour, such as batch size, or point cloud saving.
"""

# size of the fattening for the calculated observation occlusion
occlusion_fattening = 4

# size of the fattening for the simulated shadowing
shadow_fattening = 4

# device to store the tensors on. This is also the computation device.
device_name = 'cuda:0'

# intensity scale for visualizations. Loss terms are scaled with this, too.
intensity_scale = 5e-4

# whether or not to save point clouds in the experiment_state visualize() function
save_clouds = True

# how often to plot optimization progress (in iterations)
evolution_plot_frequency = 200

# the maximum batch size for simulation/extraction/loss evaluation
batch_size = 20

# localization translation dictionary for settings
path_localization = {
    "<input_data_base_folder>": "/is/rg/avg/projects/mobile_lightstage/captures/20191029_benchmark_objects_dense",
    "<output_base_folder>": "/tmp/sdonne/BRDF_refactoring",
    "<calibration_base_folder>": "/is/rg/avg/projects/mobile_lightstage/calibration"
}