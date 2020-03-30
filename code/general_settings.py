"""
Copyright (c) 2020 Simon Donné, Max Planck Institute for Intelligent Systems, Tuebingen, Germany

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