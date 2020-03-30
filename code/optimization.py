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

import os
import torch
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

from losses import LossFunctionFactory
import general_settings

from utils.logging import error

def optimize(experiment_state, data_adapter, optimization_settings, output_path_structure=None):
    """
    Optimize the current experiment_state, given the observations in data_adapter and the optimization_settings.
    Optionally saves optimization progress plots to the output_path_structure, which is formatted as
    output_path_structure % plot_name.
    """
    parameter_dictionary, learning_rate_dictionary, visualizer_dictionary = experiment_state.get_parameter_dictionaries()
    device = torch.device(general_settings.device_name)

    parameter_groups = []
    for parameter in optimization_settings['parameters']:
        if not parameter in parameter_dictionary:
            error("Cannot optimize over %s, as it is not part of the chosen parametrization." % parameter)
        parameter_groups.append({
            'params': parameter_dictionary[parameter],
            'lr': learning_rate_dictionary[parameter],
        })

    for parameter in optimization_settings['parameters']:
        if not parameter in parameter_dictionary:
            error("Cannot optimize over %s, as it is not part of the chosen parametrization." % parameter)
        parameters = parameter_dictionary[parameter]
        if not isinstance(parameters, list):
            parameters = [parameters]
        [error("cannot optimize over '%s[%d]': not a leaf variable." % (parameter, idx)) for idx, x in enumerate(parameters) if not x.is_leaf]

    optimizer = torch.optim.Adam(parameter_groups, betas=[0.9, 0.9], eps=1e-3)
    iterations = optimization_settings['iterations']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[iterations/2, iterations/4*3],
        gamma=0.1
    )

    losses = []
    for loss_name, weight in optimization_settings['losses'].items():
        losses.append([
            loss_name,
            LossFunctionFactory(loss_name)(),
            weight
        ])

    shadow_cache = {}
    occlusion_cache = {}
    loss_evolutions = {'Total': []}
    loss_evolutions.update(dict([(losses[loss_idx][0], []) for loss_idx in range(len(losses))]))
    parameter_evolutions = defaultdict(lambda: [])

    training_indices_batches, training_light_infos_batches = data_adapter.get_training_info()
    total_training_views = sum([len(training_indices) for training_indices in training_indices_batches])
    ctr_index = data_adapter.get_center_index()

    optimization_loop = tqdm(range(iterations))

    for iteration in optimization_loop:
        optimizer.zero_grad()

        iteration_losses = defaultdict(lambda: 0)
        for training_indices, training_light_infos in zip(training_indices_batches, training_light_infos_batches):
            simulations = experiment_state.simulate(
                training_indices,
                training_light_infos,
                shadow_cache=shadow_cache
            )

            observations = experiment_state.extract_observations(
                data_adapter,
                training_indices,
                occlusion_cache=occlusion_cache
            )
            
            total_loss = 0.0
            for loss_index in range(len(losses)):
                loss_name, loss_fcn, loss_weight = losses[loss_index]
                this_loss = loss_fcn.evaluate(
                    simulations, observations, experiment_state, data_adapter
                ).sum() * loss_weight * len(training_indices) / total_training_views
                total_loss += this_loss
                iteration_losses[loss_name] += this_loss.item()
            total_loss.backward()
            iteration_losses["Total"] += total_loss.item()
            experiment_state.clear_parametrization_caches()
            del simulations, observations, total_loss
        for loss in iteration_losses:
            loss_evolutions[loss].append(iteration_losses[loss])

        if ctr_index is not None:
            for parameter in parameter_dictionary['observation_poses']:
                if parameter.grad is not None:
                    parameter.grad[ctr_index].zero_()

        optimizer.step()
        experiment_state.enforce_parameter_bounds()

        if "photoconsistency L1" in loss_evolutions:
            desc_prefix = "Photometric L1 loss: %8.4f        " % iteration_losses["photoconsistency L1"]
        else:
            desc_prefix = ""
        optimization_loop.set_description(desc_prefix + "Total loss: %8.4f" % iteration_losses["Total"])

        with torch.no_grad():
            for parameter in optimization_settings['parameters']:
                if visualizer_dictionary[parameter] is None:
                    continue
                visualized = visualizer_dictionary[parameter](
                    parameter_dictionary[parameter]
                )
                if isinstance(visualized, dict):
                    for x in visualized:
                        parameter_evolutions[x].append(
                            visualized[x].view(1,-1)
                        )
                else:
                    parameter_evolutions[parameter].append(
                        visualized.view(1,-1)
                    )

            if (iteration + 1) % general_settings.evolution_plot_frequency == 0 and output_path_structure is not None:
                plt.figure("Losses")
                plt.clf()
                loss_names = []
                loss_values = []
                for loss_name in loss_evolutions:
                    loss_values.append(loss_evolutions[loss_name])
                    loss_names.append(loss_name)
                xvalues = np.arange(
                    1, len(loss_values[0])+1
                ).reshape(len(loss_values[0]),1)
                plt.semilogy(
                    xvalues.repeat(len(loss_names),1),
                    np.array(loss_values).T
                )
                plt.legend(loss_names)
                plt.savefig(output_path_structure % "loss")

                for parameter in parameter_evolutions:
                    plt.figure(parameter)
                    plt.clf()
                    plt.plot(
                        xvalues.repeat(parameter_evolutions[parameter][0].shape[1],1),
                        torch.cat(parameter_evolutions[parameter], dim=0).cpu().numpy(),
                    )
                    plt.savefig(output_path_structure % parameter)

        scheduler.step()
    plt.close("Losses")
    for parameter in parameter_evolutions:
        plt.close(parameter)
