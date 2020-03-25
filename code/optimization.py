import os
import torch
from tqdm import tqdm
from losses import LossFunctionFactory
import general_settings

from matplotlib import pyplot as plt
import numpy as np

def optimize(experiment_state, data_adapter, optimization_settings, visualization_output_path=None):
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
    ctr_index = None
    loss_evolutions = {'Total': []}
    loss_evolutions.update(dict([(losses[loss_idx][0], []) for loss_idx in range(len(losses))]))
    parameter_evolutions = dict([
        (parameter, [])
        for parameter in visualizer_dictionary
        if visualizer_dictionary[parameter] is not None
    ])

    training_indices, training_light_infos = data_adapter.get_training_info()

    optimization_loop = tqdm(range(iterations))
    for iteration in optimization_loop:
        optimizer.zero_grad()

        simulations = []
        observations = []

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
            ).sum() * loss_weight
            total_loss += this_loss
            loss_evolutions[loss_name].append(this_loss.item())
        loss_evolutions["Total"].append(total_loss.item())
        total_loss.backward()

        if ctr_index is not None:
            for parameter in parameter_dictionary['observation_poses']:
                if parameter.grad is not None:
                    parameter.grad[ctr_index].zero_()

        optimizer.step()
        experiment_state.enforce_parameter_bounds()
        experiment_state.clear_parametrization_caches()

        if "photoconsistency L1" in loss_evolutions:
            desc_prefix = "Photometric loss: %8.4f        " % loss_evolutions["photoconsistency L1"][-1]
        else:
            desc_prefix = ""
        optimization_loop.set_description(desc_prefix + "Total loss: %8.4f" % total_loss)

        for parameter in parameter_evolutions:
            parameter_evolutions[parameter].append(
                visualizer_dictionary[parameter](
                    parameter_dictionary[parameter]
                )
            )

        if (iteration + 1) % general_settings.evolution_plot_frequency == 0 and visualization_output_path is not None:
            plt.figure("Losses")
            plt.clf()
            loss_names = []
            loss_values = []
            for loss_name in loss_evolutions:
                loss_values.append(loss_evolutions[loss_name])
                loss_names.append(loss_name)
            plt.semilogy(
                np.arange(
                    1, len(loss_values[0])+1
                ).reshape(len(loss_values[0]),1).repeat(len(loss_names),1),
                np.array(loss_values).T
            )
            plt.legend(loss_names)
            plt.savefig(os.path.join(visualization_output_path, "loss_evolution.png"))

            # plot the relevant evolutions:
            #   --> the material evolutions, the average depth change, the average normal change?

        scheduler.step()
