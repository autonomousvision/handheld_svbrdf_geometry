from abc import ABC, abstractmethod
from utils.vectors import inner_product
from utils.logging import error
import torch
from utils.TOME import permutohedral_filter

from utils.sentinels import OBSERVATION_MASKED_OUT, OBSERVATION_OUT_OF_BOUNDS, SIMULATION_SHADOWED

class LossFunction(ABC):
    """
    These loss functions need to be instantiated to leave the option for parametrized loss functions open.
    """

    @abstractmethod
    def evaluate(self, simulations, observations, experiment_state, data_adapter):
        """
        Should return an un-reducted loss calculation, i.e. an image or sets of images (resp. a vector/set of vectors)
        This can then either be reducted or visualized.
        """
        pass


class PhotoconsistencyL1Loss(LossFunction):
    def evaluate(self, simulations, observations, experiment_state, data_adapter):
        photo_losses = (simulations - observations[0]).abs()

        out_of_bounds = observations[0][...,0] == OBSERVATION_OUT_OF_BOUNDS
        masked_out = observations[0][...,0] == OBSERVATION_MASKED_OUT
        occluded = observations[1]
        shadowed = simulations[...,0] == SIMULATION_SHADOWED
        valid_losses = (out_of_bounds + masked_out + occluded + shadowed)[:,:,None] == 0

        return photo_losses * valid_losses.float() / len(simulations)


class GeometricConsistencyLoss(LossFunction):
    def evaluate(self, simulations, observations, experiment_state, data_adapter):
        normals_from_depth = experiment_state.locations.implied_normal_vector()
        normals_estimated = experiment_state.normals.normals()
        inner_products = inner_product(normals_from_depth, normals_estimated)
        return 1-inner_products


class DepthCompatibilityLoss(LossFunction):
    def evaluate(self, simulations, observations, experiment_state, data_adapter):
        depth_threshold = 0.001
        estimated_depth = experiment_state.locations.implied_depth_vector().squeeze()
        initial_depth = experiment_state.locations.create_vector(data_adapter.center_depth)
        losses = ((initial_depth - estimated_depth).abs() - 0.001).div_(depth_threshold).clamp_(min=0).pow(2)
        return losses


class NormalSmoothnessLoss(LossFunction):
    def evaluate(self, simulations, observations, experiment_state, data_adapter):
        normal_image = experiment_state.locations.create_image(experiment_state.normals.normals())
        valid_normals = normal_image.norm(dim=-1, keepdim=True) > 0
        normal_smoothness_H = (normal_image[1:,1:] - normal_image[:-1,1:]).abs()
        normal_smoothness_V = (normal_image[1:,1:] - normal_image[1:,:-1]).abs()
        normal_smoothness_mask = (valid_normals[1:,1:] * valid_normals[:-1,1:] * valid_normals[1:,:-1])
        pixel_losses = (normal_smoothness_H + normal_smoothness_V) * normal_smoothness_mask
        return experiment_state.locations.create_vector(pixel_losses)


class MaterialWeightSparsityLoss(LossFunction):
    def evaluate(self, simulations, observations, experiment_state, data_adapter):
        if not hasattr(experiment_state.materials, "base_weights"):
            error("Calculating weight sparsity on a material representation that doesn't have weights")
        weights = experiment_state.materials.base_weights
        mean_weights = weights.mean(dim=0, keepdim=True)
        return 2 - (weights - mean_weights).abs()


class MaterialWeightSmoothnessLoss(LossFunction):
    def evaluate(self, simulations, observations, experiment_state, data_adapter):
        if not hasattr(experiment_state.materials, "base_weights"):
            error("Calculating weight smoothness on a material representation that doesn't have weights")

        N = experiment_state.locations.get_point_count()
        locations = experiment_state.locations.location_vector()             # Nx3
        albedo = experiment_state.materials.get_brdf_parameters()['diffuse'] # Nx3

        bilateral_positions = torch.cat(
            (locations / 0.1, albedo / 0.01),
            dim=1
        ) # Nx6

        weights = experiment_state.materials.base_weights

        smoothed_weights = permutohedral_filter(
            weights[None],
            bilateral_positions[None],
            torch.ones(1, N, dtype=torch.float, device=locations.device),
            False
        )[0]

        return (smoothed_weights - weights).abs()


def LossFunctionFactory(name):
    valid_dict = {
        "photoconsistency L1": PhotoconsistencyL1Loss,
        "geometric consistency": GeometricConsistencyLoss,
        "depth compatibility": DepthCompatibilityLoss,
        "normal smoothness": NormalSmoothnessLoss,
        "material sparsity": MaterialWeightSparsityLoss,
        "material smoothness": MaterialWeightSmoothnessLoss
    }
    if name in valid_dict:
        return valid_dict[name]
    else:
        error("Loss function '%s' is not supported." % name)
