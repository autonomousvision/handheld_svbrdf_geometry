import numpy as np
import torch
from abc import abstractmethod, abstractstaticmethod

from utils.vectors import inner_product, normalize
from parametrizations.parametrization import Parametrization

class BrdfParametrization(Parametrization):
    """
    A Parametrization subclass representing a material parametrization model.
    It should support the calculation of reflectance coefficients, based on
    all relevant geometry information. It supports the use of underlying parameters,
    in case the BRDF model is something that should be changing.
    Material parameters are always split up in a 'diffuse' and 'specular' component.
    """

    @staticmethod
    def _calculate_NdotHs(Ls, Vs, normals):
        """
        Internal function for calculation of half-vectors and their inner products
        with the surface normals.

        Inputs:
            Ls              NxLx3 torch.tensor with the directions between
                                the points and the scene lights
            Vs              Nx3 torch.tensor with the directions between
                                the points and the camera
            normals         Nx3 torch.tensor with the surface normals
        
        Outputs:
            Hs              NxLx3 torch.tensor with the normalized half-vectors between
                                viewing and light directions.
            NdotHs          NxLx1 torch.tensor containing the inner products between
                                the surface normals and the view-light half vectors
        """
        Hs = (Ls + Vs)
        Hs = normalize(Hs)
        NdotHs = inner_product(normals, Hs)
        return Hs, NdotHs

    @abstractmethod
    def calculate_rhos(self, Ls, Vs, normals, parameter_dict):
        """
        Calculate the reflectance of a set of scene points.

        Inputs:
            Ls              NxLx3 torch.tensor with the directions between
                                the points and the scene lights
            Vs              Nx3 torch.tensor with the directions between
                                the points and the camera
            normals         Nx3 torch.tensor with the surface normals
            parameter_dict  a dictionary containing:
                diffuse     NxB_d torch.tensor with the diffuse material parameters
                specular    a dictionary for torch.tensors with specular parameters
        
        Outputs:
            rhos            NxLx3 torch.tensor with, for each light and color channel,
                                the fraction of incoming light that gets reflected
                                towards the camera
            NdotHs          NxL torch.tensor containing the inner products between
                                the surface normals and the view-light half vectors
        """
        pass

    @abstractmethod
    def get_parameter_count(self):
        """
        The number of parameters necessary to parametrize a single material.

        Outputs:
            B_d             The number of diffuse parameters required
            B_s_dict        A dictionary with specular parameter names and
                                the number parameters they required
        """
        pass

    @abstractmethod
    def enforce_brdf_parameter_bounds(self, parameter_dict):
        """
        Perform Euclidean projection of the parameters onto their feasible domain.
        This is performed in-place on the underlying data of the dictionary elements.
        Note: this is distinct from this Parametrization's own enforce_parameter_bounds.

        Inputs:
            parameter_dict  a dictionary containing:
                diffuse     NxB_d torch.tensor with the diffuse material parameters
                specular    NxB_s torch.tensor with the specular material parameters
        """
        pass


class Diffuse(BrdfParametrization):
    def calculate_rhos(self, Ls, Vs, normals, parameters):
        Hs, NdotHs = BrdfParametrization._calculate_NdotHs(Ls, Vs, normals)
        rhos = parameters['diffuse'].view(-1, 1, 3) / np.pi

        return rhos, NdotHs
    
    def parameter_info(self):
        return {}

    def get_parameter_count(self):
        return 3, {}

    def enforce_brdf_parameter_bounds(self, parameters):
        parameters['diffuse'].data.clamp_(min=0.0)

    def serialize(self):
        pass

    def deserialize(self, *args):
        pass


def Fresnel(NdotLs, p_eta_mat):
    """
    Calculate the Fresnel term, given the inner product between the surface normal
    and the lighting direction.
    The environment dielectrical coefficient is assumed to be 1.

    Inputs:
        NdotLs              NxLx1 torch.tensor with the inner products
        p_eta_mat           Nx1 torch.tensor with the dielectric coefficients of the surface
    """
    cos_thetas_env = NdotLs

    # Snell's law
    # sin_thetas_env = torch.nn.functional.relu(1 - cos_thetas_env ** 2).sqrt()
    # sin_thetas_mat = sin_thetas_in / p_eta_mat
    # cos_thetas_mat = torch.nn.functional.relu(1 - sin_thetas_mat ** 2).sqrt()

    # shortcut, less numerical issues
    cos_thetas_mat = (cos_thetas_env**2 + p_eta_mat**2 - 1).sqrt() / p_eta_mat

    # Fresnel equations for both polarizations
    r_p = (
        p_eta_mat * cos_thetas_env - cos_thetas_mat
    ) / (
        p_eta_mat * cos_thetas_env + cos_thetas_mat
    )
    r_s = (
        cos_thetas_env - p_eta_mat * cos_thetas_mat
    ) / (
        cos_thetas_env + p_eta_mat * cos_thetas_mat
    )
    return (r_p ** 2 + r_s ** 2) / 2


def Beckmann(NdotHs, p_roughness):
    """
    Calculate the Beckman microfacet distribution coefficient, given the 
    inner products between the surface normals and the half vectors and the 
    surface roughness.

    Inputs:
        NdotHs          NxLx3 torch.tensor containing the inner products
        p_roughness     Nx1 torch.tensor containing the surface roughnesses
    
    Outputs:
        Ds              NxLx1 torch.tensor containing the microfacet distributions
    """
    cosNH2 = (NdotHs ** 2).clamp_(min=0., max=1.)
    cosNH4 = cosNH2 ** 2
    tanNH2 = (1 - cosNH2) / cosNH2
    p_roughness2 = p_roughness**2
    Ds = (-tanNH2 / p_roughness2).exp() / (p_roughness2 * cosNH4)
    return Ds


def GTR(NdotHs, p_roughness, gamma=1.):
    """
    Calculate the GTR microfacet distribution coefficient,given the 
    inner products between the surface normals and the half vectors and the 
    surface roughness.

    Inputs:
        NdotHs          NxLx3 torch.tensor containing the inner products
        p_roughness     Nx1 torch.tensor containing the surface roughnesses
    
    Outputs:
        Ds              NxLx1 torch.tensor containing the microfacet distributions
    """
    cosNH2 = (NdotHs ** 2).clamp_(min=0., max=1.)
    p_roughness2 = p_roughness ** 2
    if gamma == 1.:
        cs = (p_roughness2 - 1) / p_roughness2.log()
        Ds = cs / (1 + (p_roughness2 - 1) * cosNH2 + (cosNH2 == 1).float())
        Ds[cosNH2 == 1.] = (-1 / p_roughness2.log() / p_roughness2).repeat(cosNH2.shape[0],1,1)[cosNH2 == 1.]
    else:
        cs = (gamma - 1) * (p_roughness2 - 1) / (1 - p_roughness2 ** (1 - gamma))
        Ds = cs / ((1 + (p_roughness2 - 1) * cosNH2) ** gamma)
    return Ds


def SmithG1(NdotWs, p_roughness):
    """
    Calculate Smith's G1 shadowing function, given the relevant inner product
    and the inner product between the viewing vector and the view-light half vector,
    as well as the surface roughness.

    Inputs:
        NdotWs          NxLx3 torch.tensor containing inner products
        VdotHs          NxLx3 torch.tensor containing inner products
        p_roughness     Nx1 torch.tensor containing the surface roughnesses
    
    Outputs:
        Gs              NxLx1 torch.tensor containing the shadowing values
    """
    # if any of the cosines are negative, then this clamping will result in zeroes
    cos_thetas = NdotWs.clamp_(min=0.0, max=1.0)
    cos_thetas2 = cos_thetas**2
    # sin_thetas == 0 gets handled below. Trips up the sqrt() backprop otherwise
    sin_thetas = (1 - cos_thetas2 + (cos_thetas2 == 1).float()).sqrt()
    cot_thetas = cos_thetas / (sin_thetas)
    prelims = cot_thetas / p_roughness
    prelims2 = prelims**2

    Gs = (3.535 * prelims + 2.181 * prelims2) / (1 + 2.276 * prelims + 2.577 * prelims2)

    # if sin_thetas == 0 -> fix to 1.0 (no shadowing)
    Gs[sin_thetas == 0] = 1.0
    # the above function turns around the wrong way at this point
    Gs[prelims >= 1.6] = 1.0

    return Gs


class CookTorrance(BrdfParametrization):
    """
    The Cook Torrance BRDF model. It describes a material with 3 diffuse coeffients for albedo,
    3 specular coefficients for albedo, as well as a roughness and a dielectric coefficient.
    """

    def calculate_rhos(self, Ls, Vs, normals, parameters):
        Hs, NdotHs = BrdfParametrization._calculate_NdotHs(Ls, Vs, normals)
        NdotLs = inner_product(normals, Ls)
        NdotVs = inner_product(normals, Vs)
        VdotHs = inner_product(Vs, Hs)

        p_diffuse = parameters['diffuse'].view(1,-1,3)
        p_specular = parameters['specular']['albedo'].view(1,-1,3)
        # somewhat non-standard, we parametrize roughness as its square root
        # this yields better resolution around zero
        p_roughness = parameters['specular']['roughness'].view(1,-1,1) ** 2

        # fresnel term -- optional
        if 'eta' in parameters['specular']:
            p_eta = parameters['specular']['eta'].view(1,-1,1)
            Fs = Fresnel(VdotHs, p_eta)
        else:
            Fs = 1.

        # microfacet distribution
        Ds = GTR(NdotHs, p_roughness)
        # Smith's shadow-masking function
        Gs = SmithG1(NdotLs, p_roughness) * SmithG1(NdotVs, p_roughness)

        denominator = 4 * np.pi * NdotLs * NdotVs
        CTs = p_specular * (Fs * Ds * Gs) / (denominator + (denominator == 0).float())

        rhos = p_diffuse / np.pi + CTs

        return rhos, NdotHs
    
    def parameter_info(self):
        return {}

    def get_parameter_count(self):
        return 3, {'albedo': 3, 'roughness': 1, 'eta': 1}

    def enforce_brdf_parameter_bounds(self, parameters):
        parameters['diffuse'].data.clamp_(min=0.0)
        parameters['specular']['albedo'].data.clamp_(min=0.0)
        parameters['specular']['roughness'].data.clamp_(min=1e-2, max=1 - 1e-2)
        if 'eta' in parameters['specular']:
            parameters['specular']['eta'].data.clamp_(min=1.0001, max=2.999)

    def serialize(self):
        pass

    def deserialize(self, *args):
        pass


class CookTorranceF1(CookTorrance):
    """
    A simplified version of the Cook Torrance model where the Fresnel term is disregarded.
    """

    def get_parameter_count(self):
        return 3, {'albedo': 3, 'roughness': 1}


def BrdfParametrizationFactory(name):
    valid_dict = {
        "diffuse": Diffuse,
        "cook torrance": CookTorrance,
        "cook torrance F1": CookTorranceF1,
    }
    if name in valid_dict:
        return valid_dict[name]
    else:
        error("BRDF parametrization '%s' is not supported." % name)
