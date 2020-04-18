'''
Generate cosmological samples from ParameterSpace using emukit

:class: MatterDesign
'''
from typing import Tuple
import json
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from .latin_design import LatinDesign

class MatterDesign(LatinDesign):
    '''
    Initial design for matter power spec interpolator, using Latin HyperCube.

    MP-Gadget Params:
    ----
    omegab     : baryon density. Note that if we do not have gas particles,
        still set omegab, but set separate_gas = False
    omega0     : Total matter density at z=0 (includes massive neutrinos and 
        baryons)
    hubble     : Hubble parameter, h, which is H0 / (100 km/s/Mpc)
    scalar_amp : A_s at k = 0.05, comparable to the Planck value.
    ns         : Scalar spectral index

    Methods:
    ----
    :method get_samples(point_count): get samples from ParameterSpace
    :method save_json(point_count): dump samples to a json file
    '''
    def __init__(self, omegab_bounds: Tuple, omega0_bounds: Tuple, 
            hubble_bounds: Tuple, scalar_amp_bounds: Tuple, 
            ns_bounds: Tuple) -> None:
        # initialise Parameter instances
        omega0     = ContinuousParameter('omega0',     *omega0_bounds)
        omegab     = ContinuousParameter('omegab',     *omegab_bounds)
        hubble     = ContinuousParameter('hubble',     *hubble_bounds)
        scalar_amp = ContinuousParameter('scalar_amp', *scalar_amp_bounds)
        ns         = ContinuousParameter('ns',         *ns_bounds)

        parameter_space = ParameterSpace([
            omega0, omegab, hubble, scalar_amp, ns])

        super(MatterDesign, self).__init__(parameter_space)
        
    def save_json(self, point_count: int, 
            out_filename: str = "matter_power.json") -> None:
        '''
        Save Latin HyperCube of Cosmological ParameterSpace into a json file.
        '''
        dict_latin = {}

        # get a list of param names and then init the dict to save
        param_names = self.parameter_space.parameter_names

        samples = self.get_samples(point_count)

        for i,name in enumerate(param_names):
            # the samples are in order
            dict_latin[name] = samples[:, i].tolist()

        # saving some hyper-parameters
        dict_latin['bounds']          = self.parameter_space.get_bounds()
        dict_latin['parameter_names'] = param_names

        with open(out_filename, 'w') as f:
            json.dump(dict_latin, f, indent=2)
