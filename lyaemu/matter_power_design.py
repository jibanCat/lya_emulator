'''
Generate cosmological samples from ParameterSpace using emukit

:class: MatterDesign
'''
from typing import Tuple
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
        omega0     = ContinuousParameter('omega0', *omega0_bounds)
        omegab     = ContinuousParameter('omegab', *omegab_bounds)
        hubble     = ContinuousParameter('h',      *hubble_bounds)
        scalar_amp = ContinuousParameter('As',     *scalar_amp_bounds)
        ns         = ContinuousParameter('ns',     *ns_bounds)

        parameter_space = ParameterSpace([
            omega0, omegab, hubble, scalar_amp, ns])

        super(MatterDesign, self).__init__(parameter_space)
        
    def save_json(self, point_count: int, 
            out_filename: str = "matter_power.json") -> None:
        '''
        Save Latin HyperCube of Cosmological ParameterSpace into a json file.
        '''
        raise NotImplementedError


        