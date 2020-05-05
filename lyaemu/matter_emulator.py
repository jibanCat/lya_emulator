'''
An Emulator mapping from MP-Gadget param space to P(k, z) bins
'''
from typing import List, Type, Tuple

import numpy as np
import h5py
import GPy

# these two should help use define parameters in emukit's fashion
# later on it would be useful to intergrate with bayesian optimization
# and furthermore experimental designs
from emukit.core import ParameterSpace, ContinuousParameter

# IModel is the base class to define a workable emukit Emulator
# we should code up our own gradients for multi-fidelity, so IDifferentiable
# should be imported
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.multi_fidelity.convert_lists_to_array import convert_y_list_to_array
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array

# matter power specs related useful functions are from gpemulator or
# the matter_power files
from .gpemulator import MultiBinGP, SkLearnGP

class HDF5Emulator(IModel):
    '''
    An emulator build on top of the generated HDF5 file from
    SimulationRunner.multi_sims.MultiPowerSpecs.create_hdf5()

    This class should include the basic transformation from the HDF5 file
    from np.ndarray to emukit

    this class should works for :
    - placeholder for model related implementations
    - convert parameter array in the file to ParameterSpace
    - 

    Future:
    - Experimental loop implementation
    - BayesOpt implementation
    '''
    def __init__(self, multips : Type[h5py.File]) -> None:
        self.multips = multips

    @staticmethod
    def hdf52parameters(multips : Type[h5py.File]) -> Type[ParameterSpace]:
        '''
        Read out the parameter samplings and bounds in the HDF5,
        transform the samplings to the X and Y attrs
        '''
        # self._X = Latin_samplings
        # self._Y = powerspecs
        raise NotImplementedError

    def predict(self, X : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Make predictions but include the variances
        '''
        raise NotImplementedError
    
    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError

    def optimize(self, verbose: bool = False) -> None:
        raise NotImplementedError
    
    @property
    def X(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def Y(self) -> np.ndarray:
        raise NotImplementedError


class MatterEmulator(HDF5Emulator):
    '''
    An emulator to mapping from cosmological params -> P(k,z)

    To make the things easy to work on, we wrap the training data (for a single
    fidelity) in a single HDF5 file. The file is self-explanatory, generated
    by SimulationRunner.multi_sims.MultiPowerSpecs.create_hdf5()

    The structure of the file is:
    ----
    **LatinDict
    simulation_1/
        - powerspecs
        - scale_factors
        - camb_matter
        - camb_redshifts
        - **param_dict
    simulation_2/
    ...

    LatinDict:
    ----
    {
        'omega0' : [0.2686, 0.299 , 0.2934, ...],
        'hubble' : [0.6525, 0.7375, 0.6995, ...],
        ...
        'parameter_names' : ['omega0', 'hubble', ...],
        'bounds' : [[2.68e-01, 3.08e-01],
                    [6.50e-01, 7.50e-01], 
                    ... ],
    }
    '''
    def __init__(self, mutlips : Type[h5py.File]):
        raise NotImplementedError
    
    @staticmethod
    def hdf52parameters(multips : Type[h5py.File]) -> Type[ParameterSpace]:
        '''
        Read out the parameter samplings and bounds in the HDF5,
        transform the samplings to the X and Y attrs
        '''
        # self._X = Latin_samplings
        # self._Y = powerspecs
        raise NotImplementedError
    
    @staticmethod
    def rebin_matter_power(k_bins : np.ndarray) -> np.ndarray:
        raise NotImplementedError

class MultiFidelityEmulator(IModel, IDifferentiable):
    def __init__(self, mutlips_list : List[Type[h5py.File]] ):
        raise NotImplementedError



