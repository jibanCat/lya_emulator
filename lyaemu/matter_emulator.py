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
from .gpemulator_emukit import MatterGP

class HDF5Emulator(IModel):
    '''
    An emulator build on top of the generated HDF5 file from
    SimulationRunner.multi_sims.MultiPowerSpecs.create_hdf5()

    This class should include the basic transformation from the HDF5 file
    from np.ndarray to emukit

    this class should works for :
    - placeholder for model related implementations
    - convert parameter array in the file to ParameterSpace
    - load X <- params with correct dimension
    - load Y <- powerspecs with correct dimension

    Future:
    - Experimental loop implementation
    - BayesOpt implementation
    - MultiOutput GP
    '''
    def __init__(self, multips : Type[h5py.File]) -> None:
        self.multips = multips

        # set the parameter space
        self.set_parameters()

    def set_parameters(self) -> None:
        '''
        Set the parameter space of this experiment using the information
        given in the hdf5 file.
        
        :attr parameter_space: emukit.core.ParameterSpace
        :attr params: (n_points, n_dim),
            n_points: number of experiments we built
            n_dim: number of input parameters we sample for each experiment
        '''
        # query the full list of parameter names
        param_names  = self.multips['parameter_names'][()]
        param_bounds = self.multips['bounds'][()]

        nparams = param_names.shape[0]
        assert nparams == param_bounds.shape[0]
        assert param_bounds.shape[1] == 2

        # build the ParameterSpace instance
        param_list = []
        for pname,bound in zip(param_names, param_bounds):
            param_list.append( 
                ContinuousParameter(pname, *bound) )

        self.parameter_space = ParameterSpace(param_list)
        assert np.all(self.parameter_space.parameter_names == param_names)
        assert np.array(self.parameter_space.get_bounds()
            ).shape == param_bounds.shape

        # setup the X input parameters
        # We don't setup Y here since we don't know what's the shape of Y,
        # but the shape of X can be found by multips['parameter_names']
        X_list = []
        for pname in param_names:
            # it is always faster to assign a variable first than directly
            # operating on HDF5's IO
            p_samples = self.multips[pname][()]
            X_list.append(p_samples)
        
        self._X = np.array(X_list).T # (n_points, n_dim)
        assert self._X.shape[1] == nparams

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
        return self._X
    
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
    
    
    def set_powerspecs(self):
        '''
        Set the multi-dimensional X and Y for input and output,
        where X = [X1, X2, ..., Xn], Xi is an input for each GP,
        Y = [Y1, Y2, ..., Yn], Yi is the corresponding target for the GP.

        :attr X: (N_GPs, n_points, n_dim),
            n_GPs: means how many GPs we want, either all of them to be
                independent or using MultiOutput GP and building cov for GPs.
            n_points: number of experiments we built
            n_dim: number of input parameters we sample for each experiment
        :attr Y: (N_GPs, n_points, k_modes)
        :attr scale_factors:
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



def modecount_rebin(kk, pk, modes, minmodes=20, ndesired=200):
    """Rebins a power spectrum so that there are sufficient modes in each bin"""
    assert np.all(kk) > 0
    logkk=np.log10(kk)
    mdlogk = (np.max(logkk) - np.min(logkk))/ndesired
    istart=iend=1
    count=0
    k_list=[kk[0]]
    pk_list=[pk[0]]
    targetlogk=mdlogk+logkk[istart]
    while iend < np.size(logkk)-1:
        count+=modes[iend]
        iend+=1
        if count >= minmodes and logkk[iend-1] >= targetlogk:
            pk1 = np.sum(modes[istart:iend]*pk[istart:iend])/count
            kk1 = np.sum(modes[istart:iend]*kk[istart:iend])/count
            k_list.append(kk1)
            pk_list.append(pk1)
            istart=iend
            targetlogk=mdlogk+logkk[istart]
            count=0
    k_list = np.array(k_list)
    pk_list = np.array(pk_list)
    return (k_list, pk_list)

def get_power(matpow, rebin=True):
    """Plot the power spectrum from CAMB
    (or anything else where no changes are needed)"""
    data = np.loadtxt(matpow)
    kk = data[:,0]
    ii = np.where(kk > 0.)
    #Rebin power so that there are enough modes in each bin
    kk = kk[ii]
    pk = data[:,1][ii]
    if rebin:
        modes = data[:,2][ii]
        return modecount_rebin(kk, pk, modes)
    return (kk,pk)