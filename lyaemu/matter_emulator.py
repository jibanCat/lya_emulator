'''
An Emulator mapping from MP-Gadget param space to P(k, z) bins
'''
from typing import List, Type, Tuple, Generator

from functools import reduce
from collections import Counter

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
    def __init__(self, multips : Type[h5py.File]):        
        super().__init__(multips)

        # set the target Y powerspecs
        self.set_powerspecs()


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
        # loop over simulations
        kf_list            = []
        powerspecs_list    = []
        scale_factors_list = []
        mode_list          = [] # this is for rebinning the powerspecs

        for sim in self._gen_subgroups():
            # query powerspecs:
            # it is in dimension of (n redshifts, k_modes, 4)
            # 0th of 4 is k
            # 1st of 4 is pk
            powerspecs = sim['powerspecs'][()]

            # kf should be the same for each row
            kf_list.append(powerspecs[-1, :, 0]) # choose to store z = 0
            assert np.all( powerspecs[-1, :, 0] == powerspecs[0, :, 0] )

            # power spectra should store across all redshifts
            powerspecs_list.append(powerspecs[:, :, 1]) # (n redshifts, k_modes)
            mode_list.append(powerspecs[:, :, 2])      # (n redshifts, k_modes)

            scale_factors_list.append(sim['scale_factors'][()])

        # all kf should also be the same
        assert np.sum( np.abs( kf_list[0] - kf_list[-1] ) / kf_list[0] ) < 1e-4
        self.kf = kf_list[-1] # choose to store z = 0

        # all scale factors are not the same
        # so what we can to is to select the common redshifts among them
        # but I prefer separate that in a different function from clearness
        self.scale_factors_list = scale_factors_list # List[ sim_i/scale_factors ]

        self.powerspecs_list = powerspecs_list
        self.mode_list       = mode_list
        
        # filter out non-shared scale factors
        # this modifies self.scale_factors_list and self.powerspecs_list
        n_points = self._X.shape[0] # total number of simulations we ran
        scale_factors, powerspecs, modes = self.filter_scale_factors(
            num_simulations=n_points)

        # update scale factors and powerspecs after the filtering
        self.scale_factors = scale_factors

        # this is almost the Y values, just need to re-bin
        # (n_points, n_redshifts, k_modes) -> (n_redshifts, n_points, k_modes)
        self.powerspecs = powerspecs

        assert self.scale_factors[-1].shape[0] == self.scale_factors[0].shape[0]
        assert self.scale_factors[-1][-1] == self.scale_factors[0][-1]

        # rebin the power specs
        # modify self.modes, self.powerspecs and will change the size
        # of these two matrices. Note that we assume the size of matrices
        # will be the same across different simulations. So make sure
        # write some tests on that.
        kf, powerspecs = self.rebin_matter_power(
            self.kf, self.powerspecs, modes)

        self.kf         = kf
        self.powerspecs = powerspecs

        # also set to Y target value
        self._Y = powerspecs

    def filter_scale_factors(self, num_simulations:int
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Filter out the scale factors which do not appear in other

        :attr scale_factors_list:
        :attr powerspecs_list: modified accroding to the filtering of scale
            factors.
        
        :param num_simulations: number of simulations you run in this file
        :return: (scale_factors, powerspecs)
            scale_factors: (n_points, n_redshifts)
            powerspecs:    (n_points, n_redshifts, k_modes)
        '''
        # flatten the list and find the most common scale factors
        flatten_scale_factors = reduce(
            lambda x,y : np.concatenate([x, y]), self.scale_factors_list)

        assert len(flatten_scale_factors) == sum(
            map(lambda x: x.shape[0], self.scale_factors_list))

        # thresholding all the scale factors that not happenning in every sims
        # first need to get the counts of each scale factors
        count: Counter = Counter(flatten_scale_factors)

        shared_scale_factors = np.array(
            [key for key,val in count.items() if val >= num_simulations])
        assert count.most_common(1)[0][1] == num_simulations

        # update the powerspecs and scale factors based on whether they are
        # in the shared_scale_factors
        assert len(self.powerspecs_list) == num_simulations
        assert len(self.mode_list) == num_simulations

        scale_factors = np.empty((num_simulations, len(shared_scale_factors)))
        powerspecs    = np.empty((num_simulations, len(shared_scale_factors),
            len(self.kf) )) # (n_points, n_redshifts, k_modes)
        modes         = np.empty((num_simulations, len(shared_scale_factors),
            len(self.kf) )) # (n_points, n_redshifts, k_modes)

        for i in range(num_simulations):
            # only select the scale factors in the shared_scale_factors
            this_scale_factors = self.scale_factors_list[i]
            this_powerspecs    = self.powerspecs_list[i]
            this_modes         = self.mode_list[i]

            ind = np.isin(this_scale_factors, shared_scale_factors)

            scale_factors[i, :] = this_scale_factors[ind]
            powerspecs[i, :, :] = this_powerspecs[ind, :]
            modes[i, :, :]      = this_modes[ind, :]

        assert np.sum( np.abs( scale_factors - shared_scale_factors ) 
            / scale_factors ) < 1e04
        print(str(
            "[Info] Aftering filtering out non-shared scale factors, "
            "only {} powerspecs left per simulation".format(
            len(shared_scale_factors))
            ))

        return scale_factors, powerspecs, modes

    def _gen_subgroups(self) -> Generator:
        '''
        A generator to yield HDF5 subgroups
        '''
        n_points = self._X.shape[0]
        assert self._X.shape[1] == self.parameter_space.dimensionality

        for i in range(n_points):
            name = "simulation_{}".format(i)

            # query the ith subgroup
            yield self.multips[name]

    @staticmethod
    def rebin_matter_power(kf: np.ndarray, powerspecs : np.ndarray,
            modes : np.ndarray) -> Tuple[ np.ndarray, np.ndarray ]:
        '''
        Re-bin powerspecs. This is done by using MP-Gadget's tools
        `modecount_rebin`. Note that we apply the rebinning for all
        of the simulations and all of the redshifts.

        :param kf: (k_modes, )
        :param powerspecs: (n_points, n_redshifts, k_modes)
        :param modes: (n_points, n_redshifts, k_modes)

        :return: (kf, powerspecs)
        '''
        # re-bin powerspectra per simulation
        num_simulations = powerspecs.shape[0]

        powerspecs_list = []
        kf_list         = [] # keep each kf and test the lengths at the end

        for i in range(num_simulations):
            this_powerspecs = powerspecs[i, :, :]
            this_modes      = modes[i, :, :]

            # re-bin P(z,k)
            this_kf, this_powerspecs = modecount_rebin_multi_pk(
                kf, this_powerspecs, this_modes)
            
            powerspecs_list.append(this_powerspecs)
            kf_list.append(this_kf)
        
        assert np.abs( (np.array(kf_list) - kf_list[-1]) / kf_list[0] ).sum() < 1e-4
        assert powerspecs_list[0].shape == powerspecs_list[-1].shape

        return np.array(kf_list[0]), np.array(powerspecs_list)

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
        return self._Y


class MultiFidelityEmulator(IModel, IDifferentiable):
    def __init__(self, mutlips_list : List[Type[h5py.File]] ):
        raise NotImplementedError

def modecount_rebin_multi_pk(kk:np.ndarray, pk:np.ndarray, modes:np.ndarray,
        minmodes:int=20, ndesired:int=200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebins a power spectrum so that there are sufficient modes in each bin.
    Original version was in MP-Gadget/tools.
    Slightly modified for working in 2-dimensional P(z,k)
    
    :param kk: (k_modes,)
    :param pk: (n_redshifts, k_modes)
    :param modes: (n_redshifts, k_modes)

    :return: (kf, powerspecs)
    """
    assert np.all(kk) > 0
    logkk  = np.log10(kk)

    mdlogk = (np.max(logkk) - np.min(logkk)) / ndesired

    istart = iend = 1
    
    count = 0

    # assume all modes are the same across redshifts
    assert np.sum(np.abs( (modes - modes[0, :]) / modes)) < 1e-4
    modes = modes[0, :]
    assert modes.shape[0] == pk.shape[1]

    # we assume kk the same for all redshifts but pk is a 2-dimensional
    # array. So pk here should save all of the 0th elements across all zs
    k_list  = [kk[0]]
    pk_list = [pk[:, 0]] # (1, n_redshifts), have to transpose later
    
    targetlogk = mdlogk + logkk[istart]

    while iend < np.size(logkk) - 1:
        count += modes[iend]
        iend  += 1

        if count >= minmodes and logkk[iend-1] >= targetlogk:
            pk1 = np.sum(modes[istart:iend] * pk[:, istart:iend], axis=1)/count
            kk1 = np.sum(modes[istart:iend] * kk[istart:iend])/count
            k_list.append(kk1)
            pk_list.append(pk1)

            # start counting next bin
            istart = iend
            targetlogk = mdlogk + logkk[istart]
            count = 0

    k_list  = np.array(k_list)
    pk_list = np.array(pk_list).T
    return (k_list, pk_list)

def modecount_rebin(kk:np.ndarray, pk:np.ndarray, modes:np.ndarray,
        minmodes:int=20, ndesired:int=200) -> Tuple[np.ndarray, np.ndarray]:
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

def get_power(data:np.ndarray, rebin:bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """Plot the power spectrum from CAMB
    (or anything else where no changes are needed)"""
    kk = data[:,0]
    ii = np.where(kk > 0.)
    #Rebin power so that there are enough modes in each bin
    kk = kk[ii]
    pk = data[:,1][ii]
    if rebin:
        modes = data[:,2][ii]
        return modecount_rebin(kk, pk, modes)
    return (kk,pk)
