'''
Multi-Fidelity Emulator applied on matter power spectrum
'''
from typing import List, Type, Tuple, Any

import numpy as np
import h5py
import GPy

# IModel is the base class to define a workable emukit Emulator
# we should code up our own gradients for multi-fidelity, so IDifferentiable
# should be imported
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.multi_fidelity.convert_lists_to_array import convert_y_list_to_array
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array

from .matter_emulator import MatterEmulator
from .gpemulator_emukit import MatterMultiFidelityLinearGP, PkMultiFidelityLinearGP

class MultiFidelityMatterEmulator(IModel, IDifferentiable):
    '''
    Multi-Fidelity Emulator, using Linear fidelity.

        p( P(k=k,z=z0) | k, cosmological params, z=z0) = GP( P(k=k); µ, K )
    
    where P(k=k,z=z0) is the value of matter power spectrum at k=k and z=z0;
    k is the input space need to interpolate, however we do not interpolate
    on redshift space due to it is highly non-linear.

    The input of this class is the list of MatterEmulator,
    the pre-processing part is handled by MatterEmulator class

    :param matter_emulator_list: a list of matter emulators, each includes
        :attr kf: the avaliable k-bin points (recommend to rebin)

    Note:
    ----
    the order of fidelities : low -> high

    Design Note:
    ----
    I was thinking whether to put param_list, kf_list and powerspec_list
    directly to the input arguments to this class or use a list of
    MatterEmulators.
    It seems more natural to put a list of MatterEmulator since we want to
    build a multi-level matter emulator and we should require a list of
    MatterEmualtors as the input.
    '''
    def __init__(self, matter_emulator_list : List[MatterEmulator],
            minmodes: int = 20, ndesired : int = 200, z0: int = 0):
        self.matter_emulator_list = matter_emulator_list
        
        self.minmodes = minmodes
        self.ndesired = ndesired
        self.z0       = z0

        # parepare the multi-fidelity inputs and outputs to interpolate.
        # they would be saved into self
        self.prepare_mf_inputs(self.matter_emulator_list,
            minmodes=minmodes, ndesired=ndesired, z0=z0)

    # being explicit about the inputs and outputs
    @property
    def X(self):
        return self.param_list

    @property
    def Y(self):
        return self.powers_list

    def set_parameters(self, param_list: List[np.ndarray], kf_list: List[np.ndarray],
            param_limits_list: List[np.ndarray], powers_list: List[np.ndarray]
            ) -> None:
        '''
        Set the default inputs and outputs for multi-fidelity emulator
        '''
        self.param_list        = param_list
        self.kf_list           = kf_list
        self.param_limits_list = param_limits_list
        self.powers_list       = powers_list

    def get_interp(self):
        '''
        A wrapper over self._get_interp, to make the function arguments easier
        read.
        '''
        n_fidelities = len(self.kf_list)
        self._get_interp(self.param_list, self.kf_list, self.param_limits_list,
            self.powers_list, n_fidelities=n_fidelities)

    def _get_interp(self, params_list: List[np.ndarray], kf_list: List[np.ndarray],
            param_limits_list: List[np.ndarray], powers_list: List[np.ndarray],
            n_fidelities: int, kernel_list=None) -> None:
        '''
        GP interpolate on the different fidelity.

            p( P(k=k,z=z0) | k, cosmological params, z=z0) = GP( P(k=k); µ, K )

            P(k=k,z=z0) ~ GP( P(k=k); µ, K )
        and
            P_t(k,z=z0) = rho_t * P_{t-1}(k,z=z0) + delta_t(k,z=z0)

        where:
            t       is the fidelity
            t-1     is the previous fidelity
            P_t     is the function of powerspec modelling with a GP at fidelity t
            delta_t models the difference between fidelity t-1 and t
            rho_t   a scaling parameter between fidelity t and t-1
        '''
        gp = PkMultiFidelityLinearGP(params_list, kf_list, param_limits_list, powers_list,
            kernel_list=kernel_list, n_fidelities=n_fidelities)

        self.gp = gp

    def prepare_mf_inputs(self, matter_emulator_list : List[MatterEmulator],
            minmodes: int = 20, ndesired : int = 200, z0: int = 0,
            number_samples_list: List[int] = [20, 3],
            randomize: bool = False)-> Tuple[List]:
        '''
        Prepare the inputs lists for Multi-Fidelity model

        :param matter_emulator_list: a list of matter emulators you want to build
            multi-fidelity GP on.
        :param number_samples_list: number of samples we want to apply interpolation
            on each fidelity.
        :param randomize: if True, randomly select samples from each fidelity.

        :return: (kf_list, powers_list, params_list, param_limits_list)
            kf_list[i]: (k_points)
            powers_list[i]: (k_points * n_points, 1)
            param_list[i]: (k_points * n_points, n_dim + 1)
            param_limits_list[i]: (n_dim, 2)
        '''
        n_fidelities = len(matter_emulator_list)

        assert len(number_samples_list) == n_fidelities
        # you must have a sample size larger than the size you want to select
        for num,emu in zip(number_samples_list, matter_emulator_list):
            assert num <= emu.X.shape[0]

        kf_list           = []
        powers_list       = []
        params_list       = []
        param_limits_list = []

        for i in range(n_fidelities):
            this_emu = matter_emulator_list[i]

            # re-set the binning of powerspecs. Too many k modes may induce
            # a large input matrix.
            print("[Info] Rebinning the power specs with {} modes and {} desired ".format(
                minmodes, ndesired), end="")
            this_emu.set_powerspecs(minmodes, ndesired)
            print("... with length of kf = {}.".format(len(this_emu.kf)))

            this_pk, this_arg_params = self._prepare_fidelity_input(
                this_emu.X, this_emu.Y, this_emu.scale_factors[0], this_emu.kf,
                z0=z0, num_samples=number_samples_list[i], randomize=randomize)

            # prepare input for the Pk Emulator
            powers_list.append(this_pk) # P(k, z=z0)
            params_list.append(this_arg_params)
            
            kf_list.append(this_emu.kf)
            param_limits_list.append(
                np.array(this_emu.parameter_space.get_bounds()))

        # set the parameter immediately after prepare the inputs. This would
        # foster the testing progress, since we can change num_samples to select
        # very easily.
        self.set_parameters(params_list, kf_list, param_limits_list, powers_list)

    @staticmethod
    def _prepare_fidelity_input(X : np.ndarray, Y: np.ndarray,
            scale_factors: np.ndarray, kf: np.ndarray, z0: int = 0,
            num_samples = int, randomize: bool = False) -> Tuple:
        '''
        Argument the input space with each k mode.

        Input space:
            cosmological params -> (cosmological params, k)

        :param X: inputs, shape (n_samples, n_points)
        :param Y: outputs, shape (n_samples, z_modes, k_modes)
        :param scale_factors: scale factors available from MP-Gadget.
            shape (z_modes,)
        :param z0: redshift bin we want to condition on. default 0.
        :param num_samples: number of samples we would like to select from
            n_samples from the data.
        :param randomize: if True, randomly select num_samples.

        :return: (this_pk, this_arg_params)
            this_pk: (k_points * num_samples, 1)
            this_arg_params: (k_points * num_samples, n_dim + 1)
        '''
        # select only partial the data
        assert num_samples > 1 # at least having two samples to interpolate
        total_num_samples = X.shape[0]
        if randomize:
            selected_ind = np.random.randint(
                low=0, high=total_num_samples, size=num_samples)
        else:
            # otherwise, just selected the first num_samples samples
            selected_ind = np.arange(0, num_samples)
        # update the input and output arrays
        X = X[selected_ind, :]
        Y = Y[selected_ind, :, :]

        # find the redshift to condition on
        this_redshifts = (1 / scale_factors) - 1
        idx = (np.abs(this_redshifts - z0)).argmin()
        # make sure this argmin is not too far away from the conditioned value
        assert np.abs(this_redshifts[idx] - z0) < 1e-1

        # k argumentation
        # kf     : (k_modes, )       -> (k_modes  * n_points, 1)
        # params : (n_points, n_dim) -> (n_points * k_modes, n_dim)
        # concat up
        # new params : (n_points, n_dim) -> (n_points * k_modes, n_dim + 1)
        this_n_points = X.shape[0]
        this_k_modes  = kf.shape[0]

        # repeat params
        # (n_points * k_modes, n_dim)
        this_arg_params = np.repeat(X, this_k_modes, axis=0)

        # kron the kf and add second dimension
        # (k_modes  * n_points, 1)
        this_kf = np.kron( np.ones((this_n_points, 1)), kf[:, None] )
        
        # (k_modes  * n_points, 1)
        # only select z=0 for testing
        this_pk = Y[:, idx, :].ravel()[:, None]

        this_arg_params = np.concatenate((this_arg_params, this_kf), axis=1)

        return this_pk, this_arg_params

