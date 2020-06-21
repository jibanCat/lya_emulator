'''
Multi-Fidelity Emulator applied on matter power spectrum
'''
from typing import List, Type, Tuple

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

        kf_list, powers_list, params_list, param_limits_list = self.prepare_mf_inputs(
            self.matter_emulator_list,
            minmodes=minmodes, ndesired=ndesired, z0=z0)

        

    @staticmethod
    def prepare_mf_inputs(matter_emulator_list : List[MatterEmulator],
            minmodes: int = 20, ndesired : int = 200, z0: int = 0)-> Tuple[List]:
        '''
        Prepare the inputs lists for Multi-Fidelity model

        :return: (kf_list, powers_list, params_list, param_limits_list)
            kf_list[i]: (k_points)
            powers_list[i]: (k_points * n_points, 1)
            param_list[i]: (k_points * n_points, n_dim + 1)
            param_limits_list[i]: (n_dim, 2)
        '''
        n_fidelities = len(matter_emulator_list)

        kf_list           = []
        powers_list       = []
        params_list       = []
        param_limits_list = []

        for i in range(n_fidelities):
            this_emu = matter_emulator_list[i]

            # find the redshift to condition on
            this_redshifts = (1 / this_emu.scale_factors[0]) - 1
            idx = (np.abs(this_redshifts - z0)).argmin()
            # make sure this argmin is not too far away from the conditioned value
            assert np.abs(this_redshifts[idx] - z0) < 1e-1

            # re-set the binning of powerspecs. Too many k modes may induce
            # a large input matrix.
            print("[Info] Rebinning the power specs with {} modes and {} desired ".format(
                minmodes, ndesired), end="")
            this_emu.set_powerspecs(minmodes, ndesired)
            print("... with length of kf = {}.".format(len(this_emu.kf)))

            # k argumentation
            # kf     : (k_modes, )       -> (k_modes  * n_points, 1)
            # params : (n_points, n_dim) -> (n_points * k_modes, n_dim)
            # concat up
            # new params : (n_points, n_dim) -> (n_points * k_modes, n_dim + 1)
            this_n_points = this_emu.X.shape[0]
            this_k_modes  = this_emu.kf.shape[0]

            # repeat params
            # (n_points * k_modes, n_dim)
            this_arg_params = np.repeat(this_emu.X, this_k_modes, axis=0)

            # kron the kf and add second dimension
            # (k_modes  * n_points, 1)
            this_kf = np.kron( np.ones((this_n_points, 1)), this_emu.kf[:, None] )
            
            # (k_modes  * n_points, 1)
            # only select z=0 for testing
            this_pk = this_emu.Y[:, idx, :].ravel()[:, None]

            this_arg_params = np.concatenate((this_arg_params, this_kf), axis=1)
            
            # prepare input for the Pk Emulator
            kf_list.append(this_emu.kf)
            powers_list.append(this_pk) # P(k, z=z0)
            params_list.append(this_arg_params)
            param_limits_list.append(
                np.array(this_emu.parameter_space.get_bounds()))

        return kf_list, powers_list, params_list, param_limits_list
