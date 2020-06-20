'''
A short script to see if multi-fidelity can work on
our params -> pk(z,k) setting.
'''
from typing import List, Tuple

import numpy as np
import h5py, pickle
from matplotlib import pyplot as plt

import GPy

# import emukit Multi-Fidelity functions
import emukit.multi_fidelity

# Multi-Fidelity is basically a Multi-Output GP, so we need to use a
# Multi-Output wrapper wrap on top of the GPs
# Note: Multi-Output GP: An additional Covariance on top of different GPs
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import (
    make_non_linear_kernels,
    NonLinearMultiFidelityModel)

# these are the utility functions for labelling the inputs of multi-fidelity data
from emukit.multi_fidelity.convert_lists_to_array import (convert_x_list_to_array,
                                                          convert_xy_lists_to_arrays)

# Some custom emulator functions:
# MatterEmulator: handling the inputs, like rebinning and organising the scale
#  factors into a class. Designed for each fidelity.
# MatterMultiFidelityLinearGP: Linear Multi-Fidelity without interpolating on
#  the k modes. mapping from cosmological parameters -> P(k, z).
# PkMultiFidelityLinearGP: Linear Multi-Fidelity with interpolating on the k
#  modes. mapping from (cosmological params, k) -> P(z).
from lyaemu.matter_emulator import MatterEmulator
from lyaemu.gpemulator_emukit import MatterMultiFidelityLinearGP, PkMultiFidelityLinearGP


# fidelity generation function
def gen_two_fidelities(
        fname_low: str = "../SimulationRunnerDM/data/lowRes/processed/test_dmonly.hdf5", 
        fname_high: str = "../SimulationRunnerDM/data/highRes/processed/test_dmonly.hdf5"
        ) -> Tuple[MatterEmulator, MatterEmulator, int]:
    '''
    Generate the MatterEmulator for two fidelities. Primarily for holding the
    data, so we can latter on organise the inputs for the multi-fidelities.
    '''
    n_fidelities = 2

    # setting different emulators
    emu_low  = MatterEmulator(h5py.File(fname_low,  'r'))
    emu_high = MatterEmulator(h5py.File(fname_high, 'r'))

    return (emu_low, emu_high, n_fidelities)

def test_two_fidelities_MFEmulator_w_HFEmualtor(
        fname_low: str = "../SimulationRunnerDM/data/lowRes/processed/test_dmonly.hdf5", 
        fname_high: str = "../SimulationRunnerDM/data/highRes/processed/test_dmonly.hdf5"
        ) -> None:
    '''
    Test the Multi-Fidelity Emulators with a GP trained on high-Fidelity only.
    '''
    # get the MatterEmulator instances for two fidelities
    (emu_low, emu_high, n_fidelities) = gen_two_fidelities(fname_low, fname_high)

    # Make a PkEmulator Model
    # adding k into param space but
    # keep the plimits to be highRes's k limits

    # get the inputs for the MF model
    kf_list, powers_list, params_list, param_limits_list = prepare_mf_inputs(
        emu_high, emu_low)

    gp = PkMultiFidelityLinearGP(
        params_list, kf_list, param_limits_list, powers_list,
        kernel_list=None, n_fidelities=2)

    # Cheating by using pickle
    try:
        with open('model.p', 'rb') as f:
            pk_mf_model = pickle.load(f)
    except FileNotFoundError as e:
        print(e)
        pk_mf_model = GPyMultiOutputWrapper(gp, 2, n_optimization_restarts=5)
        pk_mf_model.optimize()

    # testing samples
    # uniformly sample from parameter space
    n_samples = 100

    # remember to normalise
    # uniformly sample
    X_test = emu_high.parameter_space.sample_uniform(n_samples)
    X_test = PkMultiFidelityLinearGP._map_params_to_unit_cube(
        X_test, np.array(emu_high.parameter_space.get_bounds()))
    X_test = convert_x_list_to_array([X_test, X_test])
    # test on LF input
    X_test_lf = emu_low.X
    X_test_lf = PkMultiFidelityLinearGP._map_params_to_unit_cube(
        X_test_lf, np.array(emu_low.parameter_space.get_bounds()))
    X_test_lf = convert_x_list_to_array([X_test_lf, X_test_lf])

    # training a HF model
    high_gp_model = make_high_fidelity_gp(params_list, powers_list,
        param_limits_list, n_fidelities=2)

    # plotting loop: plot first 10
    i = 0
    for i in range(0, 10):
        # param_cube    = X_test[i, :-1][None, :]
        param_cube_lf = X_test_lf[i, :-1][None, :]

        # makes prediction and interpolation on P(k) based on a single param_cube
        x_plot = np.linspace(-2, 2, 50)
        mean_pk_mf_lf, std_pk_mf_lf, mean_pk_mf_hf, std_pk_mf_hf = make_mf_predictions(
            pk_mf_model, param_cube_lf, x_plot)

        # make the residual
        _, _, mean_pk_mf_hf_on_low_kf, _ = make_mf_predictions(
            pk_mf_model, param_cube_lf, emu_low.kf)
        residual = np.log(np.abs( np.exp(mean_pk_mf_hf_on_low_kf[:, 0]) - 
            np.exp(emu_low.Y[i, -1, :]) ))

        # make predictions on HF-only model
        x_sample_plot = make_x_plot(param_cube_lf, x_plot)
        mean_pk_hf, var_pk_hf = high_gp_model.predict(x_sample_plot)
        std_pk_hf = np.sqrt(var_pk_hf)

        # plotting functions here:
        plot_pk(x_plot, mean_pk_mf_hf[:, 0], std_pk_mf_hf[:, 0],
            label='MF: HF output', color="C0")
        plot_pk(x_plot, mean_pk_hf[:, 0], std_pk_hf[:, 0],
            label='HF-only', color="C1")
        plt.plot(emu_low.kf, emu_low.Y[i, -1, :], color='grey', label="True LF")
        plt.plot(emu_low.kf, residual, label="log( P(k)_pred - P(k)_LF )",
            ls='', marker='.', color='grey')
        plt.legend()
        # make a fancy title
        title = ",".join([
            "({}:{:.2g})".format(name,val)
            for name,val in zip(emu_low.parameter_space.parameter_names,
                emu_low.X[i, :])])
        plt.title(title)
        plt.savefig("Pk_MF_sample_LF_{}".format(i))
        plt.show()

def make_high_fidelity_gp(params_list : List[np.ndarray],
        powers_list: List[np.ndarray],
        param_limits_list: List,
        n_fidelities : int = 2) -> GPy.models.GPRegression:
    '''
    Generate a GP based on provided params_list
    '''
    #Map the parameters onto a unit cube so that all the variations are
    # similar in magnitude.
    normed_param_list = []
    for i in range(n_fidelities):
        nparams = np.shape(params_list[i])[1]
        params_cube = PkMultiFidelityLinearGP._map_params_to_unit_cube(
            params_list[i][:, :-1], param_limits_list[i])
        
        params_cube = np.concatenate(
            (params_cube, params_list[i][:, -1:]), axis=1)

        normed_param_list.append(params_cube)

    # fit High-fidelity only
    X_hf = normed_param_list[1]
    Y_hf = powers_list[1]

    # modelling setup
    # make sure turn on ARD
    nparams = np.shape(params_list[-1])[1]
    kernel  = GPy.kern.Linear(nparams, ARD=True)
    kernel += GPy.kern.RBF(nparams, ARD=True)

    high_gp_model = GPy.models.GPRegression(X_hf, Y_hf, kernel)
    high_gp_model.optimize_restarts(2)

    return high_gp_model

def make_x_plot(params_cube: np.ndarray, x_plot:np.ndarray,
        ) -> np.ndarray:
    '''
    :param params_cube: (1, n_dim) without k_mode dimension
    :return: x_sample_plot (k_points, n_dim + 1),
        x_plot (k_points, )
    '''
    x_sample_plot = np.repeat(params_cube, x_plot.shape[0], axis=0)

    x_sample_plot = np.concatenate(
        (x_sample_plot, x_plot[:, None]), axis=1)

    return x_sample_plot

# [TODO] this could be saved as a utility function
def prepare_mf_inputs(emu_high: MatterEmulator, emu_low: MatterEmulator
        )-> Tuple[List]:
    '''
    Prepare the inputs lists for Multi-Fidelity model

    :return: (kf_list, powers_list, params_list, param_limits_list)
        kf_list[i]: (k_points)
        powers_list[i]: (k_points * n_points, 1)
        param_list[i]: (k_points * n_points, n_dim + 1)
        param_limits_list[i]: (n_dim, 2)
    '''
    # k argumentation
    # kf     : (k_modes, )       -> (k_modes  * n_points, 1)
    # params : (n_points, n_dim) -> (n_points * k_modes, n_dim)
    # concat up
    # new params : (n_points, n_dim) -> (n_points * k_modes, n_dim + 1)
    n_points_hf = emu_high.X.shape[0]
    n_points_lf = emu_low.X.shape[0]
    k_modes_hf  = emu_high.kf.shape[0] 
    k_modes_lf  = emu_low.kf.shape[0]

    # repeat params
    # (n_points * k_modes, n_dim)
    arg_params_hf = np.repeat(emu_high.X, k_modes_hf, axis=0)
    arg_params_lf = np.repeat(emu_low.X, k_modes_lf, axis=0)

    # kron the kf and add second dimension
    # (k_modes  * n_points, 1)
    kf_hf = np.kron( np.ones((n_points_hf, 1)), emu_high.kf[:, None] )
    kf_lf = np.kron( np.ones((n_points_lf, 1)), emu_low.kf[:, None] )

    # (k_modes  * n_points, 1)
    # only select z=0 for testing
    pk_hf = emu_high.Y[:, -1, :].ravel()[:, None]
    pk_lf = emu_low.Y[:, -1, :].ravel()[:, None]

    arg_params_hf = np.concatenate((arg_params_hf, kf_hf), axis=1)
    arg_params_lf = np.concatenate((arg_params_lf, kf_lf), axis=1)

    # prepare input for the Pk Emulator
    kf_list = [emu_low.kf, emu_high.kf]
    powers_list = [
        pk_lf, pk_hf] # only the last redshift
    params_list = [arg_params_lf, arg_params_hf]
    param_limits_list = [
        np.array(emu_low.parameter_space.get_bounds()),
        np.array(emu_high.parameter_space.get_bounds())]

    return kf_list, powers_list, params_list, param_limits_list

# MF prediction function
def make_mf_predictions(pk_mf_model: PkMultiFidelityLinearGP,
        param_cube: np.ndarray, x_plot:np.ndarray) -> Tuple[np.ndarray]:
    '''
    :param pk_mf_model: the Multi-Fidelity model you want to make
        predictions on.
    :param params_cube: (1, n_dim) without k_mode dimension.
    :param X_sample_plot: (k_points, n_dim + 1 + 1), 1 for dim for k,
        and 1 for dim for MF indicator.
    :return: (mean_pk_mf_lf, std_pk_mf_lf, mean_pk_mf_hf, std_pk_mf_hf)
        mean P(k) for LF output and its STD,
        mean P(k) for HF output and its STD.
    '''
    # Multi-Fidelity GP predictions
    x_sample_plot = make_x_plot(param_cube, x_plot)
    # plot highRes output
    X_hf_sample_plot = np.concatenate(
        (x_sample_plot, np.ones((len(x_plot), 1))), axis=1)
    # plot lowRes output
    X_lf_sample_plot = np.concatenate(
        (x_sample_plot, np.zeros((len(x_plot), 1))), axis=1)

    # MF: LF output
    mean_pk_mf_lf, var_pk_mf_lf = pk_mf_model.predict(X_lf_sample_plot)
    std_pk_mf_lf = np.sqrt(var_pk_mf_lf)
    # MF: HF output
    mean_pk_mf_hf, var_pk_mf_hf = pk_mf_model.predict(X_hf_sample_plot)
    std_pk_mf_hf = np.sqrt(var_pk_mf_hf)
    
    return mean_pk_mf_lf, std_pk_mf_lf, mean_pk_mf_hf, std_pk_mf_hf

def plot_pk(x_plot: np.ndarray, 
        mean_pk: np.ndarray, std_pk: np.ndarray,
        label: str = "", color="C0") -> None:
    '''
    :param x_plot: (k_points, )
    :param mean_pk: (k_points, ) note the output of gp.predict
        is (k_points, 1) and you only want the last dimension.
    :param std_pk: (k_points, )
    Note: n_dim, dimension of params_cube
    '''
    plt.plot(x_plot, mean_pk, color=color)
    plt.fill_between(
        x_plot,
        mean_pk - 1.96 * std_pk,
        mean_pk + 1.96 * std_pk,
        alpha=0.3, color=color, label=label)
    plt.xlabel('log10(k)')
    plt.ylabel('log10(P(k))')
    plt.xlim(-2, 2)
    plt.ylim(-1, 5)
