'''
A short script to see if multi-fidelity can work on
our params -> pk(z,k) setting.
'''
import numpy as np
import h5py
from matplotlib import pyplot as plt

import GPy

import emukit.multi_fidelity

from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import (make_non_linear_kernels,
    NonLinearMultiFidelityModel)

from emukit.multi_fidelity.convert_lists_to_array import (convert_x_list_to_array,
                                                          convert_xy_lists_to_arrays)

from lyaemu.matter_emulator import MatterEmulator
from lyaemu.gpemulator_emukit import MatterMultiFidelityLinearGP, PkMultiFidelityLinearGP

fname_low  = "../SimulationRunnerDM/data/lowRes/processed/test_dmonly.hdf5"
fname_high = "../SimulationRunnerDM/data/highRes/processed/test_dmonly.hdf5"

# setting different emulators
emu_low  = MatterEmulator(h5py.File(fname_low,  'r'))
emu_high = MatterEmulator(h5py.File(fname_high, 'r'))

# emu_low._get_interp(emu_low.X,  emu_low.kf, emu_low.Y)
# emu_high._get_interp(emu_high.X, emu_high.kf, emu_high.Y)

kf_list = [emu_low.kf, emu_high.kf]
powers_list = [
    emu_low.Y[:, -1, :], emu_high.Y[:, -1, :]] # only the last redshift
params_list = [emu_low.X, emu_high.X]
param_limits_list = [
    np.array(emu_low.parameter_space.get_bounds()),
    np.array(emu_high.parameter_space.get_bounds())]

n_points_h = params_list[-1].shape[0]

# the dimension for X and Y
# X : (n_points, n_dim)
# Y : (n_points, n_redshifts, k_modes)
gp = MatterMultiFidelityLinearGP(
    params_list, kf_list, param_limits_list, powers_list,
    kernel_list=None, n_fidelities=2)

# gp.mixed_noise.Gaussian_noise.fix(0)
# gp.mixed_noise.Gaussian_noise_1.fix(0)

lin_mf_model = GPyMultiOutputWrapper(gp, 2, n_optimization_restarts=20)

lin_mf_model.optimize()

# testing samples
# uniformly sample from parameter space
n_samples = 100

# remember to normalise
X_test = emu_high.parameter_space.sample_uniform(n_samples)
X_test = lin_mf_model.gpy_model._map_params_to_unit_cube(
    X_test, np.array(emu_high.parameter_space.get_bounds()))

X_test = convert_x_list_to_array([X_test, X_test])

lf_mean_lin_mf_model, lf_var_lin_mf_model = lin_mf_model.predict(X_test[:n_samples])
hf_mean_lin_mf_model, hf_var_lin_mf_model = lin_mf_model.predict(X_test[n_samples:])

lf_std_lin_mf_model = np.sqrt(lf_var_lin_mf_model)
hf_std_lin_mf_model = np.sqrt(hf_var_lin_mf_model)

# Plot the posterior mean and variance
i = 0
kf = kf_list[-1]
plt.fill_between(
    kf, 
    (lf_mean_lin_mf_model[i, :] - 10 * 1.96*lf_std_lin_mf_model[i, :]).flatten(),
    (lf_mean_lin_mf_model[i, :] + 10 * 1.96*lf_std_lin_mf_model[i, :]).flatten(),
    facecolor='grey', alpha=0.3, label="Linear MF lowRes")
plt.fill_between(
    kf,
    (hf_mean_lin_mf_model[i, :] - 10 * 1.96*hf_std_lin_mf_model[i, :]).flatten(),
    (hf_mean_lin_mf_model[i, :] + 10 * 1.96*hf_std_lin_mf_model[i, :]).flatten(),
    facecolor='tomato', alpha=0.3, label="Linear MF highRes")

plt.plot(kf, lf_mean_lin_mf_model[i, :], 'k')
plt.plot(kf, hf_mean_lin_mf_model[i, :], 'r')

plt.xlabel('log10(k)')
plt.ylabel('log10(P(k))')
plt.xlim(-2, 1)
plt.ylim(1, 5)
plt.legend()
plt.title("Comparison of linear multi-fidelity model's highRes and lowRes predictions")
plt.savefig('ps_mf_lin_mf_{}.pdf'.format(i), format='pdf', dpi=300)
plt.show()

# test interpolation on hubble at pk tail
# Plot the posterior mean and variance
ith_param = 0
hubble = X_test[:n_samples, ith_param]
argind = np.argsort(hubble)
plt.fill_between(
    hubble[argind], 
    (lf_mean_lin_mf_model[argind, -1] - 10 * 1.96*lf_std_lin_mf_model[argind, -1]).flatten(),
    (lf_mean_lin_mf_model[argind, -1] + 10 * 1.96*lf_std_lin_mf_model[argind, -1]).flatten(),
    facecolor='g', alpha=0.3)
plt.fill_between(
    hubble[argind],
    (hf_mean_lin_mf_model[argind, -1] - 10 * 1.96*hf_std_lin_mf_model[argind, -1]).flatten(),
    (hf_mean_lin_mf_model[argind, -1] + 10 * 1.96*hf_std_lin_mf_model[argind, -1]).flatten(),
    facecolor='y', alpha=0.3)

plt.plot(hubble[argind], lf_mean_lin_mf_model[argind, -1], 'b')
plt.plot(hubble[argind], hf_mean_lin_mf_model[argind, -1], 'r')

plt.show()


# train only on high fidelity
nparams = np.shape(params_list[-1])[1]

kernel = GPy.kern.Linear(nparams, ARD=True)
kernel += GPy.kern.RBF(nparams,   ARD=True)

# extrat normalised inputs from lin mf model
x_train_h = lin_mf_model.X[-n_points_h:, :-1]
y_train_h = lin_mf_model.Y[-n_points_h:, :]

high_gp_model = GPy.models.GPRegression(x_train_h, y_train_h, kernel)

high_gp_model.optimize_restarts(10)

hf_mean_high_gp_model, hf_var_high_gp_model = high_gp_model.predict(
    X_test[:n_samples, :-1])
hf_std_high_gp_model = np.sqrt(hf_var_high_gp_model)

# Plot the posterior mean and variance
i = 0
kf = kf_list[-1]

for i in range(10):
    plt.fill_between(
        kf, 
        (hf_mean_high_gp_model[i, :] - 10 * 1.96*hf_std_high_gp_model[i, :]).flatten(),
        (hf_mean_high_gp_model[i, :] + 10 * 1.96*hf_std_high_gp_model[i, :]).flatten(),
        facecolor='grey', alpha=0.1, label="High fidelity GP")
    plt.fill_between(
        kf,
        (hf_mean_lin_mf_model[i, :] - 10 * 1.96*hf_std_lin_mf_model[i, :]).flatten(),
        (hf_mean_lin_mf_model[i, :] + 10 * 1.96*hf_std_lin_mf_model[i, :]).flatten(),
        facecolor='tomato', alpha=0.3, label="Linear multi-fidelity GP")

    plt.plot(kf, hf_mean_high_gp_model[i, :], 'b')
    plt.plot(kf, hf_mean_lin_mf_model[i, :], 'tomato')

    plt.xlabel('log10(k)')
    plt.ylabel('log10(P(k))')
    plt.xlim(-2, 1)
    plt.ylim(1, 5)
    plt.legend()
    plt.title('Comparison of linear multi-fidelity model and high fidelity GP')
    plt.savefig('ps_mf_high_lin_mf_{}.pdf'.format(i), format='pdf', dpi=300)
    plt.show()

# test interpolation on hubble at pk tail
# Plot the posterior mean and variance
ith_param = 2

for ith_param in range(5):
    hubble = X_test[:n_samples, ith_param]
    argind = np.argsort(hubble)

    # transform back
    plimit = emu_high.parameter_space.get_bounds()[ith_param]
    hubble = hubble * (plimit[1] - plimit[0]) + plimit[0]

    plt.fill_between(
        hubble[argind], 
        (hf_mean_high_gp_model[argind, -1] - 10 * 1.96*hf_std_high_gp_model[argind, -1]).flatten(),
        (hf_mean_high_gp_model[argind, -1] + 10 * 1.96*hf_std_high_gp_model[argind, -1]).flatten(),
        facecolor='grey', alpha=0.1, label="High fidelity GP")
    plt.fill_between(
        hubble[argind],
        (hf_mean_lin_mf_model[argind, -1] - 10 * 1.96*hf_std_lin_mf_model[argind, -1]).flatten(),
        (hf_mean_lin_mf_model[argind, -1] + 10 * 1.96*hf_std_lin_mf_model[argind, -1]).flatten(),
        facecolor='tomato', alpha=0.3, label="Linear multi-fidelity GP")

    plt.plot(hubble[argind], hf_mean_high_gp_model[argind, -1], 'b')
    plt.plot(hubble[argind], hf_mean_lin_mf_model[argind, -1], 'tomato')
    plt.xlabel('{}'.format(emu_high.parameter_space.parameter_names[ith_param]))
    plt.ylabel('log10(P(k))[end]')
    plt.legend()
    plt.title('Comparison of linear multi-fidelity model and high fidelity GP')
    plt.savefig('{}_mf_high_lin_mf_{}.pdf'.format(
        emu_high.parameter_space.parameter_names[ith_param],i),
        format='pdf', dpi=300)
    plt.show()
