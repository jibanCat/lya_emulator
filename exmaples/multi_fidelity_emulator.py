'''
Train and predict on Matter Multi-Fidelity emulator
'''
import time
import numpy as np
from matplotlib import pyplot as plt

from lyaemu.utility_multi_fidelity import gen_n_fidelities
from lyaemu.multi_fidelity_matter_emulators import MultiFidelityMatterEmulator
from lyaemu.gpemulator_emukit import PkMultiFidelityLinearGP

save_figure = lambda filename : plt.savefig("{}.pdf".format(filename), format="pdf", dpi=300)

emu_list = gen_n_fidelities([
    '../SimulationRunnerDM/data/lowRes/processed/test_dmonly.hdf5',
    '../SimulationRunnerDM/data/highRes/processed/test_dmonly.hdf5'])

mf_emu = MultiFidelityMatterEmulator(emu_list)

# get MF Emulator trained
# cost 2785.657336950302
tic = time.time()
mf_emu.get_interp()
print("Cost:", time.time() - tic)

# interpolate with nonlinear MF
mf_emu.get_interp_nonlinear()

# train on HF only
mf_emu.get_interp_high_fidelity()

# get an example to predict
X_test       = mf_emu.sample_uniform(mf_emu.matter_emulator_list[0], 1)
n_fidelities = len(mf_emu.kf_list)

param_cube = X_test[n_fidelities - 1, :-1][None, :]
x_plot     = np.linspace(-2, 2, 50)

# plot comparison between MF and HF predictions
mf_emu.plot_mf_hf_comparisons(param_cube, x_plot)
plt.ylim(-3, 5)
plt.show()

# test on test data
i   = 9
idx = -1 # assume z0=0
param_cube   = mf_emu.matter_emulator_list[1].X[i, :][None, :]
param_limits = np.array(mf_emu.matter_emulator_list[1].parameter_space.get_bounds())
powerspsc  = mf_emu.matter_emulator_list[1].Y[i, idx, :]
kf         = mf_emu.matter_emulator_list[1].kf

x_plot     = np.linspace(-2, 2, 50)
param_cube = PkMultiFidelityLinearGP._map_params_to_unit_cube(
    param_cube, param_limits)

# fractional errors
res_mean, res_var = mf_emu.predict(param_cube, kf)
res_mean = np.abs(10**(res_mean[:, 0]) - 10**powerspsc) / 10**powerspsc
print("Average fractional error : {}".format(res_mean.mean()))

mf_emu.plot_mf_hf_comparisons(param_cube, x_plot)
plt.plot(kf, powerspsc, label="True HF")
# residual plot
plt.plot(kf, np.log10(res_mean), label=r"$Mean(pred - true)/pred = {:.2g}$".format(
    res_mean.mean()), color="C3")
# plotting settings
plt.legend()
plt.ylim(-3, 5)
save_figure("test_MF_nonlinear_HF_{}".format(i))
plt.show()
