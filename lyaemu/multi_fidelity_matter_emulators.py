'''
Multi-Fidelity Emulator applied on matter power spectrum
'''
from typing import List, Type, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
import h5py
import GPy

# IModel is the base class to define a workable emukit Emulator
# we should code up our own gradients for multi-fidelity, so IDifferentiable
# should be imported
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.multi_fidelity.convert_lists_to_array import convert_y_list_to_array
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array

# Multi-Fidelity is basically a Multi-Output GP, so we need to use a
# Multi-Output wrapper wrap on top of the GPs
# Note: Multi-Output GP: An additional Covariance on top of different GPs
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper

from .matter_emulator import MatterEmulator
from .gpemulator_emukit import MatterMultiFidelityLinearGP, PkMultiFidelityLinearGP, PkMultiFidelityNonLinearGP

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
            minmodes: int = 20, ndesired : int = 200, z0: int = 0,
            number_samples_list: List[int] = [10, 3]):
        self.matter_emulator_list = matter_emulator_list
        
        self.minmodes = minmodes
        self.ndesired = ndesired
        self.z0       = z0

        # parepare the multi-fidelity inputs and outputs to interpolate.
        # they would be saved into self
        self.prepare_mf_inputs(self.matter_emulator_list,
            minmodes=minmodes, ndesired=ndesired, z0=z0,
            number_samples_list=number_samples_list)

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

    def get_interp(self, n_optimization_restarts: int = 5):
        '''
        A wrapper over self._get_interp, to make the function arguments easier
        read.
        '''
        n_fidelities = len(self.kf_list)
        self._get_interp(self.param_list, self.kf_list, self.param_limits_list,
            self.powers_list, n_fidelities=n_fidelities,
            n_optimization_restarts=n_optimization_restarts)

    def _get_interp(self, params_list: List[np.ndarray], kf_list: List[np.ndarray],
            param_limits_list: List[np.ndarray], powers_list: List[np.ndarray],
            n_fidelities: int, kernel_list=None,
            n_optimization_restarts: int = 5) -> None:
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
        
        # fixed the noise to 0, since we assume simulations have no noise
        getattr(gp.mixed_noise, 'Gaussian_noise').fix(0)
        for i in range(1, n_fidelities):
            getattr(gp.mixed_noise, 'Gaussian_noise_{}'.format(i)).fix(0)

        model = GPyMultiOutputWrapper(gp, n_outputs=n_fidelities,
            n_optimization_restarts=n_optimization_restarts)

        self.gp = gp
        self.model = model

        self.model.optimize()

    def get_interp_nonlinear(self, n_samples : int = 100,
    n_optimization_restarts: int = 5):
        '''
        A wrapper over self._get_interp_nonlinear, to make the function
        arguments easier to read.
        '''
        n_fidelities = len(self.kf_list)
        self._get_interp_nonlinear(self.param_list, self.kf_list, self.param_limits_list,
            self.powers_list, n_fidelities=n_fidelities,
            n_samples=n_samples,
            n_optimization_restarts=n_optimization_restarts)

    def _get_interp_nonlinear(self, params_list: List[np.ndarray], kf_list: List[np.ndarray],
            param_limits_list: List[np.ndarray], powers_list: List[np.ndarray],
            n_fidelities: int, kernel_list=None,
            n_samples : int = 100,
            n_optimization_restarts: int = 5) -> None:
        '''
        GP interpolate on the different fidelity.
        '''
        model_nonlin = PkMultiFidelityNonLinearGP(params_list, kf_list, param_limits_list, powers_list,
            n_fidelities=n_fidelities,
            n_samples=n_samples,
            optimization_restarts=n_optimization_restarts)

        # fixed the noise to 0, since we assume simulations have no noise
        for m in model_nonlin.models:
            m.Gaussian_noise.variance.fix(0)

        self.model_nonlin = model_nonlin
        self.model_nonlin.optimize()

    def prepare_mf_inputs(self, matter_emulator_list : List[MatterEmulator],
            minmodes: int = 20, ndesired : int = 200, z0: int = 0,
            number_samples_list: List[int] = [10, 3],
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
                minmodes, ndesired))
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

    # MF prediction function: only make highest fidelity predictions here
    def predict(self, param_cube: np.ndarray, x_plot:np.ndarray) -> Tuple[np.ndarray]:
        '''
        :param params_cube: (1, n_dim) without k_mode dimension.
        :param x_plot: (k_points, ) the x points we want to plot
        :return: (mean_pk_mf_hf, std_pk_mf_hf)
            mean P(k) for HF output and its STD.
        '''
        n_fidelities = len(self.kf_list)

        # Multi-Fidelity GP predictions
        x_sample_plot = self.make_x_plot(param_cube, x_plot)
        # plot highRes output
        X_hf_sample_plot = np.concatenate(
            (x_sample_plot, np.ones((len(x_plot), 1)) * (n_fidelities - 1)),
            axis=1)

        # MF: HF output
        mean_pk_mf_hf, var_pk_mf_hf = self.model.predict(X_hf_sample_plot)
        std_pk_mf_hf = np.sqrt(var_pk_mf_hf)
        
        return mean_pk_mf_hf, std_pk_mf_hf

    @staticmethod
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

    def predict_uniform(self):
        '''
        Make predictions from uniformly sampling the parameter space (not k space).
        '''
        X_test = self.sample_uniform(self.matter_emulator_list[-1],
            n_samples=1)
        
        n_fidelities = len(self.kf_list)
        
        param_cube   = X_test[n_fidelities - 1, :-1][None, :]
        x_plot = np.linspace(-2, 2, 50)

        # makes prediction and interpolation on P(k) based on a single param_cube
        mean_pk_mf_hf, std_pk_mf_hf = self.predict(param_cube, x_plot)

        return mean_pk_mf_hf, std_pk_mf_hf

    def sample_uniform(self, emu: MatterEmulator, n_samples : int = 10
            ) -> np.ndarray:
        '''
        sample uniformly in the cosmological parameter space (not k space).
        '''
        # uniformly sample
        X_test = emu.parameter_space.sample_uniform(n_samples)

        return self._convert_mf_inputs(
            X_test, np.array(emu.parameter_space.get_bounds()), len(self.kf_list))

    @staticmethod
    def _convert_mf_inputs(X: np.ndarray, param_limits: np.ndarray,
            n_fidelities: int) -> np.ndarray:
        assert n_fidelities > 1
        X = PkMultiFidelityLinearGP._map_params_to_unit_cube(
            X, param_limits)
        X = convert_x_list_to_array([X for _ in range(n_fidelities)])
        
        return X

    # plotting functions
    @staticmethod
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

    # making HF only model for comparison
    def get_interp_high_fidelity(self, n_optimization_restarts: int = 5) -> None:
        '''
        Generate a HF only GP
        '''
        n_fidelities = len(self.kf_list)

        # fit High-fidelity only
        #Map the parameters onto a unit cube so that all the variations are
        # similar in magnitude.
        nparams = np.shape(self.param_list[n_fidelities-1])[1]
        params_cube = PkMultiFidelityLinearGP._map_params_to_unit_cube(
            self.param_list[n_fidelities-1][:, :-1],
            self.param_limits_list[n_fidelities-1])
        
        X_hf = np.concatenate(
            (params_cube, self.param_list[n_fidelities-1][:, -1:]), axis=1)
        Y_hf = self.powers_list[n_fidelities-1]

        # modelling setup
        # make sure turn on ARD
        nparams = np.shape(self.param_list[-1])[1]
        kernel  = GPy.kern.Linear(nparams, ARD=True)
        kernel += GPy.kern.RBF(nparams, ARD=True)

        model_hf = GPy.models.GPRegression(X_hf, Y_hf, kernel)
        model_hf.Gaussian_noise.fix(0)
        model_hf.optimize_restarts(n_optimization_restarts)

        self.model_hf = model_hf

    def predict_hf(self, param_cube: np.ndarray, x_plot: np.ndarray):
        '''
        Make predictions using HF-only model
        '''
        x_sample_plot = self.make_x_plot(param_cube, x_plot)

        mean_pk_hf, var_pk_hf = self.model_hf.predict(x_sample_plot)
        std_pk_hf = np.sqrt(var_pk_hf)

        return mean_pk_hf, std_pk_hf

    def plot_mf_hf_comparisons(self, param_cube: np.ndarray, x_plot: np.ndarray):
        '''
        Plot the interpolation of matter powerspecs of a multi-fidelity model
        and a high-fidelity model.
        '''
        mean_pk_mf_hf, std_pk_mf_hf = self.predict(param_cube, x_plot)
        mean_pk_hf, std_pk_hf       = self.predict_hf(param_cube, x_plot)

        # plotting functions here:
        self.plot_pk(x_plot, mean_pk_mf_hf[:, 0], std_pk_mf_hf[:, 0],
            label='MF: HF output', color="C0")
        self.plot_pk(x_plot, mean_pk_hf[:, 0], std_pk_hf[:, 0],
            label='HF-only', color="C1")
        plt.legend()
