"""
Building a surrogate using a Gaussian Process.
Emukit version.
Including a MultiFidelity GP Emulator.

:class: SimpleGP   : a practice to build a GP in emukit's syntax, similar
    to the one in emukit's repo.
:class: MultiBinGP : an emukit version of sbird's .gpemulator.MultiBinGP,
    slightly changed for matter power spectra only, typing, using emukit's
    GPy wrapper
:class: MatterGP   : an emukit version of sbird's .gpemulator.SkLearnGP,

"""
from typing import Type, List, Tuple, Optional

import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.interpolate import interp1d

import GPy
from emukit.core.interfaces import IModel
from emukit.model_wrappers import GPyModelWrapper

# multi-fidelity utility function
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import (convert_y_list_to_array,
    convert_xy_lists_to_arrays, convert_x_list_to_array)

from .latin_hypercube import map_to_unit_cube_list

class MatterGP(GPyModelWrapper):
    """
    A GP on params mapping to power spectrum bins wrapping on the emukit's
    GPy wrapper. The underlying GP functionality is hardcoded in the class,
    and the by using emukit's wrapper we should be able to take advantage of
    the Experimental Design Loop in emukit to do Bayes Opt.
    
    Note: Power spectra should be in loglog scale.

    :param params: (n_points, n_dims)  parameter vectors.
    :param powers: (n_points, k modes) flux power spectra.
    :param param_limits: (n_dim, 2) param_limits is a list of parameter limits.
    :param n_restarts (int): number of optimization restarts you want in GPy.
    """
    def __init__(self, params: np.ndarray, powers: np.ndarray, 
            param_limits: np.ndarray, n_restarts: int = 10):
        # basic input vectors for building a GPyRegression
        self.params       = params
        self.param_limits = param_limits # for mapping params to a unit cube
        self.powers       = powers
        self.n_restarts   = n_restarts

        # the minimum tolerance for the interpolation errors
        self.intol = 1e-4

        #Should we test the built emulator?
        #Turn this off because our emulator is now so large
        #that it always fails because of Gaussianity!
        self._test_interp : bool = False

        #Get the flux power and build an emulator
        # and inherent the GPyModelWrapper after initialize the GPyRegression
        self._get_interp(flux_vectors=powers)

        # Testing the interpolation
        if self._test_interp:
            self._check_interp(powers)
            self._test_interp = False

    @staticmethod
    def _normalize(flux_vectors: np.ndarray, params_cube: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize the flux_vectors to a zero mean vector
        using the median of flux_vectors

        :param flux_vectors: (n_points, k modes) flux power spectra.
        :param params_cube: (n_points, n_dims) parameter vectors in a unit cube.
        :return: (normspectra, scalefactors, paramzero)
        """
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(flux_vectors, axis=1))[
            np.shape(flux_vectors)[0]//2]

        scalefactors = flux_vectors[medind,:]
        paramzero    = params_cube[medind,:]

        #Normalise by the median value
        # -> shift power spectra to zero mean
        normspectra = flux_vectors / scalefactors -1.
        
        return normspectra, scalefactors, paramzero

    @staticmethod
    def _map_params_to_unit_cube(params: np.ndarray, 
            param_limits: np.ndarray) -> np.ndarray:
        '''
        Map the parameters onto a unit cube so that all the variations are
        similar in magnitude.
        
        :param params: (n_points, n_dims) parameter vectors
        :param param_limits: (n_dim, 2) param_limits is a list 
            of parameter limits.
        :return: params_cube, (n_points, n_dims) parameter vectors 
            in a unit cube.
        '''
        nparams = np.shape(params)[1]
        params_cube = map_to_unit_cube_list(
            params, param_limits)
        assert params_cube.shape[1] == nparams

        #Check that we span the parameter space
        # note: this is a unit LH cube spanning from Θ ∈ [0, 1]^num_dim
        for i in range(nparams):
            assert np.max(params_cube[:,i]) > 0.9
            assert np.min(params_cube[:,i]) < 0.1

        return params_cube

    def _get_interp(self, flux_vectors: np.ndarray) -> None:
        """
        Build the actual interpolator and get the GPyModelWrapper ready,
        inherent the GPyRegression to MatterGP.
        
        :param flux_vectors: (n_points, k modes) flux power spectra.
        """
        #Map the parameters onto a unit cube so that all the variations are
        # similar in magnitude.
        nparams = np.shape(self.params)[1]
        params_cube = self._map_params_to_unit_cube(
            self.params, self.param_limits)

        #Normalise by the median value
        # -> shift power spectra to zero mean
        normspectra, self.scalefactors, self.paramzero = self._normalize(
            flux_vectors, params_cube)

        #Standard squared-exponential kernel with a different length scale for 
        # each parameter, as they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)

        #Try rational quadratic kernel
        #kernel += GPy.kern.RatQuad(nparams)

        #noutput = np.shape(normspectra)[1]
        self.gp = GPy.models.GPRegression(
            params_cube, normspectra, kernel=kernel, noise_var=1e-10)

        # inherent the GP model to the GPyModelWrapper
        # this line put here because we want to change model 
        # when we re-do interp
        super().__init__(gpy_model= self.gp, n_restarts=self.n_restarts)

        self.optimize() # GPyModelWrapper use model.optimize_restarts

    def _check_interp(self, flux_vectors: np.ndarray):
        """Check we reproduce the input"""
        for i, pp in enumerate(self.params):
            means, _ = self.predict(pp.reshape(1,-1))
            worst = np.abs(np.array(means) - flux_vectors[i,:])/self.scalefactors
            if np.max(worst) > self.intol:
                print("Bad interpolation at:", np.where(worst > np.max(worst)*0.9), np.max(worst))
                assert np.max(worst) < self.intol

    def predict(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the predicted flux at a parameter value
        (or list of parameter values).

        :param params: (n_points, n_dim)
        :return: (mean, variance) array of size n_points x 1 and
            n_points x n_points of the predictive mean and variance at each
            input location
        """
        #Map the parameters onto a unit cube so that all the variations are 
        #similar in magnitude
        params_cube = map_to_unit_cube_list(params, self.param_limits)

        # make predictions using current model
        flux_predict, var = self.model.predict(params_cube)

        mean = (flux_predict + 1) * self.scalefactors
        std  = np.sqrt(var) * self.scalefactors
        return mean, std

    def predict_with_full_covariance(self, params: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the predicted flux at a parameter value
        (or list of parameter values).

        :param params: (n_points, n_dim)
        :return: (flux_predict, cov) array of size n_points x 1 and
            n_points x n_points of the predictive mean and variance at each
            input location
            Note: this is before normalization.
        """
        #Map the parameters onto a unit cube so that all the variations are 
        #similar in magnitude
        params_cube = map_to_unit_cube_list(params, self.param_limits)

        # make predictions using current model
        flux_predict, cov = self.model.predict(params_cube, full_cov=True)

        # mean = (flux_predict + 1) * self.scalefactors
        # std  = np.sqrt(var) * self.scalefactors
        return flux_predict, cov

    def get_predict_error(self, test_params: np.ndarray,
            test_exact: np.ndarray) -> np.ndarray:
        """
        Get the difference between the predicted GP
        interpolation and some exactly computed test parameters.
        """
        #Note: this is not used anywhere
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        predict, sigma = self.predict(test_params)
        return (test_exact - predict)/sigma


class SimpleGP(IModel):
    '''
    A hand-crafted GP model with RBF kernel.
    Modifications on the 
    https://github.com/amzn/emukit/blob/master/emukit/model_wrappers/simple_gp_model.py

    aiming to keep a hand-crafted version of the GP for the matter power
    spectrum emulator.

    TODO: with multi-dimensional outputs since we want params -> P(k, z)
        n_points = len(k); n_outputs = len(z).
    '''
    def __init__(self, x: np.ndarray, y: np.ndarray, 
            lengthscale: float = 1.0, kernel_variance: float = 1.0,
            likelihood_variance: float = 1.0, jitter: float = 1e-6):
        '''
        :param x: (n_points, n_dims) array training parameters
        :param y: (n_points, n_outputs)
        '''
        self.x = x
        self.y = y

        # kernel info
        self.lengthscale     = lengthscale
        self.kernel_variance = kernel_variance

        self.likelihood_variance = likelihood_variance
        self.jitter = jitter

    def __repr__(self):
        '''
        A pretty prints of hyperparameters
        '''
        return 'Simple Emukit GP: l: {:.4f}; sigma^2: {:.4f}; variance: {:.4f}'.format(
            self.lengthscale, self.kernel_variance, self.likelihood_variance)

    def optimize(self) -> None:
        '''
        Optimize those three hyperparameters of the model:
            kernel length scale, kernel variance, and likelihood variance
        '''
        def optimize_fcn(log_hyper_parameters):
            # exponential to ensure positive values
            hyper_parameters = np.exp(log_hyper_parameters)

            self.lengthscale         = hyper_parameters[0]
            self.kernel_variance     = hyper_parameters[1]
            self.likelihood_variance = hyper_parameters[2]

            return self._negative_mariginal_log_likelihood()
        
        lower_bound = np.log(1e-6)
        upper_bound = np.log(1e8)

        bounds = [(lower_bound, upper_bound) for _ in range(3)]

        scipy.optimize.minimize(optimize_fcn, 
            np.log(np.array([self.lengthscale, 
                             self.kernel_variance,
                             self.likelihood_variance])),
            bounds = bounds)

    def _calc_kernel(self, X: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        '''
        RBF kernel

        :param X:  (n_points_1, n_dims) input of first argument of kernel
        :param X2: (n_points_2, n_dims) input of second argument of kernel,
            If None K(X, X) is computed
        
        :return: Kernel matrix K(X, X2) or K(X, X)
        '''
        if X2 is None:
            X2 = X

        X1sq = np.sum(np.square(X), 1)
        X2sq = np.sum(np.square(X2), 1)

        r2 = -2. * np.dot( X, X2.T ) + (X1sq[:, None] + X2sq[None, :])
        r2 = np.clip(r2, 0, np.inf)
        
        return self.kernel_variance * np.exp(-0.5 * r2 / self.lengthscale ** 2)


    def _negative_mariginal_log_likelihood(self):
        '''
        :return: Negative marginal log likelihood of model with current 
            hyperparameters
        '''
        K = self._calc_kernel(self.x)

        # Add some jitter to the diagonal, guess is to avoid the denominator
        # too large
        K += np.identity(self.x.shape[0]) * (self.jitter 
            + self.likelihood_variance)

        # cholesky decomposition of covariance matrix
        # for a Hermitian positive-definite matrix K,
        # K = L L^dagger
        # L is a lower triangular matrix with real and positive diagonal
        # elements.
        L = np.linalg.cholesky(K)

        # Log determinant of the covariance matrix
        log_det = 2. * np.sum(np.log(np.diag(L)))

        # calculate y^T K^{-1} y
        # solve L x = y for x
        tmp   = scipy.linalg.solve_triangular(L,   self.y, lower=True)

        # solve L^T x = tmp = L^-1 y => x = (L^T)^-1 L^-1 y
        alpha = scipy.linalg.solve_triangular(L.T, tmp,    lower=False)

        log_2_pi = np.log(2 * np.pi)

        return -0.5 * (-self.y.size * log_2_pi - 
            self.y.shape[0] * log_det - np.sum(alpha * self.y))

    def predict(self, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Predict from model

        :param x_new: (n_points, n_dims) points query from predictive
            distribution
        :return: Tuple contains two (n_points, 1) arrays representing the
            mean and variance of the query locations
        '''
        K = self._calc_kernel(self.x)
        K += np.identity(self.x.shape[0]) * (
            self.jitter + self.likelihood_variance)
        
        L = np.linalg.cholesky(K)

        K_xs = self._calc_kernel(self.x, x_new)

        tmp  = scipy.linalg.solve_triangular(L, K_xs,   lower=True)
        tmp2 = scipy.linalg.solve_triangular(L, self.y, lower=True)

        mean     = np.dot(tmp.T, tmp2)
        variance = (self.kernel_variance - np.sum(np.square(tmp), axis=0))[:, None]
        return mean, variance


    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Set training data to new values

        :param X: (n_points, n_dims) training parameters
        :param Y: (n_points, 1)
        '''
        self.x = X
        self.y = Y

    @property
    def X(self) -> np.ndarray:
        return self.x
    
    @property
    def Y(self) -> np.ndarray:
        return self.y

class MatterMultiFidelityLinearGP(GPy.core.GP):
    '''
    A thin wrapper around GPy.core.GP that does some input checking and provides
    a default likelihood.
    Also transform the multi-fidelities params and powerspecs to the Multi-Output
    GP can take.
    And normalize the scale of powerspecs.

    :param params_list:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param kf_list:      (n_fidelities, k_modes)
    :param powers_list:  (n_fidelities, n_points, k modes) list of flux power spectra.
    :param param_limits: (n_fidelities, n_dim, 2) list of param_limits.

    :param n_fidelities: number of fidelities stored in the list.

    :param n_restarts (int): number of optimization restarts you want in GPy.
    '''
    def __init__(self, params_list: List[np.ndarray], kf_list: List[np.ndarray],
            param_limits_list: List[np.ndarray], powers_list : List[np.ndarray],
            kernel_list: Optional[List],
            n_fidelities: int, likelihood: GPy.likelihoods.Likelihood=None):
        # preparing X and Y
        # Y for different fidelities are required to be the same lengths
        # the treatment here is to limit the length of high fidelity using max
        # of low fidelity, and interpolate the low P(k) onto high kf
        max_kf = kf_list[0].max() # use the max kf of low fidelity

        # trim down fidelity > 1 to max_kf
        for i in range(1, n_fidelities):
            ind = kf_list[i] <= max_kf

            kf_list[i]     = kf_list[i][ind]
            powers_list[i] = powers_list[i][:, ind]

        # interpolate every fidelities < highest fidelity to the kf points of
        # the high fidelity
        kf_query = kf_list[-1] # highest kf as query points
        for i in range(n_fidelities - 1):
            n_points = powers_list[i].shape[0]
            k_modes  = kf_query.shape[0]
            interp_powers = np.empty((n_points, k_modes))

            for j in range(n_points):
                pk_interpolant = interp1d(kf_list[i], powers_list[i][j, :])
                
                interp_powers[j, :] = pk_interpolant(kf_query)

            # update the interpolated pk
            powers_list[i] = interp_powers
            kf_list[i]     = kf_query
            del interp_powers

        #Map the parameters onto a unit cube so that all the variations are
        # similar in magnitude.
        normed_param_list = []
        for i in range(n_fidelities):
            nparams = np.shape(params_list[i])[1]
            params_cube = self._map_params_to_unit_cube(
                params_list[i], param_limits_list[i])
            
            normed_param_list.append(params_cube)

        #Normalise by the median value
        # -> shift power spectra to zero mean
        normed_powers_list = []
        normspectra, scalefactors, paramzero = self._normalize(
            powers_list[-1], normed_param_list[-1])
        for i in range(n_fidelities - 1):
            normed_powers_list.append(
                powers_list[i] / scalefactors - 1)
        normed_powers_list.append(normspectra)

        # convert into X,Y for MultiOutputGP
        # not normalize due to no improvements
        X, Y = convert_xy_lists_to_arrays(normed_param_list, powers_list)

        if kernel_list == None:
            #Standard squared-exponential kernel with a different length scale for 
            # each parameter, as they may have very different physical properties.
            kernel_list = []
            for i in range(n_fidelities):
                nparams = np.shape(params_list[i])[1]

                kernel = GPy.kern.Linear(nparams)
                kernel += GPy.kern.RBF(nparams)
                kernel_list.append(kernel)

        # make multi-fidelity kernels
        kernel = LinearMultiFidelityKernel(kernel_list)

        # linear multi-fidelity setup
        if X.ndim != 2:
            raise ValueError('X should be 2d')

        if Y.ndim != 2:
            raise ValueError('Y should be 2d')

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError('One or more points has a higher fidelity index than number of fidelities')

        # Make default likelihood as different noise for each fidelity
        if likelihood is None:
            likelihood = GPy.likelihoods.mixed_noise.MixedNoise(
                [GPy.likelihoods.Gaussian(variance=1.) for _ in range(n_fidelities)])
        y_metadata = {'output_index': X[:, -1].astype(int)}
        super().__init__(X, Y, kernel, likelihood, Y_metadata=y_metadata)

    @staticmethod
    def _map_params_to_unit_cube(params: np.ndarray, 
            param_limits: np.ndarray) -> np.ndarray:
        '''
        Map the parameters onto a unit cube so that all the variations are
        similar in magnitude.
        
        :param params: (n_points, n_dims) parameter vectors
        :param param_limits: (n_dim, 2) param_limits is a list 
            of parameter limits.
        :return: params_cube, (n_points, n_dims) parameter vectors 
            in a unit cube.
        '''
        nparams = np.shape(params)[1]
        params_cube = map_to_unit_cube_list(
            params, param_limits)
        assert params_cube.shape[1] == nparams

        #Check that we span the parameter space
        # note: this is a unit LH cube spanning from Θ ∈ [0, 1]^num_dim
        for i in range(nparams):
            assert np.max(params_cube[:,i]) > 0.9
            assert np.min(params_cube[:,i]) < 0.1

        return params_cube

    @staticmethod
    def _normalize(flux_vectors: np.ndarray, params_cube: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize the flux_vectors to a zero mean vector
        using the median of flux_vectors

        :param flux_vectors: (n_points, k modes) flux power spectra.
        :param params_cube: (n_points, n_dims) parameter vectors in a unit cube.
        :return: (normspectra, scalefactors, paramzero)
        """
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(flux_vectors, axis=1))[
            np.shape(flux_vectors)[0]//2]

        scalefactors = flux_vectors[medind,:]
        paramzero    = params_cube[medind,:]

        #Normalise by the median value
        # -> shift power spectra to zero mean
        normspectra = flux_vectors / scalefactors -1.
        
        return normspectra, scalefactors, paramzero


def make_non_linear_kernels(base_kernel_class: Type[GPy.kern.Kern],
        n_fidelities: int, n_input_dims: int) -> List[Type[GPy.kern.Kern]]:
    pass
