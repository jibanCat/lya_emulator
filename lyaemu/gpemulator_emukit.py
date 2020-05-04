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
from typing import Type, List, Tuple

import numpy as np
import scipy.linalg
import scipy.optimize

import GPy
from emukit.core.interfaces import IModel
from emukit.model_wrappers import GPyModelWrapper

from .latin_hypercube import map_to_unit_cube_list

class MatterGP(GPyModelWrapper):
    """
    A GP on params mapping to power spectrum bins wrapping on the emukit's
    GPy wrapper. The underlying GP functionality is hardcoded in the class,
    and the by using emukit's wrapper we should be able to take advantage of
    the Experimental Design Loop in emukit to do Bayes Opt.
    :param params: (n_points, n_dims)  parameter vectors.
    :param powers: (n_points, k modes) flux power spectra.
    :param param_limits: (n_dim, 2) param_limits is a list of parameter limits.
    :param n_restarts (int): number of optimization restarts you want in GPy.
    """
    def __init__(self, params: np.ndarray, powers: np.ndarray, 
            param_limits: np.ndarray):
        # basic input vectors for building a GPyRegression
        self.params       = params
        self.param_limits = param_limits # for mapping params to a unit cube
        self.powers       = powers

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

    def _get_interp(self, flux_vectors):
        """Build the actual interpolator."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        # Is this because the RBF length scale is the same for all dims?
        nparams = np.shape(self.params)[1]
        params_cube = map_to_unit_cube_list(self.params, self.param_limits)
        #Check that we span the parameter space
        for i in range(nparams):
            assert np.max(params_cube[:,i]) > 0.9
            assert np.min(params_cube[:,i]) < 0.1
        #print('Normalised parameter values =', params_cube)
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(flux_vectors, axis=1))[np.shape(flux_vectors)[0]//2]
        self.scalefactors = flux_vectors[medind,:]
        self.paramzero = params_cube[medind,:]
        #Normalise by the median value
        normspectra = flux_vectors/self.scalefactors -1.

        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)

        #Try rational quadratic kernel
        #kernel += GPy.kern.RatQuad(nparams)

        #noutput = np.shape(normspectra)[1]
        self.gp = GPy.models.GPRegression(params_cube, normspectra,kernel=kernel, noise_var=1e-10)

        status = self.gp.optimize(messages=False) #True
        #print('Gradients of model hyperparameters [after optimisation] =', self.gp.gradient)
        #Let's check that hyperparameter optimisation is converged
        if status.status != 'Converged':
            print("Restarting optimization")
            self.gp.optimize_restarts(num_restarts=10)
        #print(self.gp)
        #print('Gradients of model hyperparameters [after second optimisation (x 10)] =', self.gp.gradient)

    def _check_interp(self, flux_vectors):
        """Check we reproduce the input"""
        for i, pp in enumerate(self.params):
            means, _ = self.predict(pp.reshape(1,-1))
            worst = np.abs(np.array(means) - flux_vectors[i,:])/self.scalefactors
            if np.max(worst) > self.intol:
                print("Bad interpolation at:", np.where(worst > np.max(worst)*0.9), np.max(worst))
                assert np.max(worst) < self.intol

    def _predict(self, params, GP_instance):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = map_to_unit_cube_list(params, self.param_limits)
        flux_predict, var = GP_instance.predict(params_cube)
        mean = (flux_predict+1)*self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std

    def predict(self, params):
        """Get the predicted flux power spectrum (and error) at a parameter value
        (or list of parameter values)."""
        return self._predict(params, GP_instance=self.gp)

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
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


def make_non_linear_kernels(base_kernel_class: Type[GPy.kern.Kern],
        n_fidelities: int, n_input_dims: int) -> List[Type[GPy.kern.Kern]]:
    pass
