'''
Test the trained emukit emulators
'''
import numpy as np

from emukit.experimental_design import ExperimentalDesignLoop
from emukit.core import ParameterSpace, ContinuousParameter

from .gpemulator_emukit import SimpleGP, MatterGP

class Power(object):
    """Mock power object"""
    def __init__(self, params: np.ndarray):
        '''
        :param params: (n_points, n_dims) array training parameters
        '''
        self.params = params

    def get_power(self, *, kf):
        """Get the power spectrum."""
        flux_vector = kf * 100.

        # amplitude depending linearly on one parameter and a single value
        return flux_vector * self.params

def test_single_gp():
    """
    Generate the simplest model possible,
        with an amplitude depending linearly on
        one parameter and a single value.
    """
    kf = np.array([ 0.00141,  0.00178,  0.00224,  0.00282])

    params = np.reshape(np.linspace(0.25,1.75,10), (10,1))
    powers = np.array([Power(par).get_power(kf=kf) for par in params])
    plimits = np.array((0.25,1.75),ndmin=2)

    gp = MatterGP(params = params, powers = powers, param_limits = plimits)
    predict,_ = gp.predict(np.reshape(np.array([0.5]), (1,1)))

    assert np.sum(np.abs(predict - 0.5 * kf*100)/predict) < 1e-4

# # TODO: not quite working yet
# def test_emukit_loop():
#     """
#     Test if the emukit Experiment Loop can work
#     """
#     kf = np.array([ 0.00141,  0.00178,  0.00224,  0.00282])

#     params = np.reshape(np.linspace(0.25,1.75,10), (10,1))
#     powers = np.array([Power(par).get_power(kf=kf) for par in params])
#     plimits = np.array((0.25,1.75),ndmin=2)

#     gp = MatterGP(params = params, powers = powers, param_limits = plimits)

#     # define your parameters
#     p     = ContinuousParameter('p', plimits[0, 0], plimits[0, 1])
#     space = ParameterSpace([p])

#     # generation function
#     def emulated_function(params: np.ndarray, param_limits: np.ndarray = plimits,
#             kf: np.ndarray = kf) -> np.ndarray:
#         # simple scaling relation
#         flux_vectors = kf * 100. * params

#         # rescaling and normalization
#         params_cube = gp._map_params_to_unit_cube(
#             params, param_limits)
#         normspectra, _, _ = gp._normalize(
#             flux_vectors, params_cube)

#         return normspectra

#     # experimental design loop
#     loop = ExperimentalDesignLoop(space, gp)
#     loop.run_loop(emulated_function, 30)

