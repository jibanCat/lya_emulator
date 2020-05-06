'''
Test the trained emukit emulators
'''
import numpy as np

from emukit.experimental_design import ExperimentalDesignLoop
from emukit.core import ParameterSpace, ContinuousParameter

from .gpemulator_emukit import SimpleGP, MatterGP
from .matter_emulator import modecount_rebin_multi_pk, modecount_rebin

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

def test_multi_bins():
    kk = np.array([0.0245437, 0.03471  , 0.0425109, 0.0490874, 0.0548814, 0.0601195,
       0.06942  , 0.0736311, 0.077614 , 0.0814022, 0.0850218, 0.0884935,
       0.0918341, 0.0981748, 0.101196 , 0.10413  , 0.106983 , 0.109763 ,
       0.112473 , 0.11512  ])
    pk = np.array([5.18543 , 3.64685 , 2.84354 , 2.44808 , 2.23373 , 2.09316 ,
       1.84305 , 1.70788 , 1.56009 , 1.42702 , 1.30557 , 1.18097 ,
       1.09881 , 0.976714, 0.906792, 0.867854, 0.843195, 0.828747,
       0.800168, 0.782737])
    modes = np.array([ 6., 12.,  8.,  6., 24., 24., 12., 30., 24., 24.,  8., 24., 48.,
        6., 48., 36., 24., 24., 48., 24.])

    k1, pk1 = modecount_rebin(kk, pk, modes)
    k2, pk2 = modecount_rebin_multi_pk(kk, pk[None, :], modes[None, :])

    pk2 = pk2[0, :]

    assert np.abs( (k1 - k2)   /  k1 ).sum() < 1e-4
    assert np.abs( (pk1 - pk2) / pk1 ).sum() < 1e-4

    k3, pk3 = modecount_rebin_multi_pk(kk, 
        np.vstack((pk,    pk)), 
        np.vstack((modes, modes)))
    
    assert np.abs( (pk3[0, :] - pk3[1, :]) / pk3[0, :] ).sum() < 1e-4

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

