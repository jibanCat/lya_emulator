'''
functions to test the designs
'''
import numpy as np
from .latin_design import LatinDesign
from .matter_power_design import MatterDesign

def test_matter_design() -> None:
    matter = MatterDesign(
        omegab_bounds=(0.0452, 0.0492), 
        omega0_bounds=(0.268, 0.308), 
        hubble_bounds=(0.65, 0.75), 
        scalar_amp_bounds=(1.5*1e-9, 2.8*1e-9), 
        ns_bounds=(0.9, 0.99))

    point_count = 10

    samples = matter.get_samples(point_count)

    assert np.all(samples[:, 0] > matter.parameter_space.get_bounds()[0][0])
    assert np.all(samples[:, 0] < matter.parameter_space.get_bounds()[0][1])

    # more points
    point_count = 100

    samples = matter.get_samples(point_count)

    assert np.all(samples[:, 0] > matter.parameter_space.get_bounds()[0][0])
    assert np.all(samples[:, 0] < matter.parameter_space.get_bounds()[0][1])

    # different order
    assert np.all(samples[:, -1] > matter.parameter_space.get_bounds()[-1][0])
    assert np.all(samples[:, -1] < matter.parameter_space.get_bounds()[-1][1])
