'''
Test the utility function
'''
import numpy as np
import h5py
from ..utility_multi_fidelity import HDF5Holder

def test_hdf5_addition(file1: str, file2: str):
    '''
    test the addition of two multi_power_specs
    '''
    f1 = HDF5Holder(file1, 'r')
    f2 = HDF5Holder(file2, 'r')

    f3 = f1 + f2

    parameter_names = f1['parameter_names'][()]

    assert np.all( f3['parameter_names'][()] == parameter_names )
    assert np.all( f3['bounds'][()] == f1['bounds'][()] )
    assert np.all( f3['bounds'][()] == f2['bounds'][()] )

    assert np.all(f3['simulation_1']['powerspecs'][()]
        == f1['simulation_1']['powerspecs'][()])

    assert (f3[parameter_names[0]].shape[0]
        == (f1[parameter_names[0]].shape[0] + f2[parameter_names[0]].shape[0]) )
