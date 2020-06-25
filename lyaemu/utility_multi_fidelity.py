'''
A collections of functions to help to convert single fidelity data structure
to multi-fidelity data structure.
'''
from typing import Tuple, List

import numpy as np
import h5py
from .matter_emulator import MatterEmulator

# fidelity generation function
def gen_n_fidelities(
        fname_list: List[str] = [
            "data/lowRes/processed/test_dmonly.hdf5",
            "data/highRes/processed/test_dmonly.hdf5"] ,
        ) -> List[MatterEmulator]:
    '''
    Generate the MatterEmulator for n fidelities. Primarily for holding the
    data, so we can latter on organise the inputs for the multi-fidelities.

    :param fname_list: a list of filenames of the HDF5 files generated by
        SimulationRunner.multi_sims.MultiPowerSpec;
        from low fidelity to high fidelity
        
    :return emu_n_fidelities: n MatterEmualtors with low->high fidelities
    '''
    assert type(fname_list[0]) is str

    n_fidelities = len(fname_list)

    print("[Info] Loading {} fidelity data ...".format(n_fidelities))

    # setting emulators for different fidelities
    emu_n_fidelities = []
    for fname in fname_list:
        emu_n_fidelities.append(MatterEmulator(h5py.File(fname,  'r')))

    assert len(emu_n_fidelities) == n_fidelities
    print("Done.")

    return emu_n_fidelities

