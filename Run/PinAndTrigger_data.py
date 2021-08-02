import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import os
import QPot
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import PinAndTrigger  


with h5py.File("id=000.hdf5", "r") as data:

    system = PinAndTrigger.initsystem(data)

with h5py.File("stress=3d6_id=000_element=0.hdf5", "r") as data:

    system.setU(data["/disp/0"][...])
    PinAndTrigger.pinsystem(system, data["/meta/PushAndTrigger/target_element"][...], data["/meta/PushAndTrigger/target_A"][...])

    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

    system.setU(data["/disp/1"][...])

    idx = system.plastic_CurrentIndex()[:, 0].astype(int)

    print(np.sum(idx - idx_n))
    print(np.sum(idx != idx_n))
    # print(system.plastic_CurrentYieldLeft()[500, 0], system.plastic_CurrentYieldRight()[500, 0], system.plastic_Epsp()[500, 0])
