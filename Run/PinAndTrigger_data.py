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

with h5py.File("PinAndTrigger.h5", "r") as data:

    for stress in data:

        for A in data[stress]:

            for simid in data[stress][A]:

                for incc in data[stress][A][simid]:

                    for element in data[stress][A][simid][incc]:

                        alias = data[stress][A][simid][incc][element]
                        file = str(alias["file"].asstr()[...])

                        with h5py.File(file, "r") as s:
                            system = PinAndTrigger.initsystem(s)

                        system.setU(alias["disp"]["0"][...])
                        PinAndTrigger.pinsystem(system, int(element.split('=')[1]), int(A.split('=')[1]))

                        idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

                        system.setU(alias["disp"]["1"][...])

                        idx = system.plastic_CurrentIndex()[:, 0].astype(int)

                        print(np.sum(idx - idx_n))
                        print(np.sum(idx != idx_n))
                        # print(system.plastic_CurrentYieldLeft()[500, 0], system.plastic_CurrentYieldRight()[500, 0], system.plastic_Epsp()[500, 0])
