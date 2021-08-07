import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import os
import QPot
import sys
import tqdm

basename = os.path.splitext(os.path.basename(__file__))[0]

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import PinAndTrigger

with h5py.File(basename + ".h5", "w") as output:

    with h5py.File("PinAndTrigger_collect.h5", "r") as data:

        for stress in tqdm.tqdm(data["data"]):

            for A in tqdm.tqdm(data["data"][stress]):

                ret_stress_crack = []
                ret_A = []
                ret_S = []

                for simid in tqdm.tqdm(data["data"][stress][A]):

                    for incc in data["data"][stress][A][simid]:

                        for element in data["data"][stress][A][simid][incc]:

                            alias = data["data"][stress][A][simid][incc][element]

                            if "disp" not in alias:
                                continue

                            file = str(alias["file"].asstr()[...])

                            with h5py.File(file, "r") as s:
                                system = PinAndTrigger.initsystem(s)

                            dV = system.quad().AsTensor(2, system.quad().dV())
                            plastic = system.plastic()

                            system.setU(alias["disp"]["0"][...])
                            PinAndTrigger.pinsystem(system, int(element.split('=')[1]), int(A.split('=')[1]))

                            idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

                            system.setU(alias["disp"]["1"][...])

                            idx = system.plastic_CurrentIndex()[:, 0].astype(int)

                            sel = plastic[np.logical_not(pinned)]
                            Sig = system.Sig()
                            ret_stress += [GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))]
                            ret_S += [np.sum(idx - idx_n)]
                            ret_A += [np.sum(idx != idx_n)]

                output[f"/{stress:s}/{A:s}/stress/crack"] = ret_stress_crack
                output[f"/{stress:s}/{A:s}/S"] = ret_S
                output[f"/{stress:s}/{A:s}/A"] = ret_A
