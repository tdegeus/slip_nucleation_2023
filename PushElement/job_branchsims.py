import os

import h5py
import numpy as np

dbase = "../../../data/nx=3^6x2"
N = (3**6) * 2

keys = [
    "/conn",
    "/coor",
    "/cusp/G",
    "/cusp/K",
    "/cusp/elem",
    # '/cusp/epsy',
    "/damping/alpha",
    "/damping/eta_d",
    "/damping/eta_v",
    "/dofs",
    "/dofsP",
    "/elastic/G",
    "/elastic/K",
    "/elastic/elem",
    "/rho",
    "/run/dt",
    "/run/epsd/kick",
    "/run/epsd/max",
    "/uuid",
]

with h5py.File(os.path.join(dbase, "EnsembleInfo.hdf5"), "r") as data:

    sig0 = float(data["/normalisation/sig0"][...])
    A = data["/avalanche/A"][...]
    idx = np.argwhere(A == N).ravel()
    incs = data["/avalanche/inc"][idx]
    files = data["/files"][...][data["/avalanche/file"][idx]]
    stresses = data["/avalanche/sigd"][idx] * sig0

for stress, inc, file in zip(stresses, incs, files):

    for realisation in range(1):

        outfilename = "{:s}_inc={:d}_branch={:d}.hdf5".format(
            file.split(".hdf5")[0], inc, realisation
        )

        print(file, inc, realisation)

        with h5py.File(os.path.join(dbase, file), "r") as data:

            with h5py.File(outfilename, "w") as output:

                for key in keys:
                    output[key] = data[key][...]

                epsy0 = data["/cusp/epsy"][...]
                N = epsy0.shape[0]
                M = epsy0.shape[1] * 4
                k = 2.0
                epsy = 1.0e-5 + 1.0e-3 * np.random.weibull(k, size=(N * M)).reshape(N, M)
                epsy[:, 0] += epsy0[:, -1]
                epsy = np.cumsum(epsy, axis=1)
                epsy_extendend = np.hstack((epsy0, epsy))
                output["/cusp/epsy"] = epsy_extendend

                output["/push/inc"] = inc
                output["/disp/0"] = data["disp"][str(inc)][...]

                dset = output.create_dataset("/stored", (1,), maxshape=(None,), dtype=np.int)
                dset[0] = 0

                dset = output.create_dataset("/sigd", (1,), maxshape=(None,), dtype=np.float)
                dset[0] = stress

                dset = output.create_dataset("/t", (1,), maxshape=(None,), dtype=np.float)
                dset[0] = float(data["/t"][inc])
