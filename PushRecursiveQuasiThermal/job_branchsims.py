import os
import h5py
import numpy as np

dbase = "../../../data/nx=3^6x2"
N = (3 ** 6) * 2

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
    dt = float(data["/normalisation/dt"][...])
    l0 = float(data["/normalisation/l0"][...])
    cs = float(data["/normalisation/cs"][...])
    A = data["/avalanche/A"][...]
    idx = np.argwhere(A == N).ravel()
    incs = data["/avalanche/inc"][idx]
    files = data["/files"].asstr()[...][data["/avalanche/file"][idx]]
    stresses = data["/avalanche/sigd"][idx] * sig0

sigc = 0.15464095 * sig0
push_stresses = np.array([1.0 * sigc, 0.8 * sigc, 0.6 * sigc])
push_names = ["sigc-1d0", "sigc-0d8", "sigc-0d6"]

for stress, inc, file in zip(stresses, incs, files):

    for push_stress, push_name in zip(push_stresses, push_names):

        for T in [1e-7, 2e-7, 4e-7, 6e-7, 8e-7]:

            outfilename = "{:s}_inc={:d}_target={:s}_kBT={:.0e}.hdf5".format(
                file.split(".hdf5")[0], inc, push_name, T
            )

            print(file, inc, stress, push_name)

            with h5py.File(os.path.join(dbase, file), "r") as data:

                with h5py.File(outfilename, "w") as output:

                    for key in keys:
                        output[key] = data[key][...]

                    epsy0 = data["/cusp/epsy"][...]
                    N = epsy0.shape[0]
                    M = epsy0.shape[1] * 4
                    k = 2.0
                    epsy = 1.0e-5 + 1.0e-3 * np.random.weibull(k, size=(N * M)).reshape(
                        N, M
                    )
                    epsy[:, 0] += epsy0[:, -1]
                    epsy = np.cumsum(epsy, axis=1)
                    epsy_extendend = np.hstack((epsy0, epsy))
                    output["/cusp/epsy"] = epsy_extendend

                    output["/push/inc"] = inc
                    output["/push/stress"] = push_stress
                    output["/push/interval"] = 100
                    output["/push/kBT"] = T
                    output["/disp/0"] = data["disp"][str(inc)][...]

                    dset = output.create_dataset(
                        "/stored", (1,), maxshape=(None,), dtype=np.int
                    )
                    dset[0] = 0

                    dset = output.create_dataset(
                        "/sigd", (1,), maxshape=(None,), dtype=np.float
                    )
                    dset[0] = stress

                    dset = output.create_dataset(
                        "/t", (1,), maxshape=(None,), dtype=np.float
                    )
                    dset[0] = float(data["/t"][inc])
