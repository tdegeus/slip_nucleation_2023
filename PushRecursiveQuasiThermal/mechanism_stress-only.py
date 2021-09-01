import os
import subprocess

import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseMPL as gplt
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["goose", "goose-latex"])

with h5py.File("EnsembleInfo.hdf5", "r") as data:
    sig0 = float(data["/normalisation/sig0"][...])
    dt = float(data["/normalisation/dt"][...])
    l0 = float(data["/normalisation/l0"][...])
    cs = float(data["/normalisation/cs"][...])
    N = float(data["/normalisation/N"][...])

with h5py.File("CrackEvolutionInfo.hdf5", "r") as data:
    sigc = data["/detail/last"][0]

files = sorted(
    list(
        filter(
            None,
            subprocess.check_output("find . -iname 'id*push=*.hdf5'", shell=True)
            .decode("utf-8")
            .split("\n"),
        )
    )
)

for ifile, file in enumerate(files):

    with h5py.File(file, "r") as data:

        if ifile == 0:
            xx = data["/overview/global/sig"].attrs["xx"][...]
            xy = data["/overview/global/sig"].attrs["xy"][...]
            yy = data["/overview/global/sig"].attrs["yy"][...]

        fig, ax = gplt.subplots(ncols=1, scale_y=2)

        if "event" not in data:
            continue
        if "global" not in data["event"]:
            continue
        if "A" not in data["event"]["global"]:
            continue

        t = data["/overview/global/iiter"][...] * dt * cs / (l0 * N)

        tlim = np.max(t)

        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, tlim])
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(r"$t$")

        sig = data["/overview/weak/sig"][...]
        Sig = np.zeros([sig.shape[1], 2, 2])
        Sig[:, 0, 0] = sig[xx, :]
        Sig[:, 0, 1] = sig[xy, :]
        Sig[:, 1, 0] = sig[xy, :]
        Sig[:, 1, 1] = sig[yy, :]
        sig_weak = GMat.Sigd(Sig)

        sig = data["/overview/global/sig"][...]
        Sig = np.zeros([sig.shape[1], 2, 2])
        Sig[:, 0, 0] = sig[xx, :]
        Sig[:, 0, 1] = sig[xy, :]
        Sig[:, 1, 0] = sig[xy, :]
        Sig[:, 1, 1] = sig[yy, :]
        sig_glob = GMat.Sigd(Sig)

        ax.plot(sig_weak / sig0, t, c="r", lw=1)
        ax.plot(sig_glob / sig0, t, c="k")

        ax.plot([sigc, sigc], [0, tlim], c="b", ls="--")
        ax.plot([0.8 * sigc, 0.8 * sigc], [0, tlim], c="g", ls="--")
        ax.plot([0.6 * sigc, 0.6 * sigc], [0, tlim], c="c", ls="--")

        ax.plot(ax.get_xlim(), [np.max(t), np.max(t)], c="g", ls="-", lw=1)

        print(file)
        fig.savefig("{:s}".format(os.path.basename(file).replace(".hdf5", ".pdf")))
        plt.close(fig)
