r"""
Collected scatter data at the final increment and at A = N.

Usage:
  collect_final.py [options] <files>...

Arguments:
  <files>   Files from which to collect data.

Options:
  -o, --output=<N>  Output file. [default: output.hdf5]
  -i, --info=<N>    Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
  -f, --force       Overwrite existing output-file.
  -h, --help        Print help.
"""

import os
import sys
import docopt
import click
import h5py
import numpy as np
import GMatElastoPlasticQPot.Cartesian2d as gmat
import tqdm

# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args["<files>"]
info = args["--info"]
output = args["--output"]

for file in files + [info]:
    if not os.path.isfile(file):
        raise OSError(f'"{file:s}" does not exist')

if not args["--force"]:
    if os.path.isfile(output):
        print(f'"{output:s}" exists')
        if not click.confirm("Proceed?"):
            sys.exit(1)

# ==================================================================================================
# get constants
# ==================================================================================================

with h5py.File(files[0], "r") as data:
    plastic = data["/meta/plastic"][...]
    nx = len(plastic)
    h = np.pi

# ==================================================================================================
# get normalisation
# ==================================================================================================

with h5py.File(info, "r") as data:
    dt = data["/normalisation/dt"][...]
    t0 = data["/normalisation/t0"][...]
    eps0 = data["/normalisation/eps0"][...]
    sig0 = data["/normalisation/sig0"][...]
    nx = int(data["/normalisation/N"][...])

# ==================================================================================================
# ensemble average
# ==================================================================================================

# -------------------------------------------------------------------------
# initialise normalisation, and sum of first and second statistical moments
# -------------------------------------------------------------------------

final_ret = {"global": {}, "plastic": {}}
final_stored_global = np.zeros(len(files), dtype=np.int)
final_stored_epsp = np.zeros(len(files), dtype=np.int)
final_stored_plastic = np.zeros(len(files), dtype=np.int)

final_ret["global"]["sig_xx"] = np.zeros(len(files), dtype=np.float)
final_ret["global"]["sig_xy"] = np.zeros(len(files), dtype=np.float)
final_ret["global"]["sig_yy"] = np.zeros(len(files), dtype=np.float)
final_ret["global"]["iiter"] = np.zeros(len(files), dtype=np.int)

final_ret["plastic"]["sig_xx"] = np.zeros(len(files), dtype=np.float)
final_ret["plastic"]["sig_xy"] = np.zeros(len(files), dtype=np.float)
final_ret["plastic"]["sig_yy"] = np.zeros(len(files), dtype=np.float)
final_ret["plastic"]["epsp"] = np.zeros(len(files), dtype=np.float)

system_ret = {"global": {}, "plastic": {}}
system_stored_global = np.zeros(len(files), dtype=np.int)
system_stored_epsp = np.zeros(len(files), dtype=np.int)
system_stored_plastic = np.zeros(len(files), dtype=np.int)

system_ret["global"]["sig_xx"] = np.zeros(len(files), dtype=np.float)
system_ret["global"]["sig_xy"] = np.zeros(len(files), dtype=np.float)
system_ret["global"]["sig_yy"] = np.zeros(len(files), dtype=np.float)
system_ret["global"]["iiter"] = np.zeros(len(files), dtype=np.int)

system_ret["plastic"]["sig_xx"] = np.zeros(len(files), dtype=np.float)
system_ret["plastic"]["sig_xy"] = np.zeros(len(files), dtype=np.float)
system_ret["plastic"]["sig_yy"] = np.zeros(len(files), dtype=np.float)
system_ret["plastic"]["epsp"] = np.zeros(len(files), dtype=np.float)
system_ret["plastic"]["epspdot"] = np.zeros(len(files), dtype=np.float)

edx = np.empty((2, nx), dtype="int")
edx[0, :] = np.arange(nx)

# ---------------
# loop over files
# ---------------

pbar = tqdm.tqdm(files)

for ifile, file in enumerate(pbar):

    sim = os.path.basename(file).split("_")[0]

    with h5py.File(f"../{sim:s}.hdf5", "r") as data:
        epsy = data["/cusp/epsy"][...]
        epsy = np.hstack((-epsy[:, 0].reshape(-1, 1), epsy))

    with h5py.File(file, "r") as data:

        a = data["/sync-A/stored"][...]
        A_n = a[-50]
        A_0 = np.min(a)
        A = data["/sync-A/stored"][-1]

        idx_0 = data[f"/sync-A/plastic/{A_0:d}/idx"][...]

        edx[1, :] = idx_0
        i = np.ravel_multi_index(edx, epsy.shape)
        epsy_l = epsy.flat[i]
        epsy_r = epsy.flat[i + 1]
        epsp_0 = 0.5 * (epsy_l + epsy_r)

        if A == nx:

            system_stored_global[ifile] = 1

            system_ret["global"]["sig_xx"][ifile] = data["/sync-A/global/sig_xx"][A]
            system_ret["global"]["sig_xy"][ifile] = data["/sync-A/global/sig_xy"][A]
            system_ret["global"]["sig_yy"][ifile] = data["/sync-A/global/sig_yy"][A]
            system_ret["global"]["iiter"][ifile] = data["/sync-A/global/iiter"][A]

            if f"/sync-A/plastic/{A:d}/idx" in data:

                idx = data[f"/sync-A/plastic/{A:d}/idx"][...]
                idx_n = data[f"/sync-A/plastic/{A_n:d}/idx"][...]

                edx[1, :] = idx
                i = np.ravel_multi_index(edx, epsy.shape)
                epsy_l = epsy.flat[i]
                epsy_r = epsy.flat[i + 1]
                epsp = 0.5 * (epsy_l + epsy_r)

                edx[1, :] = idx_n
                i = np.ravel_multi_index(edx, epsy.shape)
                epsy_l = epsy.flat[i]
                epsy_r = epsy.flat[i + 1]
                epsp_n = 0.5 * (epsy_l + epsy_r)

                if f"/sync-A/plastic/{A:d}/epsp" in data:
                    assert np.allclose(epsp, data[f"/sync-A/plastic/{A:d}/epsp"][...])
                    assert (
                        np.allclose(epsp_n, data[f"/sync-A/plastic/{A_n:d}/epsp"][...])
                        or A_n == 0
                    )

                t = data["/sync-A/global/iiter"][A]
                t_n = data["/sync-A/global/iiter"][A_n]

                system_ret["plastic"]["epsp"][ifile] = np.mean(epsp - epsp_0)
                system_ret["plastic"]["epspdot"][ifile] = np.mean(epsp - epsp_n) / (
                    t - t_n
                )

                system_stored_epsp[ifile] = 1

            if f"/sync-A/element/{A:d}/sig_xx" in data:

                sig_xx = data[f"/sync-A/element/{A:d}/sig_xx"][...][plastic]
                sig_xy = data[f"/sync-A/element/{A:d}/sig_xy"][...][plastic]
                sig_yy = data[f"/sync-A/element/{A:d}/sig_yy"][...][plastic]

                system_ret["plastic"]["sig_xx"][ifile] = np.mean(sig_xx)
                system_ret["plastic"]["sig_xy"][ifile] = np.mean(sig_xy)
                system_ret["plastic"]["sig_yy"][ifile] = np.mean(sig_yy)

                system_stored_plastic[ifile] = 1

            elif f"/sync-A/plastic/{A:d}/sig_xx" in data:

                sig_xx = data[f"/sync-A/plastic/{A:d}/sig_xx"][...]
                sig_xy = data[f"/sync-A/plastic/{A:d}/sig_xy"][...]
                sig_yy = data[f"/sync-A/plastic/{A:d}/sig_yy"][...]

                system_ret["plastic"]["sig_xx"][ifile] = np.mean(sig_xx)
                system_ret["plastic"]["sig_xy"][ifile] = np.mean(sig_xy)
                system_ret["plastic"]["sig_yy"][ifile] = np.mean(sig_yy)

                system_stored_plastic[ifile] = 1

        if "/sync-t/stored" in data:

            final_stored_global[ifile] = 1

            T = data["/sync-t/stored"][-1]

            final_ret["global"]["sig_xx"][ifile] = data["/sync-t/global/sig_xx"][T]
            final_ret["global"]["sig_xy"][ifile] = data["/sync-t/global/sig_xy"][T]
            final_ret["global"]["sig_yy"][ifile] = data["/sync-t/global/sig_yy"][T]
            final_ret["global"]["iiter"][ifile] = data["/sync-t/global/iiter"][T]

            if f"/sync-t/plastic/{T:d}/idx" in data:

                idx = data[f"/sync-t/plastic/{T:d}/idx"][...]

                edx[1, :] = idx
                i = np.ravel_multi_index(edx, epsy.shape)
                epsy_l = epsy.flat[i]
                epsy_r = epsy.flat[i + 1]
                epsp = 0.5 * (epsy_l + epsy_r)

                if f"/sync-t/plastic/{T:d}/epsp" in data:
                    assert np.allclose(epsp, data[f"/sync-t/plastic/{T:d}/epsp"][...])

                final_ret["plastic"]["epsp"][ifile] = np.mean(epsp - epsp_0)

                final_stored_epsp[ifile] = 1

            if f"/sync-t/element/{T:d}/sig_xx" in data:

                sig_xx = data[f"/sync-t/element/{T:d}/sig_xx"][...][plastic]
                sig_xy = data[f"/sync-t/element/{T:d}/sig_xy"][...][plastic]
                sig_yy = data[f"/sync-t/element/{T:d}/sig_yy"][...][plastic]

                final_ret["plastic"]["sig_xx"][ifile] = np.mean(sig_xx)
                final_ret["plastic"]["sig_xy"][ifile] = np.mean(sig_xy)
                final_ret["plastic"]["sig_yy"][ifile] = np.mean(sig_yy)

                final_stored_plastic[ifile] = 1

            elif f"/sync-t/plastic/{T:d}/sig_xx" in data:

                sig_xx = data[f"/sync-t/plastic/{T:d}/sig_xx"][...]
                sig_xy = data[f"/sync-t/plastic/{T:d}/sig_xy"][...]
                sig_yy = data[f"/sync-t/plastic/{T:d}/sig_yy"][...]

                final_ret["plastic"]["sig_xx"][ifile] = np.mean(sig_xx)
                final_ret["plastic"]["sig_xy"][ifile] = np.mean(sig_xy)
                final_ret["plastic"]["sig_yy"][ifile] = np.mean(sig_yy)

                final_stored_plastic[ifile] = 1

# ------------
# extract data
# ------------

idx = np.argwhere(final_stored_global).ravel()

final_ret["global"]["sig_xx"] = final_ret["global"]["sig_xx"][idx]
final_ret["global"]["sig_xy"] = final_ret["global"]["sig_xy"][idx]
final_ret["global"]["sig_yy"] = final_ret["global"]["sig_yy"][idx]
final_ret["global"]["iiter"] = final_ret["global"]["iiter"][idx]

idx = np.argwhere(final_stored_plastic).ravel()

final_ret["plastic"]["sig_xx"] = final_ret["plastic"]["sig_xx"][idx]
final_ret["plastic"]["sig_xy"] = final_ret["plastic"]["sig_xy"][idx]
final_ret["plastic"]["sig_yy"] = final_ret["plastic"]["sig_yy"][idx]

idx = np.argwhere(final_stored_epsp).ravel()

final_ret["plastic"]["epsp"] = final_ret["plastic"]["epsp"][idx]

idx = np.argwhere(system_stored_global).ravel()

system_ret["global"]["sig_xx"] = system_ret["global"]["sig_xx"][idx]
system_ret["global"]["sig_xy"] = system_ret["global"]["sig_xy"][idx]
system_ret["global"]["sig_yy"] = system_ret["global"]["sig_yy"][idx]
system_ret["global"]["iiter"] = system_ret["global"]["iiter"][idx]

idx = np.argwhere(system_stored_plastic).ravel()

system_ret["plastic"]["sig_xx"] = system_ret["plastic"]["sig_xx"][idx]
system_ret["plastic"]["sig_xy"] = system_ret["plastic"]["sig_xy"][idx]
system_ret["plastic"]["sig_yy"] = system_ret["plastic"]["sig_yy"][idx]

idx = np.argwhere(system_stored_epsp).ravel()

system_ret["plastic"]["epsp"] = system_ret["plastic"]["epsp"][idx]
system_ret["plastic"]["epspdot"] = system_ret["plastic"]["epspdot"][idx]

# -----
# store
# -----

with h5py.File(output, "w") as data:

    Sig = np.zeros((len(system_ret["global"]["sig_xx"]), 2, 2))
    Sig[:, 0, 0] = system_ret["global"]["sig_xx"]
    Sig[:, 1, 1] = system_ret["global"]["sig_yy"]
    Sig[:, 0, 1] = system_ret["global"]["sig_xy"]
    Sig[:, 1, 0] = system_ret["global"]["sig_xy"]

    data["/A=N/global/iiter"] = system_ret["global"]["iiter"] * dt / t0
    data["/A=N/global/sig_xx"] = system_ret["global"]["sig_xx"] / sig0
    data["/A=N/global/sig_xy"] = system_ret["global"]["sig_xy"] / sig0
    data["/A=N/global/sig_yy"] = system_ret["global"]["sig_yy"] / sig0
    data["/A=N/global/sig_eq"] = gmat.Sigd(Sig) / sig0
    data["/A=N/global/sig_m"] = gmat.Hydrostatic(Sig) / sig0

    Sig = np.zeros((len(system_ret["plastic"]["sig_xx"]), 2, 2))
    Sig[:, 0, 0] = system_ret["plastic"]["sig_xx"]
    Sig[:, 1, 1] = system_ret["plastic"]["sig_yy"]
    Sig[:, 0, 1] = system_ret["plastic"]["sig_xy"]
    Sig[:, 1, 0] = system_ret["plastic"]["sig_xy"]

    data["/A=N/plastic/epsp"] = system_ret["plastic"]["epsp"] / eps0
    data["/A=N/plastic/epspdot"] = system_ret["plastic"]["epspdot"] / eps0 / (dt / t0)
    data["/A=N/plastic/sig_xx"] = system_ret["plastic"]["sig_xx"] / sig0
    data["/A=N/plastic/sig_xy"] = system_ret["plastic"]["sig_xy"] / sig0
    data["/A=N/plastic/sig_yy"] = system_ret["plastic"]["sig_yy"] / sig0
    data["/A=N/plastic/sig_eq"] = gmat.Sigd(Sig) / sig0
    data["/A=N/plastic/sig_m"] = gmat.Hydrostatic(Sig) / sig0

    Sig = np.zeros((len(final_ret["global"]["sig_xx"]), 2, 2))
    Sig[:, 0, 0] = final_ret["global"]["sig_xx"]
    Sig[:, 1, 1] = final_ret["global"]["sig_yy"]
    Sig[:, 0, 1] = final_ret["global"]["sig_xy"]
    Sig[:, 1, 0] = final_ret["global"]["sig_xy"]

    data["/final/global/iiter"] = final_ret["global"]["iiter"] * dt / t0
    data["/final/global/sig_xx"] = final_ret["global"]["sig_xx"] / sig0
    data["/final/global/sig_xy"] = final_ret["global"]["sig_xy"] / sig0
    data["/final/global/sig_yy"] = final_ret["global"]["sig_yy"] / sig0
    data["/final/global/sig_eq"] = gmat.Sigd(Sig) / sig0
    data["/final/global/sig_m"] = gmat.Hydrostatic(Sig) / sig0

    Sig = np.zeros((len(final_ret["plastic"]["sig_xx"]), 2, 2))
    Sig[:, 0, 0] = final_ret["plastic"]["sig_xx"]
    Sig[:, 1, 1] = final_ret["plastic"]["sig_yy"]
    Sig[:, 0, 1] = final_ret["plastic"]["sig_xy"]
    Sig[:, 1, 0] = final_ret["plastic"]["sig_xy"]

    data["/final/plastic/epsp"] = final_ret["plastic"]["epsp"] / eps0
    data["/final/plastic/sig_xx"] = final_ret["plastic"]["sig_xx"] / sig0
    data["/final/plastic/sig_xy"] = final_ret["plastic"]["sig_xy"] / sig0
    data["/final/plastic/sig_yy"] = final_ret["plastic"]["sig_yy"] / sig0
    data["/final/plastic/sig_eq"] = gmat.Sigd(Sig) / sig0
    data["/final/plastic/sig_m"] = gmat.Hydrostatic(Sig) / sig0
