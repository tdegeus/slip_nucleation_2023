r"""
Collected data at synchronised avalanche area `A`,
for "plastic" blocks along the weak layer.

Usage:
  collect_sync-A_plastic.py [options] <files>...

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

import click
import docopt
import h5py
import numpy as np
import tqdm

# ==================================================================================================
# compute center of mass
# https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
# ==================================================================================================


def center_of_mass(x, L):
    if np.allclose(x, 0):
        return 0
    theta = 2.0 * np.pi * x / L
    xi = np.cos(theta)
    zeta = np.sin(theta)
    xi_bar = np.mean(xi)
    zeta_bar = np.mean(zeta)
    theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
    return L * theta_bar / (2.0 * np.pi)


def renumber(x, L):
    center = center_of_mass(x, L)
    N = int(L)
    M = int((N - N % 2) / 2)
    C = int(center)
    return np.roll(np.arange(N), M - C)


# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args["<files>"]
info = args["--info"]
source_dir = os.path.dirname(info)
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
    dt = float(data["/normalisation/dt"][...])
    t0 = float(data["/normalisation/t0"][...])
    sig0 = float(data["/normalisation/sig0"][...])
    eps0 = float(data["/normalisation/eps0"][...])

# ==================================================================================================
# ensemble average
# ==================================================================================================

left = int((nx - nx % 2) / 2 - 100)
right = int((nx - nx % 2) / 2 + 100 + 1)
Depsp = np.zeros((len(files), nx + 1, nx), dtype="float")  # #samples, A, r
Epspdot = np.zeros((len(files), nx + 1, nx), dtype="float")
Moving = np.zeros((len(files), nx + 1, nx), dtype="int")
Norm = np.zeros((len(files), nx + 1, nx), dtype="float")
Norm_dot = np.zeros((len(files), nx + 1, nx), dtype="float")

edx = np.empty((2, nx), dtype="int")
edx[0, :] = np.arange(nx)
dA = 50

for ifile, file in enumerate(tqdm.tqdm(files)):

    idnum = os.path.basename(file).split("_")[0]

    with h5py.File(os.path.join(source_dir, f"{idnum:s}.hdf5"), "r") as data:
        epsy = data["/cusp/epsy"][...]
        epsy = np.hstack((-epsy[:, 0].reshape(-1, 1), epsy))
        uuid = data["/uuid"].asstr()[...]

    with h5py.File(file, "r") as data:

        assert uuid == data["/meta/uuid"].asstr()[...]

        A = data["/sync-A/stored"][...]
        T = data["/sync-A/global/iiter"][...]

        idx0 = data[f"/sync-A/plastic/{np.min(A):d}/idx"][...]

        edx[1, :] = idx0
        i = np.ravel_multi_index(edx, epsy.shape)
        epsy_l = epsy.flat[i]
        epsy_r = epsy.flat[i + 1]
        epsp0 = 0.5 * (epsy_l + epsy_r)

        for ia, a in enumerate(A):

            if ia == 0:
                continue

            # read epsp_n

            if ia >= dA:

                a_n = A[ia - dA]
                idx_n = data[f"/sync-A/plastic/{a_n:d}/idx"][...]

                edx[1, :] = idx_n
                i = np.ravel_multi_index(edx, epsy.shape)
                epsy_l = epsy.flat[i]
                epsy_r = epsy.flat[i + 1]
                epsp_n = 0.5 * (epsy_l + epsy_r)

                if f"/sync-A/plastic/{a:d}/epsp" in data:
                    assert (
                        np.allclose(epsp_n, data[f"/sync-A/plastic/{a_n:d}/epsp"][...])
                        or a_n == 0
                    )

            # read epsp

            idx = data[f"/sync-A/plastic/{a:d}/idx"][...]

            edx[1, :] = idx
            i = np.ravel_multi_index(edx, epsy.shape)
            epsy_l = epsy.flat[i]
            epsy_r = epsy.flat[i + 1]
            epsp = 0.5 * (epsy_l + epsy_r)

            if f"/sync-A/plastic/{a:d}/epsp" in data and a > 0:
                assert np.allclose(epsp, data[f"/sync-A/plastic/{a:d}/epsp"][...])

            # rotate

            moved = idx0 != idx
            renum = renumber(np.argwhere(moved).ravel(), nx)
            moved = moved[renum]
            epsp0 = epsp0[renum]
            epsp = epsp[renum]

            if ia >= dA:
                epsp_n = epsp_n[renum]

            # adding to output

            Depsp[ifile, a, :] = epsp - epsp0
            Moving[ifile, a, :] = moved
            Norm[ifile, a, :] += 1

            if ia >= dA:
                Norm_dot[ifile, a, :] += 1
                Epspdot[ifile, a, :] = (epsp - epsp_n) / (T[a] - T[a_n])

# ==================================================================================================
# store
# ==================================================================================================

with h5py.File(output, "w") as data:

    # non-dimensionalising

    Depsp = Depsp / eps0
    Epspdot = Epspdot / eps0 / (dt / t0)

    # select

    i_dot = np.argwhere(np.mean(Norm_dot[:, :, 0], axis=0) >= 0.9).ravel()
    A_dot = np.arange(nx + 1)[i_dot]
    Moving_dot = Moving[:, i_dot, :]
    Norm_dot = Norm_dot[:, i_dot, :]
    Epspdot = Epspdot[:, i_dot, :]

    i = np.argwhere(np.mean(Norm[:, :, 0], axis=0) >= 0.9).ravel()
    A = np.arange(nx + 1)[i]
    Depsp = Depsp[:, i, :]
    Norm = Norm[:, i, :]
    Moving = Moving[:, i, :]

    data["/depsp/r"] = np.average(Depsp, weights=Norm, axis=0)
    data["/depsp/r"].attrs["norm"] = np.mean(Norm, axis=(0, 2))
    data["/depsp/moved"] = np.mean(Moving, axis=0)
    data["/depsp/mean/center"] = np.average(
        Depsp[:, :, left:right], weights=Norm[:, :, left:right], axis=(0, 2)
    )
    data["/depsp/mean/plastic"] = np.average(Depsp, weights=Norm, axis=(0, 2))
    data["/depsp/mean/crack"] = np.average(Depsp, weights=Moving, axis=(0, 2))
    data["/depsp/A"] = A

    data["/epspdot/r"] = np.average(Epspdot, weights=Norm_dot, axis=0)
    data["/epspdot/r"].attrs["norm"] = np.mean(Norm_dot, axis=(0, 2))
    data["/epspdot/moved"] = np.mean(Moving_dot, axis=0)
    data["/epspdot/mean/center"] = np.average(
        Epspdot[:, :, left:right], weights=Norm_dot[:, :, left:right], axis=(0, 2)
    )
    data["/epspdot/mean/plastic"] = np.average(Epspdot, weights=Norm_dot, axis=(0, 2))
    data["/epspdot/mean/crack"] = np.average(Epspdot, weights=Moving_dot, axis=(0, 2))
    data["/epspdot/A"] = A_dot
