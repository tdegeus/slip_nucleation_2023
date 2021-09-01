r"""
For avalanches synchronised at avalanche area `A`,
compute how dense a crack is on average: how many of the cracks yielded at a certain position
along the crack.

Usage:
    collect_sync-A_crack-density.py [options] <files>...

Arguments:
    <files>     Files from which to collect data.

Options:
    -o, --output=<N>    Output file. [default: output.hdf5]
    -i, --info=<N>      Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
    -f, --force         Overwrite existing output-file.
    -h, --help          Print help.
"""

import os
import sys
import docopt
import click
import h5py
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

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


def threshold(i, N):
    if i < 0:
        return i
    if i > N:
        return N
    return i


# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args["<files>"]
output = args["--output"]
info = args["--info"]

for file in files:
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

with h5py.File(info, "r") as data:
    dt = data["/normalisation/dt"][...]
    t0 = data["/normalisation/t0"][...]
    nx = int(data["/normalisation/N"][...])
    mid = int((nx - nx % 2) / 2)

# ==================================================================================================
# build histogram
# ==================================================================================================

niter = 100000
iiter_sync = 80000
count = np.zeros((niter, nx), dtype="int")  # [t, r]
norm = np.zeros((niter), dtype="int")  # [t]

for ifile, file in enumerate(files):

    print(f"({ifile + 1:3d}/{len(files):3d}) {file:s}")

    with h5py.File(file, "r") as data:

        A = data["/sync-A/stored"][...]
        idx0 = data[f"/sync-A/plastic/{np.min(A):d}/idx"][...]
        iiter = data["/sync-A/global/iiter"][...][A].astype(np.int)
        assert np.all(np.unique(A) == A)
        assert np.all(np.diff(A) > 0)
        assert np.all(np.diff(iiter) > 0)
        iiter_ref = int(iiter[np.argmin(np.abs(A - int(0.8 * nx)))])
        shift = int(iiter_sync - iiter_ref)
        iiter += shift
        iiter = np.where(iiter >= 0, iiter, 0)
        iiter = np.where(iiter <= niter, iiter, niter)
        iiter_n = np.roll(np.array(iiter, copy=True), 1)
        iiter_n[0] = iiter_n[1]
        assert np.all(iiter - iiter_n >= 0)

        for i, j, a in zip(iiter_n, iiter, A):
            idx = data[f"/sync-A/plastic/{a:d}/idx"][...]
            cracked = (idx != idx0).astype(np.int)
            renum = renumber(np.argwhere(cracked).ravel(), nx)
            count[i:j, :] += cracked[renum]
            norm[i:j] += 1

        assert np.sum(cracked) == nx
        count[j:, :] += 1
        norm[j:] += 1
        assert np.max(norm) <= ifile + 1
        assert np.min(norm) >= 0

# ==================================================================================================
# save data
# ==================================================================================================

with h5py.File(output, "w") as data:

    data["/t"] = (np.arange(niter) - iiter_sync) * dt / t0
    data["/P"] = count / np.where(norm > 0, norm, 1).reshape(-1, 1)
    data["/norm"] = norm

    data["/t"].attrs[
        "desc"
    ] = "Time at which at which /P is stored, relative to synchronisation at A = Ac"
    data["/P"].attrs[
        "desc"
    ] = "Probability that a block yielded: realisations are centered before averaging"
    data["/P"].attrs["shape"] = "[len(/t), N] or [t, x]"
    data["/norm"].attrs["desc"] = "Number of measurements per A"
