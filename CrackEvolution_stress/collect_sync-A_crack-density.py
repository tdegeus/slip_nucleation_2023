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

import click
import docopt
import h5py
import numpy as np

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


def mean_renumber(L, *args):
    centers = []
    for x in args:
        centers += [center_of_mass(x, L)]
    center = np.mean(centers)
    N = int(L)
    M = int((N - N % 2) / 2)
    C = int(center)
    return np.roll(np.arange(N), M - C)


# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args["<files>"]
output = args["--output"]

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

with h5py.File(files[0], "r") as data:
    plastic = data["/meta/plastic"][...]
    nx = len(plastic)
    mid = int((nx - nx % 2) / 2)

# ==================================================================================================
# build histogram
# ==================================================================================================

count = np.zeros((nx + 1, nx), dtype="int")  # [A, r]
norm = np.zeros((nx + 1), dtype="int")  # [A]

for ifile, file in enumerate(files):

    with h5py.File(file, "r") as data:

        print(f"({ifile + 1:3d}/{len(files):3d}) {file:s}")

        A = data["/sync-A/stored"][...]
        idx0 = data[f"/sync-A/plastic/{np.min(A):d}/idx"][...]

        for a in A:

            idx = data[f"/sync-A/plastic/{a:d}/idx"][...]
            cracked = (idx != idx0).astype(np.int)
            renum = renumber(np.argwhere(cracked).ravel(), nx)
            count[a, :] += cracked[renum]
            norm[a] += 1

# ==================================================================================================
# save data
# ==================================================================================================

with h5py.File(output, "w") as data:

    data["/A"] = np.arange(nx + 1)
    data["/P"] = count / np.where(norm > 0, norm, 1).reshape(-1, 1)
    data["/norm"] = norm

    data["/A"].attrs[
        "desc"
    ] = "Avalanche extension A at which /P is stored == np.arange(N + 1)"
    data["/P"].attrs[
        "desc"
    ] = "Probability that a block yielded: realisations are centered before averaging"
    data["/P"].attrs["shape"] = "[N + 1, N] or [A, x]"
    data["/norm"].attrs["desc"] = "Number of measurements per A"
