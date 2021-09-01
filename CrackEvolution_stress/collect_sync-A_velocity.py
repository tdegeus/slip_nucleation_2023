r"""
Collect velocities form data synchronised at avalanche area `A`

Usage:
    collect_sync-A_velocity.py [options] <files>...

Arguments:
    <files>     Files from which to collect data.

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
import GooseFEM as gf

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
# get normalisation
# ==================================================================================================

with h5py.File(info, "r") as data:
    dt = data["/normalisation/dt"][...]
    t0 = data["/normalisation/t0"][...]
    sig0 = data["/normalisation/sig0"][...]
    nx = int(data["/normalisation/N"][...])

v = np.zeros((len(files)), dtype="float")

for ifile, file in enumerate(files):

    print(f"({ifile + 1:3d}/{len(files):3d}) {file:s}")

    with h5py.File(file, "r") as data:

        A = data["/sync-A/stored"][...].astype(np.int)

        i0 = np.argmin(np.abs(A - 400))
        i1 = np.argmin(np.abs(A - 800))

        iiter = data["/sync-A/global/iiter"][...][A].astype(np.int)
        Dt = (iiter[i1] - iiter[i0]) * dt / t0
        Da = A[i1] - A[i0]

        v[ifile] = Da / Dt

with h5py.File(output, "w") as data:
    data["/v"] = v
