r"""
Collected data at synchronised time `t`,
for the macroscopic (or "global") response.

Usage:
  collect_sync-t_global.py [options] <files>...

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
    nx = data["/normalisation/N"][...]
    nx = int(nx)

# ==================================================================================================
# ensemble average
# ==================================================================================================

# -------------------------------------------------------------------------
# initialise normalisation, and sum of first and second statistical moments
# -------------------------------------------------------------------------

norm = np.zeros((5000), dtype="uint")

out = {
    "1st": {},
    "2nd": {},
}

for key in out:

    out[key]["sig_xx"] = np.zeros((5000), dtype="float")
    out[key]["sig_xy"] = np.zeros((5000), dtype="float")
    out[key]["sig_yy"] = np.zeros((5000), dtype="float")
    out[key]["iiter"] = np.zeros((5000), dtype="uint")

# ---------------
# loop over files
# ---------------

for ifile, file in enumerate(files):

    print(f"({ifile + 1:3d}/{len(files):3d}) {file:s}")

    with h5py.File(file, "r") as data:

        if "/sync-t/stored" not in data:
            continue

        # get stored "A"
        T = data["/sync-t/stored"][...]

        # normalisation
        norm[T] += 1

        # read global data
        sig_xx = data["/sync-t/global/sig_xx"][...]
        sig_xy = data["/sync-t/global/sig_xy"][...]
        sig_yy = data["/sync-t/global/sig_yy"][...]
        iiter = data["/sync-t/global/iiter"][...]

    # add to sum, only for stored "A"
    out["1st"]["sig_xx"][T] += (sig_xx)[T]
    out["1st"]["sig_xy"][T] += (sig_xy)[T]
    out["1st"]["sig_yy"][T] += (sig_yy)[T]
    out["1st"]["iiter"][T] += (iiter)[T]

    # add to sum, only for stored "A"
    out["2nd"]["sig_xx"][T] += (sig_xx)[T] ** 2.0
    out["2nd"]["sig_xy"][T] += (sig_xy)[T] ** 2.0
    out["2nd"]["sig_yy"][T] += (sig_yy)[T] ** 2.0
    out["2nd"]["iiter"][T] += (iiter)[T] ** 2

# ---------------------------------------------
# select only measurements with sufficient data
# ---------------------------------------------

idx = np.argwhere(norm > 30).ravel()

norm = norm[idx].astype(np.float)

for key in out:
    for field in out[key]:
        out[key][field] = out[key][field][idx]

# ------------------------------
# compute averages and variances
# ------------------------------

# compute mean
m_sig_xx = out["1st"]["sig_xx"] / norm
m_sig_xy = out["1st"]["sig_xy"] / norm
m_sig_yy = out["1st"]["sig_yy"] / norm
m_iiter = out["1st"]["iiter"] / norm

# compute variance
v_sig_xx = (
    (out["2nd"]["sig_xx"] / norm - (out["1st"]["sig_xx"] / norm) ** 2.0)
    * norm
    / (norm - 1.0)
)
v_sig_xy = (
    (out["2nd"]["sig_xy"] / norm - (out["1st"]["sig_xy"] / norm) ** 2.0)
    * norm
    / (norm - 1.0)
)
v_sig_yy = (
    (out["2nd"]["sig_yy"] / norm - (out["1st"]["sig_yy"] / norm) ** 2.0)
    * norm
    / (norm - 1.0)
)
v_iiter = (
    (out["2nd"]["iiter"] / norm - (out["1st"]["iiter"] / norm) ** 2.0)
    * norm
    / (norm - 1.0)
)

# hydrostatic stress
m_sig_m = (m_sig_xx + m_sig_yy) / 2.0

# variance
v_sig_m = v_sig_xx * (m_sig_xx / 2.0) ** 2.0 + v_sig_yy * (m_sig_yy / 2.0) ** 2.0

# deviatoric stress
m_sigd_xx = m_sig_xx - m_sig_m
m_sigd_xy = m_sig_xy
m_sigd_yy = m_sig_yy - m_sig_m

# equivalent stress
m_sig_eq = np.sqrt(2.0 * (m_sigd_xx ** 2.0 + m_sigd_yy ** 2.0 + 2.0 * m_sigd_xy ** 2.0))

# variance
v_sig_eq = (
    v_sig_xx * ((m_sig_xx - 0.5 * (m_sig_xx + m_sig_yy)) / m_sig_eq) ** 2.0
    + v_sig_yy * ((m_sig_yy - 0.5 * (m_sig_xx + m_sig_yy)) / m_sig_eq) ** 2.0
    + v_sig_xy * (4.0 * m_sig_xy / m_sig_eq) ** 2.0
)

# -----
# store
# -----

# open output file
with h5py.File(output, "w") as data:

    # store averages
    data["/avr/iiter"] = m_iiter * dt / t0
    data["/avr/sig_eq"] = m_sig_eq / sig0
    data["/avr/sig_m"] = m_sig_m / sig0

    # store variance (crack size by definition exact)
    data["/std/iiter"] = np.sqrt(np.abs(v_iiter)) * dt / t0
    data["/std/sig_eq"] = np.sqrt(np.abs(v_sig_eq)) / sig0
    data["/std/sig_m"] = np.sqrt(np.abs(v_sig_m)) / sig0
