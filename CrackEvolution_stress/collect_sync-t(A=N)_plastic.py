r"""
Collect data at synchronised time since the avalanche reached an area `A = N`,
for all "plastic" blocks along the weak layer.

Usage:
  collect_sync-t(A=N)_plastic.py [options] <files>...

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
from setuptools_scm import get_version

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

glob_shape = [len(files), 5000]
plas_shape = [len(files), 5000, nx]
glob_norm = np.zeros(glob_shape, dtype="int")
plas_norm = np.zeros(plas_shape, dtype="int")

glob_t = np.zeros(glob_shape, dtype="float")
glob_sig_xx = np.zeros(glob_shape, dtype="float")
glob_sig_yy = np.zeros(glob_shape, dtype="float")
glob_sig_xy = np.zeros(glob_shape, dtype="float")
plas_sig_xx = np.zeros(plas_shape, dtype="float")
plas_sig_yy = np.zeros(plas_shape, dtype="float")
plas_sig_xy = np.zeros(plas_shape, dtype="float")
plas_epsp = np.zeros(plas_shape, dtype="float")
plas_epspdot = np.zeros(plas_shape, dtype="float")

pbar = tqdm.tqdm(files)
nfiles = 0

dstep = (
    10  # with storage every 500 increments this corresponds roughly to dA = 50 during nucleation
)
astep = 50
imax = 0

# ---------------
# loop over files
# ---------------

edx = np.empty((2, nx), dtype="int")
edx[0, :] = np.arange(nx)

for ifile, file in enumerate(pbar):

    pbar.set_description(file)

    idnum = os.path.basename(file).split("_")[0]

    with h5py.File(os.path.join(source_dir, f"{idnum:s}.hdf5"), "r") as data:
        epsy = data["/cusp/epsy"][...]
        epsy = np.hstack((-epsy[:, 0].reshape(-1, 1), epsy))
        uuid = data["/uuid"].asstr()[...]

    with h5py.File(file, "r") as data:

        assert uuid == data["/meta/uuid"].asstr()[...]

        if "/sync-t/stored" not in data:
            continue

        nfiles += 1

        A = data["/sync-A/stored"][...]
        T = data["/sync-t/global/iiter"][...]
        idx0 = data[f"/sync-A/plastic/{np.min(A):d}/idx"][...]

        edx[1, :] = idx0
        i = np.ravel_multi_index(edx, epsy.shape)
        epsy_l = epsy.flat[i]
        epsy_r = epsy.flat[i + 1]
        epsp0 = 0.5 * (epsy_l + epsy_r)

        g_sig_xx = data["/sync-t/global/sig_xx"][...]
        g_sig_xy = data["/sync-t/global/sig_xy"][...]
        g_sig_yy = data["/sync-t/global/sig_yy"][...]

        stored = data["/sync-t/stored"][...]
        istore = -1

        for i in stored:

            idx = data[f"/sync-t/plastic/{i:d}/idx"][...]
            t = T[i]
            a = np.sum(idx0 != idx)

            if a < nx:
                continue

            if np.any(idx >= epsy.shape[1] - 1):
                continue

            edx[1, :] = idx
            j = np.ravel_multi_index(edx, epsy.shape)
            epsy_l = epsy.flat[j]
            epsy_r = epsy.flat[j + 1]
            epsp = 0.5 * (epsy_l + epsy_r)

            if "element" in data["sync-t"]:
                sig_xx = data[f"/sync-t/element/{i:d}/sig_xx"][...][plastic]
                sig_xy = data[f"/sync-t/element/{i:d}/sig_xy"][...][plastic]
                sig_yy = data[f"/sync-t/element/{i:d}/sig_yy"][...][plastic]
            else:
                sig_xx = data[f"/sync-t/plastic/{i:d}/sig_xx"][...]
                sig_xy = data[f"/sync-t/plastic/{i:d}/sig_xy"][...]
                sig_yy = data[f"/sync-t/plastic/{i:d}/sig_yy"][...]

            if i < dstep:

                j = np.argmin(np.abs(A - (a - dstep)))
                idx_n = data[f"/sync-A/plastic/{A[j]:d}/idx"][...]
                t_n = data["/sync-A/global/iiter"][A[j]]

            else:

                i_n = int(i - dstep)
                idx_n = data[f"/sync-t/plastic/{i_n:d}/idx"][...]
                t_n = T[i_n]

            edx[1, :] = idx_n
            j = np.ravel_multi_index(edx, epsy.shape)
            epsy_l = epsy.flat[j]
            epsy_r = epsy.flat[j + 1]
            epsp_n = 0.5 * (epsy_l + epsy_r)

            istore += 1
            imax = max(imax, istore)

            glob_norm[ifile, istore] = 1
            plas_norm[ifile, istore, :] = 1
            glob_t[ifile, istore] = T[int(i)]
            glob_sig_yy[ifile, istore] = g_sig_yy[int(i)]
            glob_sig_xy[ifile, istore] = g_sig_xy[int(i)]
            plas_sig_xx[ifile, istore, :] = sig_xx
            plas_sig_xy[ifile, istore, :] = sig_xy
            plas_sig_yy[ifile, istore, :] = sig_yy
            plas_epsp[ifile, istore, :] = epsp
            plas_epspdot[ifile, istore, :] = (epsp - epsp_n) / (t - t_n)

glob_norm = glob_norm[:, :imax]
plas_norm = plas_norm[:, :imax, :]
glob_t = glob_t[:, :imax] * dt / t0
glob_sig_xx = glob_sig_xx[:, :imax] / sig0
glob_sig_xy = glob_sig_xy[:, :imax] / sig0
glob_sig_yy = glob_sig_yy[:, :imax] / sig0
plas_sig_xx = plas_sig_xx[:, :imax, :] / sig0
plas_sig_xy = plas_sig_xy[:, :imax, :] / sig0
plas_sig_yy = plas_sig_yy[:, :imax, :] / sig0
plas_epsp = plas_epsp[:, :imax, :] / eps0
plas_epspdot = plas_epspdot[:, :imax, :] / eps0 / (dt / t0)

with h5py.File(output, "w") as data:

    data[f"/meta/versions/{os.path.basename(__file__):s}"] = get_version(
        root="..", relative_to=__file__
    )
    data["/glob_norm"] = np.sum(glob_norm, axis=0) / float(nfiles)
    data["/glob_norm"].attrs["nfiles"] = nfiles
    data["/plastic/r/sig_xx"] = np.average(plas_sig_xx, weights=plas_norm, axis=0)
    data["/plastic/r/sig_xy"] = np.average(plas_sig_xy, weights=plas_norm, axis=0)
    data["/plastic/r/sig_yy"] = np.average(plas_sig_yy, weights=plas_norm, axis=0)
    data["/plastic/r/epsp"] = np.average(plas_epsp, weights=plas_norm, axis=0)
    data["/plastic/r/epspdot"] = np.average(plas_epspdot, weights=plas_norm, axis=0)
    data["/plastic/sig_xx"] = np.average(plas_sig_xx, weights=plas_norm, axis=(0, 2))
    data["/plastic/sig_xy"] = np.average(plas_sig_xy, weights=plas_norm, axis=(0, 2))
    data["/plastic/sig_yy"] = np.average(plas_sig_yy, weights=plas_norm, axis=(0, 2))
    data["/plastic/epsp"] = np.average(plas_epsp, weights=plas_norm, axis=(0, 2))
    data["/plastic/epspdot"] = np.average(plas_epspdot, weights=plas_norm, axis=(0, 2))
    data["/global/t"] = np.average(glob_t, weights=glob_norm, axis=0)
    data["/global/sig_xx"] = np.average(glob_sig_xx, weights=glob_norm, axis=0)
    data["/global/sig_xy"] = np.average(glob_sig_xy, weights=glob_norm, axis=0)
    data["/global/sig_yy"] = np.average(glob_sig_yy, weights=glob_norm, axis=0)
