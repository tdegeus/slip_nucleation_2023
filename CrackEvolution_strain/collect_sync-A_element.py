r"""
Collected data at synchronised avalanche area `A`,
for the local response (at the individual "element" level).

Usage:

1.  Move to the folder with the output of the C++ program.
1.  Copy the relevant `EnsembleInfo.hdf5` to this folder.
2.  Run this script using Python.
"""

import os
import subprocess
import h5py
import numpy as np
import GooseFEM as gf
import GooseFEM.ParaView.HDF5 as pv

# ==================================================================================================
# horizontal shift
# ==================================================================================================


def getRenumIndex(old, new, N):

    idx = np.tile(np.arange(N), (3))

    ii = old + N - new
    jj = old + 2 * N - new
    return idx[ii:jj]


# ==================================================================================================
# get all simulation files, split in ensembles
# ==================================================================================================

files = subprocess.check_output("find . -iname '*.hdf5'", shell=True).decode("utf-8")
files = list(filter(None, files.split("\n")))
files = [os.path.relpath(file) for file in files]
files = [file for file in files if len(file.split("id=")) > 1]

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

with h5py.File("EnsembleInfo.hdf5", "r") as data:
    sig0 = data["/normalisation/sig0"][...]

# ==================================================================================================
# get mapping
# ==================================================================================================

mesh = gf.Mesh.Quad4.FineLayer(nx, nx, h)

assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

if nx % 2 == 0:
    mid = nx / 2
else:
    mid = (nx - 1) / 2

mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)

regular = mapping.getRegularMesh()

coor = regular.coor()
conn = regular.conn()
elmat = regular.elementMatrix()

# ==================================================================================================
# ensemble average
# ==================================================================================================

# crack sizes to read
A_read = np.arange(nx + 1)[::10]

# open XDMF-file with metadata that allow ParaView to interpret the HDF5-file
xdmf = pv.TimeSeries()

# open the output HDF5-file
with h5py.File("data_sync-A_element.hdf5", "w") as out:

    # write mesh
    out["/coor"] = coor
    out["/conn"] = conn

    # loop over cracks
    for a in A_read:

        # initialise average
        Sig_xx = np.zeros(regular.nelem())
        Sig_xy = np.zeros(regular.nelem())
        Sig_yy = np.zeros(regular.nelem())

        # normalisation
        norm = 0

        # print progress
        print("A = ", a)

        # loop over files
        for file in files:

            # open data file
            with h5py.File(file, "r") as data:

                # get stored "A"
                A = data["/sync-A/stored"][...]

                # skip file if "a" is not stored
                if a not in A:
                    continue

                # get the reference configuration
                idx0 = data[f"/sync-A/plastic/{np.min(A):d}/idx"][...]
                epsp0 = data[f"/sync-A/plastic/{np.min(A):d}/epsp"][...]

                # read data
                sig_xx = data[f"/sync-A/element/{a:d}/sig_xx"][...]
                sig_xy = data[f"/sync-A/element/{a:d}/sig_xy"][...]
                sig_yy = data[f"/sync-A/element/{a:d}/sig_yy"][...]

                # get current configuration
                idx = data[f"/sync-A/plastic/{a:d}/idx"][...]
                epsp = data[f"/sync-A/plastic/{a:d}/epsp"][...]
                x = data[f"/sync-A/plastic/{a:d}/x"][...]

            # indices of blocks where yielding took place
            icell = np.argwhere(idx0 != idx).ravel()

            # shift to compute barycentre
            icell[icell > mid] -= nx

            # renumber index
            if len(icell) > 0:
                center = np.mean(icell)
                renum = getRenumIndex(int(center), 0, nx)
            else:
                renum = np.arange(nx)

            # element numbers such that the crack is aligned
            get = elmat[:, renum].ravel()

            # add to average
            Sig_xx += mapping.mapToRegular(sig_xx)[get]
            Sig_xy += mapping.mapToRegular(sig_xy)[get]
            Sig_yy += mapping.mapToRegular(sig_yy)[get]

            # update normalisation
            norm += 1

        # ensure sufficient data
        if norm < 30:
            continue

        # average
        sig_xx = Sig_xx / float(norm)
        sig_xy = Sig_xy / float(norm)
        sig_yy = Sig_yy / float(norm)

        # hydrostatic stress
        sig_m = (sig_xx + sig_yy) / 2.0

        # deviatoric stress
        sigd_xx = sig_xx - sig_m
        sigd_xy = sig_xy
        sigd_yy = sig_yy - sig_m

        # equivalent stress
        sig_eq = np.sqrt(2.0 * (sigd_xx ** 2.0 + sigd_yy ** 2.0 + 2.0 * sigd_xy ** 2.0))

        # write equivalent stress
        dataset_eq = "/sig_eq/" + str(a)
        out[dataset_eq] = sig_eq / sig0

        # write hydrostatic stress
        dataset_m = "/sig_m/" + str(a)
        out[dataset_m] = sig_m / sig0

        # add to metadata
        # - initialise Increment
        xdmf_inc = pv.Increment(
            pv.Connectivity(
                out.filename, "/conn", pv.ElementType.Quadrilateral, conn.shape
            ),
            pv.Coordinates(out.filename, "/coor", coor.shape),
        )
        # - add attributes to Increment
        xdmf_inc.push_back(
            pv.Attribute(
                out.filename,
                dataset_eq,
                "sig_eq",
                pv.AttributeType.Cell,
                out[dataset_eq].shape,
            )
        )
        # - add attributes to Increment
        xdmf_inc.push_back(
            pv.Attribute(
                out.filename,
                dataset_m,
                "sig_m",
                pv.AttributeType.Cell,
                out[dataset_m].shape,
            )
        )
        # - add Increment to TimeSeries
        xdmf.push_back(xdmf_inc)

# write metadata
xdmf.write("data_sync-A_element.xdmf")
