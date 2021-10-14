r"""
Collected data at synchronised avalanche area `A`,
for the local response (at the individual "element" level).
Note that this function stores the individual stress components.

Usage:
  collect_sync-A_element-components.py [options] <files>...

Arguments:
  <files>   Files from which to collect data.

Options:
  -o, --output=<N>  Output file. [default: output.hdf5]
  -x, --xdmf=<N>    Extension of XDMF file: basename is "output" option. [default: xdmf]
  -i, --info=<N>    Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
  -f, --force       Overwrite existing output-file.
  -h, --help        Print help.
"""
import os
import sys

import click
import docopt
import GooseFEM as gf
import GooseFEM.ParaView.HDF5 as pv
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


# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args["<files>"]
info = args["--info"]
output = args["--output"]
output_xdmf = os.path.splitext(output)[0] + "." + args["--xdmf"]

for file in files + [info]:
    if not os.path.isfile(file):
        raise OSError(f'"{file:s}" does not exist')

if not args["--force"]:
    for file in [output, output_xdmf]:
        if os.path.isfile(file):
            print(f'"{file:s}" exists')
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
    sig0 = data["/normalisation/sig0"][...]

# ==================================================================================================
# get mapping
# ==================================================================================================

mesh = gf.Mesh.Quad4.FineLayer(nx, nx, h)

assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

if nx % 2 == 0:
    mid = int(nx / 2)
else:
    mid = int((nx - 1) / 2)

mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)

regular = mapping.getRegularMesh()

coor = regular.coor()
conn = regular.conn()
elmat = regular.elementMatrix()

# ==================================================================================================
# ensemble average
# ==================================================================================================

# crack sizes to read
A_read = np.arange(nx + 1)[:mid:10]

# open XDMF-file with metadata that allow ParaView to interpret the HDF5-file
xdmf = pv.TimeSeries()

# open the output HDF5-file
with h5py.File(output, "w") as out:

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

                if f"/sync-A/element/{a:d}/sig_xx" not in data:
                    continue

                # get the reference configuration
                idx0 = data[f"/sync-A/plastic/{np.min(A):d}/idx"][...]

                # read data
                sig_xx = data[f"/sync-A/element/{a:d}/sig_xx"][...]
                sig_xy = data[f"/sync-A/element/{a:d}/sig_xy"][...]
                sig_yy = data[f"/sync-A/element/{a:d}/sig_yy"][...]

                # get current configuration
                idx = data[f"/sync-A/plastic/{a:d}/idx"][...]

            # element numbers such that the crack is aligned
            renum = renumber(np.argwhere(idx0 != idx).ravel(), nx)
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

        # write equivalent stress
        dataset_xx = "/sig_xx/" + str(a)
        dataset_xy = "/sig_xy/" + str(a)
        dataset_yy = "/sig_yy/" + str(a)
        out[dataset_xx] = sig_xx / sig0
        out[dataset_xy] = sig_xy / sig0
        out[dataset_yy] = sig_yy / sig0

        # add to metadata
        # - initialise Increment
        xdmf_inc = pv.Increment(
            pv.Connectivity(out.filename, "/conn", pv.ElementType.Quadrilateral, conn.shape),
            pv.Coordinates(out.filename, "/coor", coor.shape),
        )
        # - add attributes to Increment
        xdmf_inc.push_back(
            pv.Attribute(
                out.filename,
                dataset_xx,
                "sig_xx",
                pv.AttributeType.Cell,
                out[dataset_xx].shape,
            )
        )
        xdmf_inc.push_back(
            pv.Attribute(
                out.filename,
                dataset_xy,
                "sig_xy",
                pv.AttributeType.Cell,
                out[dataset_xy].shape,
            )
        )
        xdmf_inc.push_back(
            pv.Attribute(
                out.filename,
                dataset_yy,
                "sig_yy",
                pv.AttributeType.Cell,
                out[dataset_yy].shape,
            )
        )
        # - add Increment to TimeSeries
        xdmf.push_back(xdmf_inc)

# write metadata
xdmf.write(output_xdmf)
