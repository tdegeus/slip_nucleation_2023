r'''
Collected data at synchronised avalanche area `A`,
for the local response (at the individual "element" level).

Usage:
  collect_sync-A_element.py [options] <files>...

Arguments:
  <files>   Files from which to collect data.

Options:
  -o, --output=<N>  Output file. [default: output.hdf5]
  -x, --xdmf=<N>    Extension of XDMF file: basename is "output" option. [default: xdmf]
  -i, --info=<N>    Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
'''

import os
import sys
import docopt
import click
import h5py
import numpy as np
import GooseFEM as gf
import GooseFEM.ParaView.HDF5 as pv

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

files = args['<files>']
info = args['--info']
output = args['--output']
output_xdmf = os.path.splitext(output)[0] + args['--xdmf']

for file in files + [info]:
  if not os.path.isfile(file):
    raise IOError('"{0:s}" does not exist'.format(file))

for file in [output, output_xdmf]:
  if os.path.isfile(file):
    print('"{0:s}" exists'.format(file))
    if not click.confirm('Proceed?'):
      sys.exit(1)

# ==================================================================================================
# get constants
# ==================================================================================================

with h5py.File(files[0], 'r') as data:
  plastic = data['/meta/plastic'][...]
  nx      = len(plastic)
  h       = np.pi

# ==================================================================================================
# get normalisation
# ==================================================================================================

with h5py.File(info, 'r') as data:
  sig0 = data['/normalisation/sig0'][...]

# ==================================================================================================
# get mapping
# ==================================================================================================

mesh = gf.Mesh.Quad4.FineLayer(nx, nx, h)

assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

if nx % 2 == 0: mid =  nx      / 2
else          : mid = (nx - 1) / 2

mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)

regular = mapping.getRegularMesh()

coor  = regular.coor()
conn  = regular.conn()
elmat = regular.elementMatrix()

# ==================================================================================================
# ensemble average
# ==================================================================================================

# crack sizes to read
A_read = np.arange(nx+1)[::10]

# open XDMF-file with metadata that allow ParaView to interpret the HDF5-file
xdmf = pv.TimeSeries()

# open the output HDF5-file
with h5py.File(output, 'w') as out:

  # write mesh
  out['/coor'] = coor
  out['/conn'] = conn

  # loop over cracks
  for a in A_read:

    # initialise average
    Sig_xx = np.zeros(regular.nelem())
    Sig_xy = np.zeros(regular.nelem())
    Sig_yy = np.zeros(regular.nelem())

    # normalisation
    norm = 0

    # print progress
    print('A = ', a)

    # loop over files
    for file in files:

      # open data file
      with h5py.File(file, 'r') as data:

        # get stored "A"
        A = data["/sync-A/stored"][...]

        # skip file if "a" is not stored
        if a not in A:
          continue

        # get the reference configuration
        idx0  = data['/sync-A/plastic/{0:d}/idx' .format(np.min(A))][...]
        epsp0 = data['/sync-A/plastic/{0:d}/epsp'.format(np.min(A))][...]

        # read data
        sig_xx = data["/sync-A/element/{0:d}/sig_xx".format(a)][...]
        sig_xy = data["/sync-A/element/{0:d}/sig_xy".format(a)][...]
        sig_yy = data["/sync-A/element/{0:d}/sig_yy".format(a)][...]

        # get current configuration
        idx  = data['/sync-A/plastic/{0:d}/idx' .format(a)][...]
        epsp = data['/sync-A/plastic/{0:d}/epsp'.format(a)][...]
        x    = data['/sync-A/plastic/{0:d}/x'   .format(a)][...]

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

    # hydrostatic stress
    sig_m = (sig_xx + sig_yy) / 2.

    # deviatoric stress
    sigd_xx = sig_xx - sig_m
    sigd_xy = sig_xy
    sigd_yy = sig_yy - sig_m

    # equivalent stress
    sig_eq = np.sqrt(2.0 * (sigd_xx**2.0 + sigd_yy**2.0 + 2.0 * sigd_xy**2.0))

    # write equivalent stress
    dataset_eq = '/sig_eq/' + str(a)
    out[dataset_eq] = sig_eq / sig0

    # write hydrostatic stress
    dataset_m = '/sig_m/' + str(a)
    out[dataset_m] = sig_m / sig0

    # add to metadata
    # - initialise Increment
    xdmf_inc = pv.Increment(
      pv.Connectivity(out.filename, "/conn", pv.ElementType.Quadrilateral, conn.shape),
      pv.Coordinates (out.filename, "/coor"                              , coor.shape),
    )
    # - add attributes to Increment
    xdmf_inc.push_back(pv.Attribute(
      out.filename, dataset_eq, "sig_eq", pv.AttributeType.Cell, out[dataset_eq].shape))
    # - add attributes to Increment
    xdmf_inc.push_back(pv.Attribute(
      out.filename, dataset_m, "sig_m", pv.AttributeType.Cell, out[dataset_m].shape))
    # - add Increment to TimeSeries
    xdmf.push_back(xdmf_inc)

# write metadata
xdmf.write(output_xdmf)
