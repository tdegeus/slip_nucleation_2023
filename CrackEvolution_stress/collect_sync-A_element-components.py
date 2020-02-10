r'''
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
# horizontal shift
# ==================================================================================================

def getRenumIndex(old, new, N):

  idx = np.tile(np.arange(N), (3))

  return idx[old+N-new: old+2*N-new]

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

if nx % 2 == 0: mid = int( nx      / 2)
else          : mid = int((nx - 1) / 2)

mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)

regular = mapping.getRegularMesh()

coor  = regular.coor()
conn  = regular.conn()
elmat = regular.elementMatrix()

# ==================================================================================================
# ensemble average
# ==================================================================================================

# crack sizes to read
A_read = np.arange(nx+1)[:mid:10]

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
        idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(A))][...]

        # read data
        sig_xx = data["/sync-A/element/{0:d}/sig_xx".format(a)][...]
        sig_xy = data["/sync-A/element/{0:d}/sig_xy".format(a)][...]
        sig_yy = data["/sync-A/element/{0:d}/sig_yy".format(a)][...]

        # get current configuration
        idx = data['/sync-A/plastic/{0:d}/idx'.format(a)][...]

      # indices of blocks where yielding took place
      icell = np.argwhere(idx0 != idx).ravel()

      # shift to compute barycentre
      icell[icell > mid] -= nx

      # renumber index
      if len(icell) > 0:
        center = np.mean(icell)
        renum  = getRenumIndex(int(center), 0, nx)
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

    # write equivalent stress
    dataset_xx = '/sig_xx/' + str(a)
    dataset_xy = '/sig_xy/' + str(a)
    dataset_yy = '/sig_yy/' + str(a)
    out[dataset_xx] = sig_xx / sig0
    out[dataset_xy] = sig_xy / sig0
    out[dataset_yy] = sig_yy / sig0

    # add to metadata
    # - initialise Increment
    xdmf_inc = pv.Increment(
      pv.Connectivity(out.filename, "/conn", pv.ElementType.Quadrilateral, conn.shape),
      pv.Coordinates (out.filename, "/coor"                              , coor.shape),
    )
    # - add attributes to Increment
    xdmf_inc.push_back(pv.Attribute(out.filename, dataset_xx, "sig_xx", pv.AttributeType.Cell, out[dataset_xx].shape))
    xdmf_inc.push_back(pv.Attribute(out.filename, dataset_xy, "sig_xy", pv.AttributeType.Cell, out[dataset_xy].shape))
    xdmf_inc.push_back(pv.Attribute(out.filename, dataset_yy, "sig_yy", pv.AttributeType.Cell, out[dataset_yy].shape))
    # - add Increment to TimeSeries
    xdmf.push_back(xdmf_inc)

# write metadata
xdmf.write(output_xdmf)
