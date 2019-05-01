
import os, subprocess, h5py
import numpy                  as np
import GooseFEM               as gf
import GooseFEM.ParaView.HDF5 as pv

# ==================================================================================================
# horizontal shift
# ==================================================================================================

def getRenumIndex(old, new, N):

  idx = np.tile(np.arange(N), (3))

  return idx[old+N-new: old+2*N-new]

# ==================================================================================================
# get all simulation files, split in ensembles
# ==================================================================================================

files = subprocess.check_output("find . -iname '*.hdf5'", shell=True).decode('utf-8')
files = list(filter(None, files.split('\n')))
files = [os.path.relpath(file) for file in files]
files = [file for file in files if len(file.split('id='))>1]

# ==================================================================================================
# get constants
# ==================================================================================================

data = h5py.File(files[0], 'r')

plastic = data['/meta/plastic'][...]
nx      = len(plastic)
h       = np.pi

data.close()

# ==================================================================================================
# get normalisation
# ==================================================================================================

ensemble = os.path.split(os.path.dirname(os.path.abspath(files[0])))[-1].split('_stress')[0]
dbase = '../../../data'

data = h5py.File(os.path.join(dbase, ensemble, 'EnsembleInfo.hdf5'), 'r')

sig0 = data['/normalisation/sigy'][...]

data.close()

# ==================================================================================================
# get mapping
# ==================================================================================================

mesh = gf.Mesh.Quad4.FineLayer(nx, nx, h)

assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

if nx % 2 == 0: mid =  nx      / 2
else          : mid = (nx - 1) / 2

mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)

regular = mapping.getRegular()

coor  = regular.coor()
conn  = regular.conn()
elmat = regular.elementMatrix()

# ==================================================================================================
# ensemble average
# ==================================================================================================

# get time step
# -------------

norm = np.zeros((5000), dtype='uint')

for file in files:

  source = h5py.File(file, 'r')

  T = source["/sync-t/stored"][...]

  norm[T] += 1

  data.close()

T = T[np.where(norm > 30)]

T_read = T[::10]

# read
# ----

# output
data = h5py.File('data_sync-t_element.hdf5', 'w')
xdmf = pv.TimeSeries()

# write mesh
data['/coor'] = coor
data['/conn'] = conn

# loop over cracks
for t in T_read:

  # initialise average
  Sig_xx = np.zeros(regular.nelem())
  Sig_xy = np.zeros(regular.nelem())
  Sig_yy = np.zeros(regular.nelem())

  # normalisation
  norm = 0

  # print progress
  print('T = ', t)

  # loop over files
  for file in files:

    # open data file
    source = h5py.File(file, 'r')

    # get stored "T"
    T = source["/sync-t/stored"][...]

    # skip file if "t" is not stored
    if t not in T:
      source.close()
      continue

    # get the reference configuration
    idx0  = source['/sync-t/plastic/{0:d}/idx' .format(np.min(T))][...]
    epsp0 = source['/sync-t/plastic/{0:d}/epsp'.format(np.min(T))][...]

    # read data
    sig_xx = source["/sync-t/element/{0:d}/sig_xx".format(t)][...]
    sig_xy = source["/sync-t/element/{0:d}/sig_xy".format(t)][...]
    sig_yy = source["/sync-t/element/{0:d}/sig_yy".format(t)][...]

    # get current configuration
    idx  = source['/sync-t/plastic/{0:d}/idx' .format(t)][...]
    epsp = source['/sync-t/plastic/{0:d}/epsp'.format(t)][...]
    x    = source['/sync-t/plastic/{0:d}/x'   .format(t)][...]

    # close the file
    source.close()

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
    Sig_xx += mapping.map(sig_xx)[get]
    Sig_xy += mapping.map(sig_xy)[get]
    Sig_yy += mapping.map(sig_yy)[get]

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

  # write
  dataset = '/sig_eq/' + str(t)
  data[dataset] = sig_eq / sig0

  # add to metadata
  # - initialise Increment
  xdmf_inc = pv.Increment(
    pv.Connectivity(data.filename, "/conn", pv.ElementType.Quadrilateral, conn.shape),
    pv.Coordinates (data.filename, "/coor"                              , coor.shape),
  )
  # - add attributes to Increment
  xdmf_inc.push_back(pv.Attribute(
    data.filename, dataset, "sig_eq", pv.AttributeType.Cell, data[dataset].shape))
  # - add Increment to TimeSeries
  xdmf.push_back(xdmf_inc)

# write output
xdmf.write('data_sync-t_element.xdmf')
