
import os, subprocess, h5py
import numpy      as np
import GooseFEM   as gf

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

with h5py.File(files[0], 'r') as data:
  plastic = data['/meta/plastic'][...]
  nx      = len(plastic)
  h       = np.pi

# ==================================================================================================
# get normalisation
# ==================================================================================================

ensemble = os.path.split(os.path.dirname(os.path.abspath(files[0])))[-1].split('_stress')[0]
dbase = '../../../data'

with h5py.File(os.path.join(dbase, ensemble, 'EnsembleInfo.hdf5'), 'r') as data:
  dt   = float(data['/normalisation/dt'  ][...])
  t0   = float(data['/normalisation/t0'  ][...])
  sig0 = float(data['/normalisation/sig0'][...])
  eps0 = float(data['/normalisation/eps0'][...])

# ==================================================================================================
# get mapping
# ==================================================================================================

mesh = gf.Mesh.Quad4.FineLayer(nx, nx, h)

assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

if nx % 2 == 0: mid =  nx      / 2
else          : mid = (nx - 1) / 2

mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)

regular = mapping.getRegularMesh()

elmat = regular.elementMatrix()

# ==================================================================================================
# ensemble average
# ==================================================================================================

# -------------------------------------------------------------------------
# initialise normalisation, and sum of first and second statistical moments
# -------------------------------------------------------------------------

norm = np.zeros((5000), dtype='uint')

out = {
  '1st': {},
  '2nd': {},
}

for key in out:

  out[key]['sig_xx'] = np.zeros((5000, nx), dtype='float')
  out[key]['sig_xy'] = np.zeros((5000, nx), dtype='float')
  out[key]['sig_yy'] = np.zeros((5000, nx), dtype='float')
  out[key]['epsp'  ] = np.zeros((5000, nx), dtype='float')
  out[key]['depsp' ] = np.zeros((5000, nx), dtype='float')
  out[key]['x'     ] = np.zeros((5000, nx), dtype='float')
  out[key]['S'     ] = np.zeros((5000, nx), dtype='int'  )

# ---------------
# loop over files
# ---------------

for file in files:

  # print progress
  print(file)

  # open data file
  with h5py.File(file, 'r') as data:

    # get stored "A"
    T = source["/sync-t/stored"][...]

    # normalisation
    norm[T] += 1

    # get the reference configuration
    idx0  = data['/sync-t/plastic/{0:d}/idx' .format(np.min(T))][...]
    epsp0 = data['/sync-t/plastic/{0:d}/epsp'.format(np.min(T))][...]

    # loop over cracks
    for t in T:

      # read data
      sig_xx = data["/sync-t/element/{0:d}/sig_xx".format(t)][...][plastic]
      sig_xy = data["/sync-t/element/{0:d}/sig_xy".format(t)][...][plastic]
      sig_yy = data["/sync-t/element/{0:d}/sig_yy".format(t)][...][plastic]

      # get current configuration
      idx  = data['/sync-t/plastic/{0:d}/idx' .format(t)][...]
      epsp = data['/sync-t/plastic/{0:d}/epsp'.format(t)][...]
      x    = data['/sync-t/plastic/{0:d}/x'   .format(t)][...]

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

      # add to sum
      out['1st']['sig_xx'][t,:] += (sig_xx      )[renum]
      out['1st']['sig_xy'][t,:] += (sig_xy      )[renum]
      out['1st']['sig_yy'][t,:] += (sig_yy      )[renum]
      out['1st']['epsp'  ][t,:] += (epsp        )[renum]
      out['1st']['depsp' ][t,:] += (epsp - epsp0)[renum]
      out['1st']['x'     ][t,:] += (x           )[renum]
      out['1st']['S'     ][t,:] += (idx  - idx0 )[renum].astype(np.int)

      # add to sum
      out['2nd']['sig_xx'][t,:] += ((sig_xx      )[renum]               ) ** 2.0
      out['2nd']['sig_xy'][t,:] += ((sig_xy      )[renum]               ) ** 2.0
      out['2nd']['sig_yy'][t,:] += ((sig_yy      )[renum]               ) ** 2.0
      out['2nd']['epsp'  ][t,:] += ((epsp        )[renum]               ) ** 2.0
      out['2nd']['depsp' ][t,:] += ((epsp - epsp0)[renum]               ) ** 2.0
      out['2nd']['x'     ][t,:] += ((x           )[renum]               ) ** 2.0
      out['2nd']['S'     ][t,:] += ((idx  - idx0 )[renum].astype(np.int)) ** 2

# ---------------------------------------------
# select only measurements with sufficient data
# ---------------------------------------------

idx = np.argwhere(norm > 30).ravel()

norm = norm[idx].astype(np.float)

for key in out:
  for field in out[key]:
    out[key][field] = out[key][field][idx, :]

# ----------------------
# store support function
# ----------------------

def store(data, key,
  m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_depsp, m_x, m_S,
  v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_depsp, v_x, v_S):

  # hydrostatic stress
  m_sig_m = (m_sig_xx + m_sig_yy) / 2.

  # variance
  v_sig_m = v_sig_xx * (m_sig_xx / 2.0)**2.0 + v_sig_yy * (m_sig_yy / 2.0)**2.0

  # deviatoric stress
  m_sigd_xx = m_sig_xx - m_sig_m
  m_sigd_xy = m_sig_xy
  m_sigd_yy = m_sig_yy - m_sig_m

  # equivalent stress
  m_sig_eq = np.sqrt(2.0 * (m_sigd_xx**2.0 + m_sigd_yy**2.0 + 2.0 * m_sigd_xy**2.0))

  # correct for division
  sig_eq = np.where(m_sig_eq != 0.0, m_sig_eq, 1.0)

  # variance
  v_sig_eq = v_sig_xx * ((m_sig_xx - 0.5 * (m_sig_xx + m_sig_yy)) / sig_eq)**2.0 +\
             v_sig_yy * ((m_sig_yy - 0.5 * (m_sig_xx + m_sig_yy)) / sig_eq)**2.0 +\
             v_sig_xy * (4.0 * m_sig_xy                           / sig_eq)**2.0

  # store mean
  data['/{0:s}/avr/sig_eq'.format(key)] = m_sig_eq / sig0
  data['/{0:s}/avr/sig_m' .format(key)] = m_sig_m  / sig0
  data['/{0:s}/avr/epsp'  .format(key)] = m_epsp   / eps0
  data['/{0:s}/avr/depsp' .format(key)] = m_depsp  / eps0
  data['/{0:s}/avr/x'     .format(key)] = m_x      / eps0
  data['/{0:s}/avr/S'     .format(key)] = m_S

  # store variance
  data['/{0:s}/std/sig_eq'.format(key)] = np.sqrt(np.abs(v_sig_eq)) / sig0
  data['/{0:s}/std/sig_m' .format(key)] = np.sqrt(np.abs(v_sig_m )) / sig0
  data['/{0:s}/std/epsp'  .format(key)] = np.sqrt(np.abs(v_epsp  )) / eps0
  data['/{0:s}/std/depsp' .format(key)] = np.sqrt(np.abs(v_depsp )) / eps0
  data['/{0:s}/std/x'     .format(key)] = np.sqrt(np.abs(v_x     )) / eps0
  data['/{0:s}/std/S'     .format(key)] = np.sqrt(np.abs(v_S     ))

# -----
# store
# -----

# open output file
with h5py.File('data_sync-t_plastic.hdf5', 'w') as data:

  # ---------

  # allow broadcasting
  norm = norm.reshape((-1,1))


  # compute mean
  m_sig_xx = out['1st']['sig_xx'] / norm
  m_sig_xy = out['1st']['sig_xy'] / norm
  m_sig_yy = out['1st']['sig_yy'] / norm
  m_epsp   = out['1st']['epsp'  ] / norm
  m_depsp  = out['1st']['depsp' ] / norm
  m_x      = out['1st']['x'     ] / norm
  m_S      = out['1st']['S'     ] / norm

  # compute variance
  v_sig_xx = (out['2nd']['sig_xx'] / norm - (out['1st']['sig_xx'] / norm) ** 2.0 ) * norm / (norm - 1)
  v_sig_xy = (out['2nd']['sig_xy'] / norm - (out['1st']['sig_xy'] / norm) ** 2.0 ) * norm / (norm - 1)
  v_sig_yy = (out['2nd']['sig_yy'] / norm - (out['1st']['sig_yy'] / norm) ** 2.0 ) * norm / (norm - 1)
  v_epsp   = (out['2nd']['epsp'  ] / norm - (out['1st']['epsp'  ] / norm) ** 2.0 ) * norm / (norm - 1)
  v_depsp  = (out['2nd']['depsp' ] / norm - (out['1st']['depsp' ] / norm) ** 2.0 ) * norm / (norm - 1)
  v_x      = (out['2nd']['x'     ] / norm - (out['1st']['x'     ] / norm) ** 2.0 ) * norm / (norm - 1)
  v_S      = (out['2nd']['S'     ] / norm - (out['1st']['S'     ] / norm) ** 2.0 ) * norm / (norm - 1)

  # store
  store(data, 'element',
    m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_depsp, m_x, m_S,
    v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_depsp, v_x, v_S)

  # ---------

  # disable broadcasting
  norm = norm.ravel()

  # compute mean
  m_sig_xx = np.sum(out['1st']['sig_xx'],axis=1) / (norm*nx)
  m_sig_xy = np.sum(out['1st']['sig_xy'],axis=1) / (norm*nx)
  m_sig_yy = np.sum(out['1st']['sig_yy'],axis=1) / (norm*nx)
  m_epsp   = np.sum(out['1st']['epsp'  ],axis=1) / (norm*nx)
  m_depsp  = np.sum(out['1st']['depsp' ],axis=1) / (norm*nx)
  m_x      = np.sum(out['1st']['x'     ],axis=1) / (norm*nx)
  m_S      = np.sum(out['1st']['S'     ],axis=1) / (norm*nx)

  # compute variance
  v_sig_xx = (np.sum(out['2nd']['sig_xx'],axis=1) / (norm*nx) - (np.sum(out['1st']['sig_xx'],axis=1) / (norm*nx)) ** 2.0 ) * (norm*nx) / ((norm*nx) - 1.0)
  v_sig_xy = (np.sum(out['2nd']['sig_xy'],axis=1) / (norm*nx) - (np.sum(out['1st']['sig_xy'],axis=1) / (norm*nx)) ** 2.0 ) * (norm*nx) / ((norm*nx) - 1.0)
  v_sig_yy = (np.sum(out['2nd']['sig_yy'],axis=1) / (norm*nx) - (np.sum(out['1st']['sig_yy'],axis=1) / (norm*nx)) ** 2.0 ) * (norm*nx) / ((norm*nx) - 1.0)
  v_epsp   = (np.sum(out['2nd']['epsp'  ],axis=1) / (norm*nx) - (np.sum(out['1st']['epsp'  ],axis=1) / (norm*nx)) ** 2.0 ) * (norm*nx) / ((norm*nx) - 1.0)
  v_depsp  = (np.sum(out['2nd']['depsp' ],axis=1) / (norm*nx) - (np.sum(out['1st']['depsp' ],axis=1) / (norm*nx)) ** 2.0 ) * (norm*nx) / ((norm*nx) - 1.0)
  v_x      = (np.sum(out['2nd']['x'     ],axis=1) / (norm*nx) - (np.sum(out['1st']['x'     ],axis=1) / (norm*nx)) ** 2.0 ) * (norm*nx) / ((norm*nx) - 1.0)
  v_S      = (np.sum(out['2nd']['S'     ],axis=1) / (norm*nx) - (np.sum(out['1st']['S'     ],axis=1) / (norm*nx)) ** 2.0 ) * (norm*nx) / ((norm*nx) - 1.0)

  # store
  store(data, 'layer',
    m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_depsp, m_x, m_S,
    v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_depsp, v_x, v_S)

  # ---------
