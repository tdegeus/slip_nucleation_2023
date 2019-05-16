
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

dt   = float(data['/normalisation/dt'    ][...])
t0   = float(data['/normalisation/t0'    ][...])
sig0 = float(data['/normalisation/sig0'  ][...])
eps0 = float(data['/normalisation/eps0'  ][...])

data.close()

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

norm = np.zeros((nx+1), dtype='uint')

out = {
  '1st': {},
  '2nd': {},
}

for key in out:

  out[key]['sig_xx'] = np.zeros((nx+1, nx), dtype='float')
  out[key]['sig_xy'] = np.zeros((nx+1, nx), dtype='float')
  out[key]['sig_yy'] = np.zeros((nx+1, nx), dtype='float')
  out[key]['epsp'  ] = np.zeros((nx+1, nx), dtype='float')
  out[key]['depsp' ] = np.zeros((nx+1, nx), dtype='float')
  out[key]['x'     ] = np.zeros((nx+1, nx), dtype='float')
  out[key]['S'     ] = np.zeros((nx+1, nx), dtype='int'  )

# ---------------
# loop over files
# ---------------

for file in files:

  # print progress
  print(file)

  # open data file
  source = h5py.File(file, 'r')

  # get stored "A"
  A = source["/sync-A/stored"][...]

  # normalisation
  norm[A] += 1

  # get the reference configuration
  idx0  = source['/sync-A/plastic/{0:d}/idx' .format(np.min(A))][...]
  epsp0 = source['/sync-A/plastic/{0:d}/epsp'.format(np.min(A))][...]

  # loop over cracks
  for a in A:

    # read data
    sig_xx = source["/sync-A/element/{0:d}/sig_xx".format(a)][...][plastic]
    sig_xy = source["/sync-A/element/{0:d}/sig_xy".format(a)][...][plastic]
    sig_yy = source["/sync-A/element/{0:d}/sig_yy".format(a)][...][plastic]

    # get current configuration
    idx  = source['/sync-A/plastic/{0:d}/idx' .format(a)][...]
    epsp = source['/sync-A/plastic/{0:d}/epsp'.format(a)][...]
    x    = source['/sync-A/plastic/{0:d}/x'   .format(a)][...]

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
    out['1st']['sig_xx'][a,:] += (sig_xx      )[renum]
    out['1st']['sig_xy'][a,:] += (sig_xy      )[renum]
    out['1st']['sig_yy'][a,:] += (sig_yy      )[renum]
    out['1st']['epsp'  ][a,:] += (epsp        )[renum]
    out['1st']['depsp' ][a,:] += (epsp - epsp0)[renum]
    out['1st']['x'     ][a,:] += (x           )[renum]
    out['1st']['S'     ][a,:] += (idx  - idx0 )[renum].astype(np.int)

    # add to sum
    out['2nd']['sig_xx'][a,:] += ((sig_xx      )[renum]               ) ** 2.0
    out['2nd']['sig_xy'][a,:] += ((sig_xy      )[renum]               ) ** 2.0
    out['2nd']['sig_yy'][a,:] += ((sig_yy      )[renum]               ) ** 2.0
    out['2nd']['epsp'  ][a,:] += ((epsp        )[renum]               ) ** 2.0
    out['2nd']['depsp' ][a,:] += ((epsp - epsp0)[renum]               ) ** 2.0
    out['2nd']['x'     ][a,:] += ((x           )[renum]               ) ** 2.0
    out['2nd']['S'     ][a,:] += ((idx  - idx0 )[renum].astype(np.int)) ** 2

# ---------------------------------------------
# select only measurements with sufficient data
# ---------------------------------------------

idx = np.argwhere(norm > 30).ravel()

A    = np.arange(nx + 1)
norm = norm[idx].astype(np.float)
A    = A   [idx].astype(np.float)

for key in out:
  for field in out[key]:
    out[key][field] = out[key][field][idx, :]

# -----
# store
# -----

# open output file
data = h5py.File('data_sync-A_plastic.hdf5', 'w')

# allow broadcasting
norm = norm.reshape((-1,1))
A    = A   .reshape((-1,1))

# ------------------------------
# compute averages and variances
# ------------------------------

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

# hydrostatic stress
m_sig_m = (m_sig_xx + m_sig_yy) / 2.

# deviatoric stress
m_sigd_xx = m_sig_xx - m_sig_m
m_sigd_xy = m_sig_xy
m_sigd_yy = m_sig_yy - m_sig_m

# equivalent stress
m_sig_eq = np.sqrt(2.0 * (m_sigd_xx**2.0 + m_sigd_yy**2.0 + 2.0 * m_sigd_xy**2.0))

# variance
v_sig_eq = v_sig_xx * ((m_sig_xx - 0.5 * (m_sig_xx + m_sig_yy)) / m_sig_eq)**2.0 +\
           v_sig_yy * ((m_sig_yy - 0.5 * (m_sig_xx + m_sig_yy)) / m_sig_eq)**2.0 +\
           v_sig_xy * (4.0 * m_sig_xy                           / m_sig_eq)**2.0

# -----
# store
# -----

# store mean
data['/element/avr/sig_eq'] = m_sig_eq / sig0
data['/element/avr/epsp'  ] = m_epsp   / eps0
data['/element/avr/depsp' ] = m_depsp  / eps0
data['/element/avr/x'     ] = m_x      / eps0
data['/element/avr/S'     ] = m_S

# store variance
data['/element/std/sig_eq'] = np.sqrt(v_sig_eq) / sig0
data['/element/std/epsp'  ] = np.sqrt(v_epsp  ) / eps0
data['/element/std/depsp' ] = np.sqrt(v_depsp ) / eps0
data['/element/std/x'     ] = np.sqrt(v_x     ) / eps0
data['/element/std/S'     ] = np.sqrt(v_S     )

# --------------------
# disable broadcasting
# --------------------

norm = norm.ravel()
A    = A   .ravel()

# ------------------------------
# compute averages and variances
# ------------------------------

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

# hydrostatic stress
m_sig_m = (m_sig_xx + m_sig_yy) / 2.

# deviatoric stress
m_sigd_xx = m_sig_xx - m_sig_m
m_sigd_xy = m_sig_xy
m_sigd_yy = m_sig_yy - m_sig_m

# equivalent stress
m_sig_eq = np.sqrt(2.0 * (m_sigd_xx**2.0 + m_sigd_yy**2.0 + 2.0 * m_sigd_xy**2.0))

# variance
v_sig_eq = v_sig_xx * ((m_sig_xx - 0.5 * (m_sig_xx + m_sig_yy)) / m_sig_eq)**2.0 +\
           v_sig_yy * ((m_sig_yy - 0.5 * (m_sig_xx + m_sig_yy)) / m_sig_eq)**2.0 +\
           v_sig_xy * (4.0 * m_sig_xy                           / m_sig_eq)**2.0

# -----
# store
# -----

# store mean
data['/layer/avr/sig_eq'] = m_sig_eq / sig0
data['/layer/avr/epsp'  ] = m_epsp   / eps0
data['/layer/avr/depsp' ] = m_depsp  / eps0
data['/layer/avr/x'     ] = m_x      / eps0
data['/layer/avr/S'     ] = m_S

# store variance
data['/layer/std/sig_eq'] = np.sqrt(np.abs(v_sig_eq)) / sig0
data['/layer/std/epsp'  ] = np.sqrt(np.abs(v_epsp  )) / eps0
data['/layer/std/depsp' ] = np.sqrt(np.abs(v_depsp )) / eps0
data['/layer/std/x'     ] = np.sqrt(np.abs(v_x     )) / eps0
data['/layer/std/S'     ] = np.sqrt(np.abs(v_S     ))

# -------------------------
# remove data outside crack
# -------------------------

for i, a in enumerate(A):

  a = int(a)

  if a % 2 == 0:
    lwr = int(a/2)
    upr = nx - int(a/2)
  else:
    lwr = int((a+1)/2)
    upr = nx - int((a-1)/2)

  for key in out:
    for field in out[key]:
      out[key][field] = out[key][field].astype(np.float)
      out[key][field][i, lwr:upr] = 0.

A[A == 0.] = 1.

# ------------------------------
# compute averages and variances
# ------------------------------

# compute mean
m_sig_xx = np.sum(out['1st']['sig_xx'],axis=1) / (norm*A)
m_sig_xy = np.sum(out['1st']['sig_xy'],axis=1) / (norm*A)
m_sig_yy = np.sum(out['1st']['sig_yy'],axis=1) / (norm*A)
m_epsp   = np.sum(out['1st']['epsp'  ],axis=1) / (norm*A)
m_depsp  = np.sum(out['1st']['depsp' ],axis=1) / (norm*A)
m_x      = np.sum(out['1st']['x'     ],axis=1) / (norm*A)
m_S      = np.sum(out['1st']['S'     ],axis=1) / (norm*A)

# compute variance
v_sig_xx = (np.sum(out['2nd']['sig_xx'],axis=1) / (norm*A) - (np.sum(out['1st']['sig_xx'],axis=1) / (norm*A)) ** 2.0 ) * (norm*A) / ((norm*A) - 1.0)
v_sig_xy = (np.sum(out['2nd']['sig_xy'],axis=1) / (norm*A) - (np.sum(out['1st']['sig_xy'],axis=1) / (norm*A)) ** 2.0 ) * (norm*A) / ((norm*A) - 1.0)
v_sig_yy = (np.sum(out['2nd']['sig_yy'],axis=1) / (norm*A) - (np.sum(out['1st']['sig_yy'],axis=1) / (norm*A)) ** 2.0 ) * (norm*A) / ((norm*A) - 1.0)
v_epsp   = (np.sum(out['2nd']['epsp'  ],axis=1) / (norm*A) - (np.sum(out['1st']['epsp'  ],axis=1) / (norm*A)) ** 2.0 ) * (norm*A) / ((norm*A) - 1.0)
v_depsp  = (np.sum(out['2nd']['depsp' ],axis=1) / (norm*A) - (np.sum(out['1st']['depsp' ],axis=1) / (norm*A)) ** 2.0 ) * (norm*A) / ((norm*A) - 1.0)
v_x      = (np.sum(out['2nd']['x'     ],axis=1) / (norm*A) - (np.sum(out['1st']['x'     ],axis=1) / (norm*A)) ** 2.0 ) * (norm*A) / ((norm*A) - 1.0)
v_S      = (np.sum(out['2nd']['S'     ],axis=1) / (norm*A) - (np.sum(out['1st']['S'     ],axis=1) / (norm*A)) ** 2.0 ) * (norm*A) / ((norm*A) - 1.0)

# hydrostatic stress
m_sig_m = (m_sig_xx + m_sig_yy) / 2.

# deviatoric stress
m_sigd_xx = m_sig_xx - m_sig_m
m_sigd_xy = m_sig_xy
m_sigd_yy = m_sig_yy - m_sig_m

# equivalent stress
m_sig_eq = np.sqrt(2.0 * (m_sigd_xx**2.0 + m_sigd_yy**2.0 + 2.0 * m_sigd_xy**2.0))

# normalisation
sig_eq = np.array(m_sig_eq, copy=True)
sig_eq[sig_eq < 1.e-12] = 1.

# variance
v_sig_eq = v_sig_xx * ((m_sig_xx - 0.5 * (m_sig_xx + m_sig_yy)) / sig_eq)**2.0 +\
           v_sig_yy * ((m_sig_yy - 0.5 * (m_sig_xx + m_sig_yy)) / sig_eq)**2.0 +\
           v_sig_xy * (4.0 * m_sig_xy                           / sig_eq)**2.0

# -----
# store
# -----

# store mean
data['/crack/avr/sig_eq'] = m_sig_eq / sig0
data['/crack/avr/epsp'  ] = m_epsp   / eps0
data['/crack/avr/depsp' ] = m_depsp  / eps0
data['/crack/avr/x'     ] = m_x      / eps0
data['/crack/avr/S'     ] = m_S

# store variance
data['/crack/std/sig_eq'] = np.sqrt(np.abs(v_sig_eq)) / sig0
data['/crack/std/epsp'  ] = np.sqrt(np.abs(v_epsp  )) / eps0
data['/crack/std/depsp' ] = np.sqrt(np.abs(v_depsp )) / eps0
data['/crack/std/x'     ] = np.sqrt(np.abs(v_x     )) / eps0
data['/crack/std/S'     ] = np.sqrt(np.abs(v_S     ))

# -----------------
# close output file
# -----------------

data.close()

