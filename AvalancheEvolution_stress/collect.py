
import os, subprocess, h5py
import numpy      as np
import GooseFEM   as gf

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

nx = len(data['/meta/plastic'][...])

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
# ensemble average
# ==================================================================================================

# -------------------------------------------------------------------------
# initialise normalisation, and sum of first and second statistical moments
# -------------------------------------------------------------------------

norm = np.zeros((nx+1), dtype='uint')

macro = {
  '1st': {},
  '2nd': {},
}

for key in macro:

  macro[key]['sig_xx'] = np.zeros((nx+1), dtype='float')
  macro[key]['sig_xy'] = np.zeros((nx+1), dtype='float')
  macro[key]['sig_yy'] = np.zeros((nx+1), dtype='float')
  macro[key]['iiter' ] = np.zeros((nx+1), dtype='uint' )
  macro[key]['A'     ] = np.zeros((nx+1), dtype='uint' )

micro = {
  '1st': {},
  '2nd': {},
}

for key in micro:

  micro[key]['sig_xx'] = np.zeros((nx+1, nx), dtype='float')
  micro[key]['sig_xy'] = np.zeros((nx+1, nx), dtype='float')
  micro[key]['sig_yy'] = np.zeros((nx+1, nx), dtype='float')
  micro[key]['x'     ] = np.zeros((nx+1, nx), dtype='float')
  micro[key]['depsp' ] = np.zeros((nx+1, nx), dtype='float')
  micro[key]['S'     ] = np.zeros((nx+1, nx), dtype='uint' )

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

  # read global data
  sig_xx = source["/sync-A/global/sig_xx"][...]
  sig_xy = source["/sync-A/global/sig_xy"][...]
  sig_yy = source["/sync-A/global/sig_yy"][...]
  iiter  = source["/sync-A/global/iiter" ][...]

  # add to sum, only for stored "A"
  macro['1st']['sig_xx'][A] += sig_xx
  macro['1st']['sig_xy'][A] += sig_xy
  macro['1st']['sig_yy'][A] += sig_yy
  macro['1st']['iiter' ][A] += iiter
  macro['1st']['A'     ][A] += A

  # add to sum, only for stored "A"
  macro['2nd']['sig_xx'][A] += sig_xx ** 2.0
  macro['2nd']['sig_xy'][A] += sig_xy ** 2.0
  macro['2nd']['sig_yy'][A] += sig_yy ** 2.0
  macro['2nd']['iiter' ][A] += iiter  ** 2

  # read data along the weak layer
  sig_xx = source["/sync-A/plastic/sig_xx"][...]
  sig_xy = source["/sync-A/plastic/sig_xy"][...]
  sig_yy = source["/sync-A/plastic/sig_yy"][...]
  x      = source["/sync-A/plastic/x"     ][...]
  epsp   = source["/sync-A/plastic/epsp"  ][...]
  idx    = source["/sync-A/plastic/idx"   ][...]

  # add to sum, only for stored "A"
  micro['1st']['sig_xx'][A,:] += sig_xx
  micro['1st']['sig_xy'][A,:] += sig_xy
  micro['1st']['sig_yy'][A,:] += sig_yy
  micro['1st']['x'     ][A,:] += x
  micro['1st']['depsp' ][A,:] += (epsp - epsp[0,:])
  micro['1st']['S'     ][A,:] += (idx  - idx [0,:])

  # add to sum, only for stored "A"
  micro['2nd']['sig_xx'][A,:] += (sig_xx          ) ** 2.0
  micro['2nd']['sig_xy'][A,:] += (sig_xy          ) ** 2.0
  micro['2nd']['sig_yy'][A,:] += (sig_yy          ) ** 2.0
  micro['2nd']['x'     ][A,:] += (x               ) ** 2.0
  micro['2nd']['depsp' ][A,:] += (epsp - epsp[0,:]) ** 2.0
  micro['2nd']['S'     ][A,:] += (idx  - idx [0,:]) ** 2

  # close file
  source.close()

# ---------------------------------------------
# select only measurements with sufficient data
# ---------------------------------------------

idx = np.argwhere(norm > 10).ravel()

norm = norm[idx].astype(np.float)

for key in macro:
  for field in macro[key]:
    macro[key][field] = macro[key][field][idx]

for key in micro:
  for field in micro[key]:
    micro[key][field] = micro[key][field][idx,:]

# ------------------------------
# compute averages and variances
# ------------------------------

with h5py.File('data_sync-A.hdf5', 'w') as data:

  # -----

  # compute mean
  m_sig_xx = macro['1st']['sig_xx'] / norm
  m_sig_xy = macro['1st']['sig_xy'] / norm
  m_sig_yy = macro['1st']['sig_yy'] / norm
  m_iiter  = macro['1st']['iiter' ] / norm

  # compute variance
  v_sig_xx = (macro['2nd']['sig_xx'] / norm - (macro['1st']['sig_xx'] / norm) ** 2.0) * norm / (norm - 1.0)
  v_sig_xy = (macro['2nd']['sig_xy'] / norm - (macro['1st']['sig_xy'] / norm) ** 2.0) * norm / (norm - 1.0)
  v_sig_yy = (macro['2nd']['sig_yy'] / norm - (macro['1st']['sig_yy'] / norm) ** 2.0) * norm / (norm - 1.0)
  v_iiter  = (macro['2nd']['iiter' ] / norm - (macro['1st']['iiter' ] / norm) ** 2.0) * norm / (norm - 1.0)

  # hydrostatic stress
  m_sig_m = (m_sig_xx + m_sig_yy) / 2.0

  # variance
  v_sig_m = v_sig_xx * (m_sig_xx / 2.0)**2.0 + v_sig_yy * (m_sig_yy / 2.0)**2.0

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

  # store averages
  data['/global/avr/A'     ] = (macro['1st']['A'] / norm).astype(np.int)
  data['/global/avr/iiter' ] = m_iiter  * dt / t0
  data['/global/avr/sig_m' ] = m_sig_m       / sig0
  data['/global/avr/sig_eq'] = m_sig_eq      / sig0

  # store variance (crack size by definition exact)
  data['/global/std/A'     ] = np.zeros(data['/global/avr/A'].shape)
  data['/global/std/iiter' ] = np.sqrt(np.abs(v_iiter )) * dt / t0
  data['/global/std/sig_m' ] = np.sqrt(np.abs(v_sig_m ))      / sig0
  data['/global/std/sig_eq'] = np.sqrt(np.abs(v_sig_eq))      / sig0

  # -----

  # enable broadcasting
  norm = norm.reshape(-1,1)

  # compute mean
  m_sig_xx = micro['1st']['sig_xx'] / norm
  m_sig_xy = micro['1st']['sig_xy'] / norm
  m_sig_yy = micro['1st']['sig_yy'] / norm
  m_x      = micro['1st']['x'     ] / norm
  m_depsp  = micro['1st']['depsp' ] / norm
  m_S      = micro['1st']['S'     ] / norm

  # compute variance
  v_sig_xx = (micro['2nd']['sig_xx'] / norm - (micro['1st']['sig_xx'] / norm) ** 2.0) * norm / (norm - 1.0)
  v_sig_xy = (micro['2nd']['sig_xy'] / norm - (micro['1st']['sig_xy'] / norm) ** 2.0) * norm / (norm - 1.0)
  v_sig_yy = (micro['2nd']['sig_yy'] / norm - (micro['1st']['sig_yy'] / norm) ** 2.0) * norm / (norm - 1.0)
  v_x      = (micro['2nd']['x'     ] / norm - (micro['1st']['x'     ] / norm) ** 2.0) * norm / (norm - 1.0)
  v_depsp  = (micro['2nd']['depsp' ] / norm - (micro['1st']['depsp' ] / norm) ** 2.0) * norm / (norm - 1.0)
  v_S      = (micro['2nd']['S'     ] / norm - (micro['1st']['S'     ] / norm) ** 2.0) * norm / (norm - 1.0)

  # hydrostatic stress
  m_sig_m = (m_sig_xx + m_sig_yy) / 2.0

  # variance
  v_sig_m = v_sig_xx * (m_sig_xx / 2.0)**2.0 + v_sig_yy * (m_sig_yy / 2.0)**2.0

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

  # store averages
  data['/plastic/avr/sig_m' ] = m_sig_m  / sig0
  data['/plastic/avr/sig_eq'] = m_sig_eq / sig0
  data['/plastic/avr/x'     ] = m_x      / eps0
  data['/plastic/avr/depsp' ] = m_depsp  / eps0
  data['/plastic/avr/S'     ] = m_S

  # store variance (crack size by definition exact)
  data['/plastic/std/sig_m' ] = np.sqrt(np.abs(v_sig_m )) / sig0
  data['/plastic/std/sig_eq'] = np.sqrt(np.abs(v_sig_eq)) / sig0
  data['/plastic/std/x'     ] = np.sqrt(np.abs(v_x     )) / eps0
  data['/plastic/std/depsp' ] = np.sqrt(np.abs(v_depsp )) / eps0
  data['/plastic/std/S'     ] = np.sqrt(np.abs(v_S     ))

