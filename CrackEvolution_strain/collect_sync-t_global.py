
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

dt   = float(data['/normalisation/dt'  ][...])
t0   = float(data['/normalisation/t0'  ][...])
sig0 = float(data['/normalisation/sig0'][...])

data.close()

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

  out[key]['sig_xx'] = np.zeros((5000), dtype='float')
  out[key]['sig_xy'] = np.zeros((5000), dtype='float')
  out[key]['sig_yy'] = np.zeros((5000), dtype='float')
  out[key]['iiter' ] = np.zeros((5000), dtype='uint' )

# ---------------
# loop over files
# ---------------

for file in files:

  # print progress
  print(file)

  # open data file
  source = h5py.File(file, 'r')

  # get stored "A"
  T = source["/sync-t/stored"][...]

  # normalisation
  norm[T] += 1

  # read global data
  sig_xx = source["/sync-t/global/sig_xx"][...]
  sig_xy = source["/sync-t/global/sig_xy"][...]
  sig_yy = source["/sync-t/global/sig_yy"][...]
  iiter  = source["/sync-t/global/iiter" ][...]

  # add to sum, only for stored "A"
  out['1st']['sig_xx'][T] += (sig_xx)[T]
  out['1st']['sig_xy'][T] += (sig_xy)[T]
  out['1st']['sig_yy'][T] += (sig_yy)[T]
  out['1st']['iiter' ][T] += (iiter )[T]

  # add to sum, only for stored "A"
  out['2nd']['sig_xx'][T] += (sig_xx)[T] ** 2.0
  out['2nd']['sig_xy'][T] += (sig_xy)[T] ** 2.0
  out['2nd']['sig_yy'][T] += (sig_yy)[T] ** 2.0
  out['2nd']['iiter' ][T] += (iiter )[T] ** 2

  # close file
  source.close()

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
m_sig_xx = out['1st']['sig_xx'] / norm
m_sig_xy = out['1st']['sig_xy'] / norm
m_sig_yy = out['1st']['sig_yy'] / norm
m_iiter  = out['1st']['iiter' ] / norm

# compute variance
v_sig_xx = (out['2nd']['sig_xx'] / norm - (out['1st']['sig_xx'] / norm) ** 2.0 ) * norm / (norm - 1.0)
v_sig_xy = (out['2nd']['sig_xy'] / norm - (out['1st']['sig_xy'] / norm) ** 2.0 ) * norm / (norm - 1.0)
v_sig_yy = (out['2nd']['sig_yy'] / norm - (out['1st']['sig_yy'] / norm) ** 2.0 ) * norm / (norm - 1.0)
v_iiter  = (out['2nd']['iiter' ] / norm - (out['1st']['iiter' ] / norm) ** 2.0 ) * norm / (norm - 1.0)

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

# open output file
data = h5py.File('data_sync-t_global.hdf5', 'w')

# store averages
data['/avr/iiter' ] = m_iiter  * dt / t0
data['/avr/sig_eq'] = m_sig_eq      / sig0

# store variance (crack size by definition exact)
data['/std/iiter' ] = np.sqrt(np.abs(v_iiter )) * dt / t0
data['/std/sig_eq'] = np.sqrt(np.abs(v_sig_eq))      / sig0

# close output file
data.close()

