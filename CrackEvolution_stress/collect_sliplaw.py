
import os, subprocess, h5py
import numpy as np
import GooseMPL as gplt

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
  dt   = data['/normalisation/dt'  ][...]
  t0   = data['/normalisation/t0'  ][...]
  sig0 = data['/normalisation/sig0'][...]
  eps0 = data['/normalisation/eps0'][...]

# ==================================================================================================
# ensemble average
# ==================================================================================================

def histo(sig_xx, sig_xy, sig_yy, depsp, bins):

  # make sure that input are NumPy-arrays
  sig_xx = np.array(sig_xx)
  sig_xy = np.array(sig_xy)
  sig_yy = np.array(sig_yy)
  depsp  = np.array(depsp )

  # bin-edges
  bin_edges = gplt.histogram_bin_edges(
    data = depsp,
    bins = bins,
    mode = 'equal',
    min_count = 100,
    remove_empty_edges = True)

  # sort data
  isort = np.argsort(depsp)
  sorted_data = depsp[isort]

  # initialise bins: average and standard deviation per bin
  out = {key:{field:[] for field in ['sig_eq','sig_m','depsp']} for key in ['avr', 'std']}

  # fill bins
  for a, b in zip(bin_edges[:-1], bin_edges[1:]):

    # get indices in sorted data
    i = np.amin(np.argwhere(sorted_data >= a))
    j = np.amax(np.argwhere(sorted_data <= b))

    # bin averages and variance
    m_sig_xx = np.mean(sig_xx[isort[i:j]])
    m_sig_yy = np.mean(sig_yy[isort[i:j]])
    m_sig_xy = np.mean(sig_xy[isort[i:j]])
    m_depsp  = np.mean(depsp [isort[i:j]])
    v_sig_xx = np.var (sig_xx[isort[i:j]])
    v_sig_yy = np.var (sig_yy[isort[i:j]])
    v_sig_xy = np.var (sig_xy[isort[i:j]])
    v_depsp  = np.var (depsp [isort[i:j]])

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

    # store average and standard deviation
    out['avr']['sig_eq'] += [m_sig_eq]
    out['avr']['sig_m' ] += [m_sig_m ]
    out['avr']['depsp' ] += [m_depsp ]
    out['std']['sig_eq'] += [np.sqrt(v_sig_eq)]
    out['std']['sig_m' ] += [np.sqrt(v_sig_m )]
    out['std']['depsp' ] += [np.sqrt(v_depsp )]

  # convert the output to NumPy-arrays
  for key in out:
    for field in out[key]:
      out[key][field] = np.array(out[key][field])

  # return output
  return out

# --------------------------------------------------------------------------------------------------

# ----------
# initialise
# ----------

Sig_xx = []
Sig_xy = []
Sig_yy = []
Depsp  = []

# ---------------
# loop over files
# ---------------

for file in files:

  # print progress
  print(file)

  # open data file
  with h5py.File(file, 'r') as data:

    # get stored "A"
    A = data["/sync-A/stored"][...]
    A = A[np.argwhere(A < 500).ravel()]

    # get the reference configuration
    epsp0 = data['/sync-A/plastic/{0:d}/epsp'.format(np.min(A))][...]

    # loop over cracks
    for a in A:

      # read data
      sig_xx = data["/sync-A/element/{0:d}/sig_xx".format(a)][...][plastic]
      sig_xy = data["/sync-A/element/{0:d}/sig_xy".format(a)][...][plastic]
      sig_yy = data["/sync-A/element/{0:d}/sig_yy".format(a)][...][plastic]

      # get current configuration
      epsp = data['/sync-A/plastic/{0:d}/epsp'.format(a)][...]

      # add to list
      Sig_xx += list(sig_xx)
      Sig_xy += list(sig_xy)
      Sig_yy += list(sig_yy)
      Depsp  += list(epsp - epsp0)

    # get stored "T"
    T = data["/sync-t/stored"][...][::10]

    # loop over cracks
    for t in T:

      # read data
      sig_xx = data["/sync-t/element/{0:d}/sig_xx".format(t)][...][plastic]
      sig_xy = data["/sync-t/element/{0:d}/sig_xy".format(t)][...][plastic]
      sig_yy = data["/sync-t/element/{0:d}/sig_yy".format(t)][...][plastic]

      # get current configuration
      epsp = data['/sync-t/plastic/{0:d}/epsp'.format(t)][...]

      # add to list
      Sig_xx += list(sig_xx)
      Sig_xy += list(sig_xy)
      Sig_yy += list(sig_yy)
      Depsp  += list(epsp - epsp0)

# -----
# store
# -----

with h5py.File('data_sliplaw.hdf5', 'w') as data:

  bins = 30

  out = histo(Sig_xx, Sig_xy, Sig_yy, Depsp, bins)

  data['/avr/sig_eq'] = np.array(out['avr']['sig_eq']) / sig0
  data['/avr/sig_m' ] = np.array(out['avr']['sig_m' ]) / sig0
  data['/avr/depsp' ] = np.array(out['avr']['depsp' ]) / eps0

  data['/std/sig_eq'] = np.array(out['std']['sig_eq']) / sig0
  data['/std/sig_m' ] = np.array(out['std']['sig_m' ]) / sig0
  data['/std/depsp' ] = np.array(out['std']['depsp' ]) / eps0
