import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# --------------------------------------------------------------------------------------------------

from definitions import *

# --------------------------------------------------------------------------------------------------

def hdf2dict(data):

  out = {}

  for file in data:

    out[file] = {}

    for field in data[file]:

      out[file][field] = data[file][field][...]

  return out

# --------------------------------------------------------------------------------------------------

def getRenumIndex(old, new, N, center=False):

  idx = np.tile(np.arange(N), (3))

  shift = 0

  if center:
    shift = int(new / 2.)

  return idx[old+N-(new-shift) : old+2*N-(new-shift)]

# --------------------------------------------------------------------------------------------------

def average(nx, stress, Ac):

  # ensemble info
  # -------------

  data = h5py.File(path(nx=nx, fname='EnsembleInfo.hdf5'), 'r')

  N     = int(data['/normalisation/N'][...])
  sig0  = float(data['/normalisation/sigy'][...])
  sig_n = float(data['/averages/sigd_top'   ][...])
  sig_c = float(data['/averages/sigd_bottom'][...])

  data.close()

  # meta-data
  # ---------

  data = h5py.File(os.path.join('meta-data', '{nx:s}_{stress:s}.hdf5'.format(nx=nx, stress=stress)), 'r')

  meta = hdf2dict(data)

  data.close()

  # raw results & average
  # ---------------------

  # open raw-results
  data = h5py.File(path(nx=nx, fname='TimeEvolution_{stress:s}.hdf5'.format(stress=stress)), 'r')

  # allocate averages
  Sig_xx = np.zeros((5, N)) # middle layer + two layers above and below
  Sig_xy = np.zeros((5, N)) # middle layer + two layers above and below
  Sig_yy = np.zeros((5, N)) # middle layer + two layers above and below
  S      = np.zeros((   N)) # middle layer
  norm   = 0.

  # loop over measurements
  for file in sorted([f for f in data]):

    # - check if a crack of "Ac" is reached
    if not np.any(meta[file]['A'] >= Ac):
      continue

    # - find the increment for which the "crack" is first bigger than "Ac"
    inc = np.argmin(np.abs(meta[file]['A'] - Ac))

    # - read yield-index and stress tensor (of the middle layer + two layers above and below)
    idx0 = data[file]['idx']['0'     ][...]
    idx  = data[file]['idx'][str(inc)][...]
    Sig  = data[file]['Sig'][str(inc)][...]

    # - decompose stress
    sig_xx = Sig[:, 0, 0].reshape(-1, N)
    sig_xy = Sig[:, 0, 1].reshape(-1, N)
    sig_yy = Sig[:, 1, 1].reshape(-1, N)

    # - read left- and right-most-block containing the crack
    l = meta[file]['LEFT' ][inc]
    r = meta[file]['RIGHT'][inc]

    # - make sure that left is below right
    if l > r: l -= N

    # - take corrected right as the average
    r = int(l + (r - l) / 2. + Ac / 2.)

    # - renumber such that the first "Ac" items are in the crack
    renum = getRenumIndex(r, Ac, N, center=True)

    # - align stress to overlapping right side of the crack
    sig_xx = sig_xx[:, renum]
    sig_xy = sig_xy[:, renum]
    sig_yy = sig_yy[:, renum]

    # - align number of times yielded to overlapping right side of the crack
    idx  = idx .astype(np.int)
    idx0 = idx0.astype(np.int)
    s    = (idx - idx0)[renum]

    # - add stress to ensemble average
    Sig_xx += sig_xx
    Sig_xy += sig_xy
    Sig_yy += sig_yy

    # - add number of times yielded to ensemble average
    S += s

    # - update number of measurement in ensemble average
    norm += 1.

  # close raw-data
  data.close()

  # normalise stress tensor
  Sig_xx /= norm
  Sig_xy /= norm
  Sig_yy /= norm

  # normalise number of times yielded
  S /= norm

  # hydrostatic stress
  Sig_m = (Sig_xx + Sig_yy) / 2.

  # deviatoric stress
  Sigd_xx = Sig_xx - Sig_m
  Sigd_xy = Sig_xy
  Sigd_yy = Sig_yy - Sig_m

  # equivalent stress
  Sig_eq = np.sqrt(2.0 * (Sigd_xx**2.0 + Sigd_yy**2.0 + 2.0 * Sigd_xy**2.0))

  # normalise stress
  Sig_eq /= sig0
  Sig_xx /= (sig0 * np.sqrt(2.))
  Sig_xy /= (sig0 * np.sqrt(2.))
  Sig_yy /= (sig0 * np.sqrt(2.))

  # output
  return {
    'Sig_eq' : Sig_eq,
    'Sig_xx' : Sig_xx,
    'Sig_xy' : Sig_xy,
    'Sig_yy' : Sig_yy,
    'S'      : S,
    'sig_c'  : sig_c,
    'sig_n'  : sig_n,
  }

# --------------------------------------------------------------------------------------------------

def applyFilter(xn):

  from scipy import signal

  # Size of the measurement
  N = xn.size

  # Periodic repetitions
  xn = np.tile(xn, (3))

  # Create an order 3 lowpass butterworth filter:
  b, a = signal.butter(3, 0.05)

  # Apply the filter to xn. Use lfilter_zi to choose the initial condition of the filter:
  zi   = signal.lfilter_zi(b, a)
  z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

  # Apply the filter again, to have a result filtered at an order the same as filtfilt:
  z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

  # Use filtfilt to apply the filter:
  y = signal.filtfilt(b, a, xn)

  # Select middle of the periodic repetitions
  y = y[N: 2*N]

  # Return data
  return y

# --------------------------------------------------------------------------------------------------

if not os.path.isdir('size'): os.makedirs('size')

for nx in list_nx()[::-1]:

  N = num_nx(nx)

  for stress in list_stress()[::-1]:

    fig, ax = plt.subplots()

    ax.set_xlim([0, N])
    ax.set_ylim([0, 50])

    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$S$')

    if nx == 'nx=3^6': Ac_list = np.linspace(100,  500, 10).astype(np.int)
    else             : Ac_list = np.linspace(100, 1000, 20).astype(np.int)

    cmap = plt.get_cmap('jet', len(Ac_list))

    for i, Ac in enumerate(Ac_list):

      data = average(nx, stress, Ac)

      x = np.arange(N)
      y = applyFilter(data['S'])

      ax.plot(x, y, c=cmap(i))

    plt.savefig('size/{nx:s}_{stress:s}.pdf'.format(nx=nx, stress=stress))
    plt.close()
