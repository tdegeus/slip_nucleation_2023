import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# --------------------------------------------------------------------------------------------------

from definitions import *

rho  = 1.
h    = np.pi
G    = 1.0
mu   = G / 2.
cs   = np.sqrt(mu/rho)

# --------------------------------------------------------------------------------------------------

def hdf2dict(data):

  out = {}

  for file in data:

    out[file] = {}

    for field in data[file]:

      out[file][field] = data[file][field][...]

  return out

# --------------------------------------------------------------------------------------------------

def average(nx, stress, Aref):

  # ensemble info
  # -------------

  data = h5py.File(path(nx=nx, fname='EnsembleInfo.hdf5'), 'r')

  N     = int(data['/normalisation/N'][...])
  sig0  = float(data['/normalisation/sigy'][...])
  sig_n = float(data['/averages/sigd_top'   ][...])
  sig_c = float(data['/averages/sigd_bottom'][...])

  data.close()

  # raw results & average
  # ---------------------

  # open raw-results
  data = h5py.File(path(nx=nx, fname='TimeEvolutionFull_collect-stress_{stress:s}.hdf5'.format(stress=stress)), 'r')

  # allocate sizes
  inc_low = 99999
  inc_hgh = 0

  # loop over measurements
  for file in sorted([f for f in data]):

    # - crack size
    A = data[file]['A'][...]

    # - check if a crack of "Aref" is reached
    if not np.any(A >= Aref):
      continue

    # - find the increment for which the "crack" is first bigger than "Aref"
    inc = np.argmin(np.abs(A - Aref))

    # - update sizes
    inc_low = min(inc_low, inc)
    inc_hgh = max(inc_hgh, A.size - inc)

  # allocate averages
  sigbar  = np.zeros((inc_hgh + inc_low + 1000))
  sigweak = np.zeros((inc_hgh + inc_low + 1000))
  Dt      = np.zeros((inc_hgh + inc_low + 1000))
  norm    = np.zeros((inc_hgh + inc_low + 1000))

  # loop over measurements
  for file in sorted([f for f in data]):

    # - crack size
    A  = data[file]['A' ][...]
    dt = data[file]['dt'][...]

    # - check if a crack of "Aref" is reached
    if not np.any(A >= Aref):
      continue

    # - find the increment for which the "crack" is first bigger than "Aref"
    inc = np.argmin(np.abs(A - Aref))

    # - lower index
    sigbar [inc-inc_low: (inc-inc_low)+A.size] += data[file]['sigbar'][...]
    sigweak[inc-inc_low: (inc-inc_low)+A.size] += data[file]['sigweak'][...]
    Dt     [inc-inc_low: (inc-inc_low)+A.size] += (dt - dt[inc])
    norm   [inc-inc_low: (inc-inc_low)+A.size] += 1

  # averages
  idx = np.sort(np.argwhere(norm > 0).ravel())
  sigbar  = sigbar [idx]
  sigweak = sigweak[idx]
  Dt      = Dt     [idx]
  norm    = norm   [idx]

  return sigbar/norm/sig0, sigweak/norm/sig0, Dt/norm, sig_c, sig_n

# --------------------------------------------------------------------------------------------------

if not os.path.isdir('stress'): os.makedirs('stress')

for nx in [list_nx()[-1]]:

  N = num_nx(nx)

  fig, ax = gplt.subplots(scale_x=1.2)

  ax.set_ylim([0.0, 0.6])
  # ax.set_xlim([0, N])

  ax.set_xlabel(r'$\Delta t c_s / h$')
  ax.set_ylabel(r'$\sigma$')

  for stress in [list_stress()[-1]]:

    if nx == 'nx=3^6': Aref = 400
    else             : Aref = 600

    sigbar, sigweak, dt, sig_c, sig_n = average(nx, stress, Aref)

    t = dt * cs / h

    ax.plot(t, sigbar , **color_stress(nx, stress), **label_stress(stress), ls='-' , zorder=0)
    ax.plot(t, sigweak, **color_stress(nx, stress)                        , ls='--', zorder=0)

  ax.plot(ax.get_xlim(), sig_c * np.ones(2), c='k', ls='--')
  ax.plot(ax.get_xlim(), sig_n * np.ones(2), c='k', ls='-.')

  # Shrink current axis by 20%
  plt.subplots_adjust(right=0.8)

  # Put a legend to the right of the current axis
  legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.savefig('stress/{nx:s}.pdf'.format(nx=nx))
  plt.close()

