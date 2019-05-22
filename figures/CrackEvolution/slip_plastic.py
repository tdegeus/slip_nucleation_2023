import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import matplotlib        as mpl
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

# ==================================================================================================

A_ss = {
  'stress=1d6' : 276.5,
  'stress=2d6' : 226.5,
  'stress=3d6' : 176.5,
  'stress=4d6' : 126.5,
  'stress=5d6' : 126.5,
  'stress=6d6' : 77.5,
}

# ==================================================================================================

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

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    fig, ax = plt.subplots()

    ax.set_xlim([0, num_nx(nx)])
    ax.set_ylim([0, 1])

    ax.set_xlabel(r'$x / h$')
    ax.set_ylabel(r'$\dot{\varepsilon}_\mathrm{p} \; h / c_s$')

    cmap = plt.get_cmap('jet', 20)

    Asel = np.linspace(25, 500, 20)

    for i in range(1, len(Asel)):

      with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r') as data:
        A = data['/avr/A'][...]
        t = data['/avr/iiter'][...]

      with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_plastic.hdf5'), 'r') as data:
        depsp = data['/element/avr/depsp'][...]

      idx = np.argwhere(A == Asel[i-1]).ravel()[0]
      jdx = np.argwhere(A == Asel[i  ]).ravel()[0]

      x = np.arange(num_nx(nx))
      y = applyFilter(depsp[jdx,:] - depsp[idx,:]) / (t[jdx] - t[idx])

      ax.plot(x, y, color=cmap(i))

    gplt.savefig('slip_plastic/slip-rate/{0:s}.pdf'.format(stress))


