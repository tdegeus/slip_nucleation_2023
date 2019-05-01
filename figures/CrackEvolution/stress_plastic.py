import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

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

  for Asel in [200, 400, 600]:

    fig, ax = plt.subplots()

    ax.set_xlim([0, num_nx(nx)])
    ax.set_ylim([0, 0.5])

    ax.set_xlabel(r'$A$')
    ax.set_ylabel(r'$\sigma$')

    for stress in ['stress=0d6', 'stress=3d6', 'stress=6d6']:

      data = h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r')

      A = data['/avr/A'][...]

      data.close()

      data = h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_plastic.hdf5'), 'r')

      sig_eq = data['/element/avr/sig_eq'][...]

      data.close()

      idx = np.argwhere(A == Asel).ravel()[0]

      x = np.arange(num_nx(nx))
      y = applyFilter(sig_eq[idx,:])

      ax.plot(x, y, **color_stress(nx, stress))

    plt.savefig('stress_plastic/A={0:d}.pdf'.format(Asel))


