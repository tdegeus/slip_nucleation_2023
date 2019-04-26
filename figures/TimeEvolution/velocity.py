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

def average(nx, stress, Ac, Aref):

  # ensemble info
  # -------------

  data = h5py.File(path(nx=nx, fname='EnsembleInfo.hdf5'), 'r')

  N = int(data['/normalisation/N'][...])

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
  dt = []

  # loop over measurements
  for file in sorted([f for f in data]):

    # - check if a crack of "Aref" is reached
    if not np.any(meta[file]['A'] >= Aref):
      continue

    # - check if a crack of "Ac" is reached
    if not np.any(meta[file]['A'] >= Ac):
      continue

    # - find the increment for which the "crack" is first bigger than "Ac"
    inc     = np.argmin(np.abs(meta[file]['A'] - Ac  ))
    inc_ref = np.argmin(np.abs(meta[file]['A'] - Aref))

    # - find the time
    dt += [(meta[file]['dt'][inc] - meta[file]['dt'][inc_ref])]

  # close raw-data
  data.close()

  dt = np.array(dt)

  return np.mean(dt), np.std(dt)

# --------------------------------------------------------------------------------------------------

if not os.path.isdir('velocity'): os.makedirs('velocity')

for nx in list_nx()[::-1]:

  N = num_nx(nx)

  fig, ax = gplt.subplots(scale_x=1.2)

  for stress in list_stress()[::-1]:

    if nx == 'nx=3^6': Ac_list = np.linspace(50,  700, 15).astype(np.int)
    else             : Ac_list = np.linspace(50, 1400, 29).astype(np.int)

    if nx == 'nx=3^6': Aref = 400
    else             : Aref = 600

    dt = []
    error = []

    for i, Ac in enumerate(Ac_list):

      mean, std = average(nx, stress, Ac, Aref)

      dt += [mean]
      error += [std]

    dt = np.array(dt)
    error = np.array(error)

    x  = Ac_list - Aref
    t  = dt * cs / h
    s  = error * cs / h

    ax.errorbar(x, t * 2, yerr=s, **color_stress(nx, stress), **label_stress(stress), zorder=0)

  xpos = np.array([0, np.max(x)])

  ax.plot(x   , x    / 2         , c='b', ls='--', label=r'$2        c_s$', zorder=10)
  ax.plot(xpos, xpos / np.sqrt(2), c='b', ls='-.', label=r'$\sqrt{2} c_s$', zorder=10)
  ax.plot(xpos, xpos             , c='b', ls=':' , label=r'$         c_s$', zorder=10)

  ax.set_xlabel(r'$\Delta r / h$')
  ax.set_ylabel(r'$\Delta t c_s / h$')

  # Shrink current axis by 20%
  plt.subplots_adjust(right=0.8)

  # Put a legend to the right of the current axis
  legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.savefig('velocity/{nx:s}.pdf'.format(nx=nx))
  plt.close()

