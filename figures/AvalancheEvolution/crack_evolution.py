import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

# ==================================================================================================

def getCrackEvolutionGlobal(stress, nx='nx=3^6x2'):

  avr  = {}
  std  = {}
  info = {}

  with h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r') as data:

    N     = int(  data['/normalisation/N'     ][...])
    dt    = float(data['/normalisation/dt'    ][...])
    t0    = float(data['/normalisation/t0'    ][...])
    sig0  = float(data['/normalisation/sig0'  ][...])
    sig_n = float(data['/averages/sigd_top'   ][...])
    sig_c = float(data['/averages/sigd_bottom'][...])

    info['sig_c'] = sig_c

  with h5py.File(path(key='AvalancheEvolution_stress', nx=nx, stress=stress, fname='data_sync-A.hdf5'), 'r') as data:

    avr['A'     ] = data['/global/avr/A'     ][...][2:]
    std['A'     ] = data['/global/std/A'     ][...][2:]
    avr['t'     ] = data['/global/avr/iiter' ][...][2:]
    std['t'     ] = data['/global/std/iiter' ][...][2:]
    avr['sig_eq'] = data['/global/avr/sig_eq'][...][2:]
    std['sig_eq'] = data['/global/std/sig_eq'][...][2:]

  return avr, std, info

# ==================================================================================================

def getCrackEvolutionPlastic(stress, nx='nx=3^6x2'):

  avr  = {}
  std  = {}
  info = {}

  with h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r') as data:

    N     = int(  data['/normalisation/N'     ][...])
    dt    = float(data['/normalisation/dt'    ][...])
    t0    = float(data['/normalisation/t0'    ][...])
    sig0  = float(data['/normalisation/sig0'  ][...])
    sig_n = float(data['/averages/sigd_top'   ][...])
    sig_c = float(data['/averages/sigd_bottom'][...])

    info['sig_c'] = sig_c

  with h5py.File(path(key='AvalancheEvolution_stress', nx=nx, stress=stress, fname='data_sync-A.hdf5'), 'r') as data:

    avr['A'] = data['/global/avr/A'    ][...]
    std['A'] = data['/global/std/A'    ][...]
    avr['t'] = data['/global/avr/iiter'][...]
    std['t'] = data['/global/std/iiter'][...]

  with h5py.File(path(key='AvalancheEvolution_stress', nx=nx, stress=stress, fname='data_sync-A.hdf5'), 'r') as data:

    avr['sig_eq'] = np.mean(data['/plastic/avr/sig_eq'][...], axis=1)
    std['sig_eq'] = np.mean(data['/plastic/std/sig_eq'][...], axis=1)
    avr['epsp'  ] = np.mean(data['/plastic/avr/epsp'  ][...], axis=1)
    std['epsp'  ] = np.mean(data['/plastic/std/epsp'  ][...], axis=1)
    avr['depsp' ] = np.mean(data['/plastic/avr/depsp' ][...], axis=1)
    std['depsp' ] = np.mean(data['/plastic/std/depsp' ][...], axis=1)
    avr['S'     ] = np.mean(data['/plastic/avr/S'     ][...], axis=1)
    std['S'     ] = np.mean(data['/plastic/std/S'     ][...], axis=1)
    avr['x'     ] = np.mean(data['/plastic/avr/x'     ][...], axis=1)
    std['x'     ] = np.mean(data['/plastic/std/x'     ][...], axis=1)

  return avr, std, info

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    fig, ax = plt.subplots()

    ax.set_xlim([0, 30])
    ax.set_ylim([0, 0.4])

    ax.set_xlabel(r'$A$')
    ax.set_ylabel(r'$\sigma$')

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    ax.fill_between(
      avr['A'],
      avr['sig_eq'] + var['sig_eq'],
      avr['sig_eq'] - var['sig_eq'],
      color = 'k',
      alpha = 0.5,
      lw    = 0.0)

    ax.plot(
      avr['A'],
      avr['sig_eq'],
       color = 'k',
       label = 'macroscopic')

    ax.plot(
      ax.get_xlim(),
      info['sig_c'] * np.ones(2),
      c  = 'b',
      ls = '--')

    ax.legend()

    gplt.savefig('crack_evolution/stress/{0:s}.pdf'.format(stress))

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 30])
  ax.set_ylim([0, 0.4])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$\sigma$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    ax.plot(
      avr['A'],
      avr['sig_eq'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.plot(
    ax.get_xlim(),
    info['sig_c'] * np.ones(2),
    c  = 'b',
    ls = '--')

  ax.legend(ncol=3)

  gplt.savefig('crack_evolution/stress_global.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 30])
  ax.set_ylim([0, 200])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$t c_s / h$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    ax.plot(
      avr['A'],
      avr['t'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=3)

  gplt.savefig('crack_evolution/t.pdf')

