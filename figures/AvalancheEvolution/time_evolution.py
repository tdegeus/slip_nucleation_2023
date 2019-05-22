import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

# ==================================================================================================

def getTimeEvolutionGlobal(stress, nx='nx=3^6x2'):

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

    avr['A'     ] = data['/global/avr/A'     ][...]
    avr['t'     ] = data['/global/avr/iiter' ][...]
    avr['sig_eq'] = data['/global/avr/sig_eq'][...]
    std['A'     ] = data['/global/std/A'     ][...]
    std['t'     ] = data['/global/std/iiter' ][...]
    std['sig_eq'] = data['/global/std/sig_eq'][...]

  return avr, std, info

# ==================================================================================================

def getTimeEvolutionPlastic(stress, nx='nx=3^6x2'):

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

    avr['A'] = data['/global/avr/A'     ][...]
    avr['t'] = data['/global/avr/iiter' ][...]
    std['A'] = data['/global/std/A'     ][...]
    std['t'] = data['/global/std/iiter' ][...]

  with h5py.File(path(key='AvalancheEvolution_stress', nx=nx, stress=stress, fname='data_sync-A.hdf5'), 'r') as data:

    avr['sig_eq'] = np.mean(data['/plastic/avr/sig_eq'][...], axis=1)
    std['sig_eq'] = np.mean(data['/plastic/std/sig_eq'][...], axis=1)
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

    ax.set_xlim([0, 200])
    ax.set_ylim([0, 0.4])

    ax.set_xlabel(r'$t c_s / h$')
    ax.set_ylabel(r'$\sigma$')

    avr, var, info = getTimeEvolutionGlobal(stress, nx)

    ax.fill_between(
      avr['t'],
      avr['sig_eq'] + var['sig_eq'],
      avr['sig_eq'] - var['sig_eq'],
      color = 'k',
      alpha = 0.5,
      lw    = 0.0)

    ax.plot(
      avr['t'],
      avr['sig_eq'],
       color = 'k',
       label = 'macroscopic')

    ax.plot(
      ax.get_xlim(),
      info['sig_c'] * np.ones(2),
      c  = 'b',
      ls = '--')

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['sig_eq'],
      color = 'r',
      label = 'weak layer')

    ax.legend()

    gplt.savefig('time_evolution/stress/{0:s}.pdf'.format(stress))

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 200])
  ax.set_ylim([0, 0.4])

  ax.set_xlabel(r'$t c_s / h$')
  ax.set_ylabel(r'$\sigma$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionGlobal(stress, nx)

    ax.plot(
      avr['t'],
      avr['sig_eq'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.plot(
    ax.get_xlim(),
    info['sig_c'] * np.ones(2),
    c  = 'b',
    ls = '--')

  ax.legend(ncol=3)

  gplt.savefig('time_evolution/stress_global.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 200])
  ax.set_ylim([0, 0.4])

  ax.set_xlabel(r'$t c_s / h$')
  ax.set_ylabel(r'$\sigma$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['sig_eq'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.plot(
    ax.get_xlim(),
    info['sig_c'] * np.ones(2),
    c  = 'b',
    ls = '--')

  ax.legend(ncol=3)

  gplt.savefig('time_evolution/stress_plastic.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 200])
  ax.set_ylim([0, 400])

  ax.set_xlabel(r'$t c_s / h$')
  ax.set_ylabel(r'$\Delta \varepsilon_\mathrm{p}$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['depsp'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=3)

  gplt.savefig('time_evolution/depsp_plastic.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 200])
  ax.set_ylim([0, 200])

  ax.set_xlabel(r'$t c_s / h$')
  ax.set_ylabel(r'$S$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['S'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=3)

  gplt.savefig('time_evolution/S_plastic.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 200])
  ax.set_ylim([0, 1.2])

  ax.set_xlabel(r'$t c_s / h$')
  ax.set_ylabel(r'$x_\varepsilon$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['x'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=3)

  gplt.savefig('time_evolution/x_plastic.pdf')

