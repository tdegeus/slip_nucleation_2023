import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

# ==================================================================================================

def getCrackEvolutionGlobal(stress='strain', nx='nx=3^6x2'):

  if re.match('strain.*', stress):
    key = 'CrackEvolution_strain'
  elif re.match('stress.*', stress):
    key = 'CrackEvolution_stress'

  avr  = {}
  std  = {}
  info = {}

  # -----

  data = h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r')

  N     = int(  data['/normalisation/N'     ][...])
  dt    = float(data['/normalisation/dt'    ][...])
  t0    = float(data['/normalisation/t0'    ][...])
  sig0  = float(data['/normalisation/sig0'  ][...])
  sig_n = float(data['/averages/sigd_top'   ][...])
  sig_c = float(data['/averages/sigd_bottom'][...])

  info['sig_c'] = sig_c

  data.close()

  # -----

  data = h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r')

  avr['A'     ] = data['/avr/A'     ][...][2:]
  std['A'     ] = data['/std/A'     ][...][2:]
  avr['t'     ] = data['/avr/iiter' ][...][2:]
  std['t'     ] = data['/std/iiter' ][...][2:]
  avr['sig_eq'] = data['/avr/sig_eq'][...][2:]
  std['sig_eq'] = data['/std/sig_eq'][...][2:]

  data.close()

  # -----

  return avr, std, info

# ==================================================================================================

def getCrackEvolutionPlastic(stress='strain', nx='nx=3^6x2'):

  if re.match('strain.*', stress):
    key = 'CrackEvolution_strain'
  elif re.match('stress.*', stress):
    key = 'CrackEvolution_stress'

  avr  = {}
  std  = {}
  info = {}

  # -----

  data = h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r')

  N     = int(  data['/normalisation/N'     ][...])
  dt    = float(data['/normalisation/dt'    ][...])
  t0    = float(data['/normalisation/t0'    ][...])
  sig0  = float(data['/normalisation/sig0'  ][...])
  sig_n = float(data['/averages/sigd_top'   ][...])
  sig_c = float(data['/averages/sigd_bottom'][...])

  info['sig_c'] = sig_c

  data.close()

  # -----

  data = h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r')

  avr['A'] = data['/avr/A'    ][...]
  std['A'] = data['/std/A'    ][...]
  avr['t'] = data['/avr/iiter'][...]
  std['t'] = data['/std/iiter'][...]

  data.close()

  # -----

  data = h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_plastic.hdf5'), 'r')

  avr['sig_eq'] = data['/layer/avr/sig_eq'][...]
  std['sig_eq'] = data['/layer/std/sig_eq'][...]
  avr['epsp'  ] = data['/layer/avr/epsp'  ][...]
  std['epsp'  ] = data['/layer/std/epsp'  ][...]
  avr['depsp' ] = data['/layer/avr/depsp' ][...]
  std['depsp' ] = data['/layer/std/depsp' ][...]
  avr['S'     ] = data['/layer/avr/S'     ][...]
  std['S'     ] = data['/layer/std/S'     ][...]
  avr['x'     ] = data['/layer/avr/x'     ][...]
  std['x'     ] = data['/layer/std/x'     ][...]

  data.close()

  # -----

  return avr, std, info

# ==================================================================================================

def getCrackEvolutionCrack(stress='strain', nx='nx=3^6x2'):

  if re.match('strain.*', stress):
    key = 'CrackEvolution_strain'
  elif re.match('stress.*', stress):
    key = 'CrackEvolution_stress'

  avr  = {}
  std  = {}
  info = {}

  # -----

  data = h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r')

  N     = int(  data['/normalisation/N'     ][...])
  dt    = float(data['/normalisation/dt'    ][...])
  t0    = float(data['/normalisation/t0'    ][...])
  sig0  = float(data['/normalisation/sig0'  ][...])
  sig_n = float(data['/averages/sigd_top'   ][...])
  sig_c = float(data['/averages/sigd_bottom'][...])

  info['sig_c'] = sig_c

  data.close()

  # -----

  data = h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r')

  avr['A'] = data['/avr/A'    ][...]
  std['A'] = data['/std/A'    ][...]
  avr['t'] = data['/avr/iiter'][...]
  std['t'] = data['/std/iiter'][...]

  data.close()

  # -----

  data = h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_plastic.hdf5'), 'r')

  avr['sig_eq'] = data['/crack/avr/sig_eq'][...]
  std['sig_eq'] = data['/crack/std/sig_eq'][...]
  avr['epsp'  ] = data['/crack/avr/epsp'  ][...]
  std['epsp'  ] = data['/crack/std/epsp'  ][...]
  avr['depsp' ] = data['/crack/avr/depsp' ][...]
  std['depsp' ] = data['/crack/std/depsp' ][...]
  avr['S'     ] = data['/crack/avr/S'     ][...]
  std['S'     ] = data['/crack/std/S'     ][...]
  avr['x'     ] = data['/crack/avr/x'     ][...]
  std['x'     ] = data['/crack/std/x'     ][...]

  data.close()

  # -----

  return avr, std, info

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  for stress in ['strain', 'stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    fig, ax = plt.subplots()

    ax.set_xlim([0, num_nx(nx)])
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

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['sig_eq'],
       color = 'r',
       label = 'crack')

    ax.plot(
      ax.get_xlim(),
      info['sig_c'] * np.ones(2),
      c  = 'b',
      ls = '--',
      label = r'$\sigma_c$')

    ax.legend()

    plt.savefig('crack_evolution/stress/{0:s}.pdf'.format(stress))

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
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

  plt.savefig('crack_evolution/stress_global.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 0.4])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$\sigma$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['sig_eq'],
      **color_stress(nx, stress),
      **label_stress(stress))

    print(stress, 'sig_c = ', np.mean(avr['sig_eq'][400:-400]))

  ax.plot(
    ax.get_xlim(),
    info['sig_c'] * np.ones(2),
    c  = 'b',
    ls = '--')

  ax.legend(ncol=3)

  plt.savefig('crack_evolution/stress_crack.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 70])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$\Delta \varepsilon_\mathrm{p}$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['depsp'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=3)

  plt.savefig('crack_evolution/depsp_crack.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 50])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$S$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['S'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=3)

  plt.savefig('crack_evolution/S_crack.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 1.2])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$x_\varepsilon$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['x'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=3)

  plt.savefig('crack_evolution/x_crack.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 1100])

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

  plt.savefig('crack_evolution/t.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 1100])

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

  plt.savefig('crack_evolution/t.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([-500, 500])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$t c_s / h$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    idx = np.argmin(np.abs(avr['A'] - int(float(num_nx(nx))/2.)))

    A0 = avr['A'][idx]

    ax.plot(
      avr['A'],
      avr['t'] - avr['t'][idx],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.plot(ax.get_xlim(), (ax.get_xlim() - A0)/2./2., label=r'$2 c_s$', c='b', ls='--')

  ax.legend(ncol=3)

  plt.savefig('crack_evolution/t-sync.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 2.5])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$v_f / c_s$')

  ax.fill_between(
    ax.get_xlim(),
    1.          * np.ones(2),
    np.sqrt(2.) * np.ones(2),
    color = 'b',
    alpha = 0.2,
    lw    = 0.0,
  )

  ax.plot(ax.get_xlim(), 1.          * np.ones(2), label=r'$c_s$', c='b', ls='--')
  ax.plot(ax.get_xlim(), np.sqrt(2.) * np.ones(2), label=r'$c_s$', c='b', ls=':')
  ax.plot(ax.get_xlim(), 2.          * np.ones(2), label=r'$c_s$', c='b', ls='-.')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    A = avr['A']
    t = avr['t']
    N = len(A)

    A = np.mean(A[:N-N%50].reshape(-1,50), axis=1)
    t = np.mean(t[:N-N%50].reshape(-1,50), axis=1)

    dA = np.diff(A)
    dt = np.diff(t)
    A  = A[0] + np.cumsum(dA)

    print(stress, 'velocity =', np.mean((dA / dt)[10:] / 2.))

    ax.plot(
      A,
      (dA / dt) / 2.,
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=4)

  plt.savefig('crack_evolution/velocity.pdf')

