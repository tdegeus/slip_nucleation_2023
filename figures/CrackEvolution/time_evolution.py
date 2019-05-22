import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

# ==================================================================================================

def getTimeEvolutionGlobal(stress='strain', nx='nx=3^6x2'):

  if re.match('strain.*', stress):
    key = 'CrackEvolution_strain'
  elif re.match('stress.*', stress):
    key = 'CrackEvolution_stress'

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

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-t_global.hdf5'), 'r') as data:

    avr['t'     ] = data['/avr/iiter' ][...]
    avr['sig_eq'] = data['/avr/sig_eq'][...]
    std['t'     ] = data['/std/iiter' ][...]
    std['sig_eq'] = data['/std/sig_eq'][...]

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r') as data:

    t = data['/avr/iiter'][...]
    A = data['/avr/A'    ][...]

    info['t(A=N)'] = t[np.where(A == N)[0].ravel()[0]]

    idx = np.argsort(t)
    idx = idx[2:]

    idx = idx[:np.max(np.argwhere(t[idx] < avr['t'][0]))]

    avr['t'     ] = np.hstack((data['/avr/iiter' ][...][idx], avr['t'     ]))
    avr['sig_eq'] = np.hstack((data['/avr/sig_eq'][...][idx], avr['sig_eq']))
    std['t'     ] = np.hstack((data['/std/iiter' ][...][idx], std['t'     ]))
    std['sig_eq'] = np.hstack((data['/std/sig_eq'][...][idx], std['sig_eq']))

  return avr, std, info

# ==================================================================================================

def getTimeEvolutionPlastic(stress='strain', nx='nx=3^6x2'):

  if re.match('strain.*', stress):
    key = 'CrackEvolution_strain'
  elif re.match('stress.*', stress):
    key = 'CrackEvolution_stress'

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

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-t_global.hdf5'), 'r') as data:

    avr['t'] = data['/avr/iiter'][...]
    std['t'] = data['/std/iiter'][...]

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r') as data:

    t = data['/avr/iiter'][...]
    A = data['/avr/A'    ][...]

    info['t(A=N)'] = t[np.where(A == N)[0].ravel()[0]]

    idx = np.argsort(t)
    idx = idx[2:]

    idx = idx[:np.max(np.argwhere(t[idx] < avr['t'][0]))]

    avr['t'] = np.hstack((data['/avr/iiter' ][...][idx], avr['t']))
    std['t'] = np.hstack((data['/std/iiter' ][...][idx], std['t']))

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-t_plastic.hdf5'), 'r') as data:

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

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_plastic.hdf5'), 'r') as data:

    avr['sig_eq'] = np.hstack((data['/layer/avr/sig_eq'][...][idx], avr['sig_eq']))
    std['sig_eq'] = np.hstack((data['/layer/std/sig_eq'][...][idx], std['sig_eq']))
    avr['epsp'  ] = np.hstack((data['/layer/avr/epsp'  ][...][idx], avr['epsp'  ]))
    std['epsp'  ] = np.hstack((data['/layer/std/epsp'  ][...][idx], std['epsp'  ]))
    avr['depsp' ] = np.hstack((data['/layer/avr/depsp' ][...][idx], avr['depsp' ]))
    std['depsp' ] = np.hstack((data['/layer/std/depsp' ][...][idx], std['depsp' ]))
    avr['S'     ] = np.hstack((data['/layer/avr/S'     ][...][idx], avr['S'     ]))
    std['S'     ] = np.hstack((data['/layer/std/S'     ][...][idx], std['S'     ]))
    avr['x'     ] = np.hstack((data['/layer/avr/x'     ][...][idx], avr['x'     ]))
    std['x'     ] = np.hstack((data['/layer/std/x'     ][...][idx], std['x'     ]))

  return avr, std, info

# ==================================================================================================

def getTimeEvolutionCrack(stress='strain', nx='nx=3^6x2'):

  if re.match('strain.*', stress):
    key = 'CrackEvolution_strain'
  elif re.match('stress.*', stress):
    key = 'CrackEvolution_stress'

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

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r') as data:

    t = data['/avr/iiter'][...]

    idx = np.argsort(t)
    idx = idx[2:]

    avr['t'] = data['/avr/iiter'][...][idx]
    std['t'] = data['/std/iiter'][...][idx]

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_plastic.hdf5'), 'r') as data:

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

  return avr, std, info

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  for stress in ['strain', 'stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    fig, ax = plt.subplots()

    ax.set_xlim([0, 8000])
    ax.set_ylim([0, 0.4])

    ax.set_xlabel(r'$t \; c_s / h$')
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

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['sig_eq'],
       color = 'r',
       label = 'weak layer')

    ax.plot(
      ax.get_xlim(),
      info['sig_c'] * np.ones(2),
      c  = 'b',
      ls = '--',
      label = r'$\sigma_c$')

    ax.legend()

    plt.savefig('time_evolution/stress/{0:s}.pdf'.format(stress))

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 8000])
  ax.set_ylim([0, 0.4])

  ax.set_xlabel(r'$t \; c_s / h$')
  ax.set_ylabel(r'$\sigma$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionGlobal(stress, nx)

    ax.plot(
      avr['t'],
      avr['sig_eq'],
      **color_stress(nx, stress),
      **label_stress(stress))

    tc = info['t(A=N)']
    idx = np.where(avr['t'] > tc)[0][0]

    ax.plot(
      avr['t'][idx],
      avr['sig_eq'][idx],
      **color_stress(nx, stress),
      marker = 'o')

  ax.plot(
    ax.get_xlim(),
    info['sig_c'] * np.ones(2),
    c  = 'b',
    ls = '--')

  ax.legend(ncol=3)

  plt.savefig('time_evolution/stress_global.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 8000])
  ax.set_ylim([0, 0.4])

  ax.set_xlabel(r'$t \; c_s / h$')
  ax.set_ylabel(r'$\sigma$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['sig_eq'],
      **color_stress(nx, stress),
      **label_stress(stress))

    ax.plot(
      avr['t'][idx],
      avr['sig_eq'][idx],
      **color_stress(nx, stress),
      marker = 'o')

  ax.plot(
    ax.get_xlim(),
    info['sig_c'] * np.ones(2),
    c  = 'b',
    ls = '--')

  ax.legend(ncol=3)

  plt.savefig('time_evolution/stress_plastic.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 8000])
  ax.set_ylim([0, 400])

  ax.set_xlabel(r'$t \; c_s / h$')
  ax.set_ylabel(r'$\Delta \varepsilon_\mathrm{p}$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['depsp'],
      **color_stress(nx, stress),
      **label_stress(stress))

    tc = info['t(A=N)']
    idx = np.where(avr['t'] > tc)[0][0]

    ax.plot(
      avr['t'][idx],
      avr['depsp'][idx],
      **color_stress(nx, stress),
      marker = 'o')

  ax.legend(ncol=3)

  plt.savefig('time_evolution/depsp_plastic.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 8000])
  ax.set_ylim([0, 200])

  ax.set_xlabel(r'$t \; c_s / h$')
  ax.set_ylabel(r'$S$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['S'],
      **color_stress(nx, stress),
      **label_stress(stress))

    tc = info['t(A=N)']
    idx = np.where(avr['t'] > tc)[0][0]

    ax.plot(
      avr['t'][idx],
      avr['S'][idx],
      **color_stress(nx, stress),
      marker = 'o')

  ax.legend(ncol=3)

  plt.savefig('time_evolution/S_plastic.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, 8000])
  ax.set_ylim([0, 1.2])

  ax.set_xlabel(r'$t \; c_s / h$')
  ax.set_ylabel(r'$x_\varepsilon$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getTimeEvolutionPlastic(stress, nx)

    ax.plot(
      avr['t'],
      avr['x'],
      **color_stress(nx, stress),
      **label_stress(stress))

  ax.legend(ncol=3)

  plt.savefig('time_evolution/x_plastic.pdf')

