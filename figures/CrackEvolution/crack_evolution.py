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

  with h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r') as data:

    N     = int(  data['/normalisation/N'     ][...])
    dt    = float(data['/normalisation/dt'    ][...])
    t0    = float(data['/normalisation/t0'    ][...])
    sig0  = float(data['/normalisation/sig0'  ][...])
    sig_n = float(data['/averages/sigd_top'   ][...])
    sig_c = float(data['/averages/sigd_bottom'][...])

    info['sig_c'] = sig_c

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r') as data:

    avr['A'     ] = data['/avr/A'     ][...][2:]
    std['A'     ] = data['/std/A'     ][...][2:]
    avr['t'     ] = data['/avr/iiter' ][...][2:]
    std['t'     ] = data['/std/iiter' ][...][2:]
    avr['sig_eq'] = data['/avr/sig_eq'][...][2:]
    std['sig_eq'] = data['/std/sig_eq'][...][2:]

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

  with h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r') as data:

    N     = int(  data['/normalisation/N'     ][...])
    dt    = float(data['/normalisation/dt'    ][...])
    t0    = float(data['/normalisation/t0'    ][...])
    sig0  = float(data['/normalisation/sig0'  ][...])
    sig_n = float(data['/averages/sigd_top'   ][...])
    sig_c = float(data['/averages/sigd_bottom'][...])

    info['sig_c'] = sig_c

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r') as data:

    avr['A'] = data['/avr/A'    ][...]
    std['A'] = data['/std/A'    ][...]
    avr['t'] = data['/avr/iiter'][...]
    std['t'] = data['/std/iiter'][...]

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_plastic.hdf5'), 'r') as data:

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

  with h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r') as data:

    N     = int(  data['/normalisation/N'     ][...])
    dt    = float(data['/normalisation/dt'    ][...])
    t0    = float(data['/normalisation/t0'    ][...])
    sig0  = float(data['/normalisation/sig0'  ][...])
    sig_n = float(data['/averages/sigd_top'   ][...])
    sig_c = float(data['/averages/sigd_bottom'][...])

    info['sig_c'] = sig_c

  with h5py.File(path(key=key, nx=nx, stress=stress, fname='data_sync-A_global.hdf5'), 'r') as data:

    avr['A'] = data['/avr/A'    ][...]
    std['A'] = data['/std/A'    ][...]
    avr['t'] = data['/avr/iiter'][...]
    std['t'] = data['/std/iiter'][...]

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

    ax.set_xlim([0, num_nx(nx)])
    ax.set_ylim([0, 0.4])

    ax.set_xlabel(r'$A / h$')
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

    gplt.savefig('crack_evolution/sigxy/{0:s}.pdf'.format(stress))

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 0.4])

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$\sigma$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    ax.plot(
      avr['A'],
      avr['sig_eq'],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  ax.plot(
    ax.get_xlim(),
    info['sig_c'] * np.ones(2),
    c  = 'b',
    ls = '--')

  ax.legend()

  gplt.savefig('crack_evolution/macro_A-sigxy.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 0.4])

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$\sigma$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['sig_eq'],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

    print(stress, 'sig_c = ', np.mean(avr['sig_eq'][400:-400]))

  ax.plot(
    [avr['A'][400], avr['A'][400]],
    ax.get_ylim(),
    c  = 'k',
    ls = '-',
    lw = 1.0)

  ax.plot(
    [avr['A'][-400], avr['A'][-400]],
    ax.get_ylim(),
    c  = 'k',
    ls = '-',
    lw = 1.0)

  ax.plot(
    ax.get_xlim(),
    info['sig_c'] * np.ones(2),
    c  = 'b',
    ls = '--')

  ax.legend()

  gplt.savefig('crack_evolution/weak_A-sigxy.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 70])

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$\Delta \varepsilon_\mathrm{p}$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['depsp'],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  ax.legend()

  gplt.savefig('crack_evolution/weak_A-depsp.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 50])

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$S$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['S'],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  ax.legend()

  gplt.savefig('crack_evolution/weak_A-S.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 1.2])

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$x_\varepsilon$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionCrack(stress, nx)

    ax.plot(
      avr['A'],
      avr['x'],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  ax.legend()

  gplt.savefig('crack_evolution/weak_A-x.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 1100])

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$t c_s / h$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    ax.plot(
      avr['A'],
      avr['t'],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  ax.legend()

  gplt.savefig('crack_evolution/global_A-t.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$t c_s / h$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    Ac = read_Ac(nx, stress)

    idx = np.min(np.where(avr['A'] >= Ac)[0])

    ax.plot(
      avr['A'][:idx],
      avr['t'][:idx],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  ax.legend()

  gplt.savefig('crack_evolution/global_A-t_avalanche.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([-num_nx(nx) * 1.02, 0])
  ax.set_ylim([-1000, 0])

  dx = np.array(ax.get_xlim())

  ax.set_xlabel(r'$(A - A_\mathrm{final}) / h$')
  ax.set_ylabel(r'$(t - t_\mathrm{final}) c_s / h$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    Ac = read_Ac(nx, stress)

    idx = np.min(np.where(avr['A'] >= Ac)[0])

    if stress == 'stress=0d6': v = 0.7925641927543035
    if stress == 'stress=1d6': v = 1.2917752844519859
    if stress == 'stress=2d6': v = 1.5808473165796881
    if stress == 'stress=3d6': v = 1.7864728761051447
    if stress == 'stress=4d6': v = 1.9117088063051701
    if stress == 'stress=5d6': v = 2.0135932622332615
    if stress == 'stress=6d6': v = 2.1199393689195576

    ax.plot(dx, dx/v/2., **color_stress(nx, stress), ls='-', lw=1)

    ax.plot(
      avr['A'][idx:] - avr['A'][-1],
      avr['t'][idx:] - avr['t'][-1],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

    ax.plot(
      avr['A'][:idx] - avr['A'][-1],
      avr['t'][:idx] - avr['t'][-1],
      **alternative_color_stress(nx, stress))

    ax.plot(
      avr['A'][idx] - avr['A'][-1],
      avr['t'][idx] - avr['t'][-1],
      **alternative_color_stress(nx, stress),
      marker='o')

  ax.legend()

  gplt.savefig('crack_evolution/global_A-t_reverse.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([-num_nx(nx) * 1.02, 0])
  ax.set_ylim([-120, 20])

  dx = np.array(ax.get_xlim())

  ax.set_xlabel(r'$(A - A_\mathrm{final}) / h$')
  ax.set_ylabel(r'$(t - t_\mathrm{final}) c_s / h - (A/h) / (2 v_f/c_s)$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    Ac = read_Ac(nx, stress)

    idx = np.min(np.where(avr['A'] >= Ac)[0])

    if stress == 'stress=0d6': v = 0.7925641927543035
    if stress == 'stress=1d6': v = 1.2917752844519859
    if stress == 'stress=2d6': v = 1.5808473165796881
    if stress == 'stress=3d6': v = 1.7864728761051447
    if stress == 'stress=4d6': v = 1.9117088063051701
    if stress == 'stress=5d6': v = 2.0135932622332615
    if stress == 'stress=6d6': v = 2.1199393689195576

    ax.plot(
      avr['A'][idx:] - avr['A'][-1],
      avr['t'][idx:] - avr['t'][-1] - (avr['A'][idx:] - avr['A'][-1])/v/2.,
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

    ax.plot(
      avr['A'][:idx] - avr['A'][-1],
      avr['t'][:idx] - avr['t'][-1] - (avr['A'][:idx] - avr['A'][-1])/v/2.,
      **alternative_color_stress(nx, stress))

  ax.legend()

  gplt.savefig('crack_evolution/global_A-t_reverse_velocity.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xscale('log')
  ax.set_yscale('log')

  ax.set_xlim([1e-3, 1e0])
  ax.set_ylim([1e0, 7e2])

  ax.set_xlabel(r'$(A / h)^{-1}$')
  ax.set_ylabel(r'$t c_s / h$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    Ac = read_Ac(nx, stress)

    idx = np.min(np.where(avr['A'] >= Ac)[0])

    ax.plot(
      1. / avr['A'][:idx],
      avr['t'][:idx],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  gplt.plot_powerlaw(                -0.87, 0., 1., 1., units='relative', axis=ax, ls='--', c='k', lw=1)
  gplt.annotate_powerlaw(r'$-0.87$', -0.87, 0., 1., 1., units='relative', axis=ax, rx=.2, ry=.2)

  ax.legend()

  gplt.savefig('crack_evolution/global_A-t_avalanche_log-log.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([-500, 500])

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$t c_s / h$')

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    avr, var, info = getCrackEvolutionGlobal(stress, nx)

    idx = np.argmin(np.abs(avr['A'] - int(float(num_nx(nx))/2.)))

    A0 = avr['A'][idx]

    ax.plot(
      avr['A'],
      avr['t'] - avr['t'][idx],
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  ax.plot(ax.get_xlim(), (ax.get_xlim() - A0)/2./2., label=r'$2 c_s$', c='b', ls='--')

  ax.legend()

  gplt.savefig('crack_evolution/global_t-sync.pdf')

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlim([0, num_nx(nx)])
  ax.set_ylim([0, 2.5])

  ax.set_xlabel(r'$A / h$')
  ax.set_ylabel(r'$v_f / c_s$')

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

    if np.any((dA / dt) / 2.0 > 1.0):
      print(stress, 'A_ss = ', A[np.min(np.where((dA / dt) / 2.0 > 1.0))])

    ax.plot(
      A,
      (dA / dt) / 2.,
      **color_stress(nx, stress),
      **label_stress_minimal(stress))

  ax.legend(ncol=2)

  gplt.savefig('crack_evolution/global_velocity.pdf')

