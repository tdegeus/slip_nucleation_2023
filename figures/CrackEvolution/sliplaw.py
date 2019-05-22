import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import matplotlib        as mpl
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

sigma_c = {
  'stress=0d6' : 0.15490721596789805,
  'stress=1d6' : 0.14998521525303932,
  'stress=2d6' : 0.14840373943914173,
  'stress=3d6' : 0.13998807437965494,
  'stress=4d6' : 0.13181176453793328,
  'stress=5d6' : 0.14326478188785668,
  'stress=6d6' : 0.11904209626971712,
}

# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  fig, ax = plt.subplots()

  ax.set_xlabel(r'$\Delta \varepsilon_\mathrm{p}$')
  ax.set_ylabel(r'$\sigma$')

  ax.set_ylim([0, 0.6])

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sliplaw.hdf5'), 'r') as data:

      avr_sig_eq = data['/avr/sig_eq'][...]
      avr_sig_m  = data['/avr/sig_m' ][...]
      avr_depsp  = data['/avr/depsp' ][...]
      std_sig_eq = data['/std/sig_eq'][...]
      std_sig_m  = data['/std/sig_m' ][...]
      std_depsp  = data['/std/depsp' ][...]

    ax.plot(avr_depsp, avr_sig_eq, **color_stress(nx, stress))

  gplt.savefig('sliplaw/sliplaw.pdf')
