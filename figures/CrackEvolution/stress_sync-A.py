import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *



# ==================================================================================================

if True:

  nx = 'nx=3^6x2'

  # for key in ['strain', 'stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:
  for key in ['strain']:

    fig, ax = plt.subplots()

    ax.set_xlim([0, 8000])
    ax.set_ylim([0, 0.6])

    ax.set_xlabel(r'$t c_s / h$')
    ax.set_ylabel(r'$\sigma$')

    # --

    avr, var, info = getTimeEvolutionGlobal('strain', nx)

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
       color = 'k')

    ax.plot(
      ax.get_xlim(),
      info['sig_c'] * np.ones(2),
      c  = 'b',
      ls = '--')

    # --

    avr, var, info = getTimeEvolutionPlastic('strain', nx)

    ax.plot(
      avr['t'],
      avr['sig_eq'],
       color = 'r')

    ax.plot(
      ax.get_xlim(),
      info['sig_c'] * np.ones(2),
      c  = 'b',
      ls = '--')

    # --

    plt.show()
