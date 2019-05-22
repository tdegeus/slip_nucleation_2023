import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

A_c = {
  'stress=0d6' : 932.3844723939063,
  'stress=1d6' : 467.3199738145631,
  'stress=2d6' : 305.8574308533365,
  'stress=3d6' : 147.71061575923957,
  'stress=4d6' : 86.65425420195837,
  'stress=5d6' : 57.32524892440282,
  'stress=6d6' : 29.4690945768259,
}

A_ss = {
  'stress=1d6' : 276.5,
  'stress=2d6' : 226.5,
  'stress=3d6' : 176.5,
  'stress=4d6' : 126.5,
  'stress=5d6' : 126.5,
  'stress=6d6' : 77.5,
}

stresses = [
  'stress=1d6',
  'stress=2d6',
  'stress=3d6',
  'stress=4d6',
  'stress=5d6',
  'stress=6d6',
]

velocity = {
  'stress=0d6' : 0.7925641927543035,
  'stress=1d6' : 1.2917752844519859,
  'stress=2d6' : 1.5808473165796881,
  'stress=3d6' : 1.7864728761051447,
  'stress=4d6' : 1.9117088063051701,
  'stress=5d6' : 2.0135932622332615,
  'stress=6d6' : 2.1199393689195576,
}

# ==================================================================================================

nx = 'nx=3^6x2'

with h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r') as data:

  sig_bot = data['/averages/sigd_bottom'][...]
  sig_top = data['/averages/sigd_top'][...]

# ==================================================================================================

a_c  = [A_c [stress] for stress in stresses]
a_ss = [A_ss[stress] for stress in stresses]
vel  = [velocity[stress] for stress in stresses]
sig  = [num_stress(stress) * (sig_top - sig_bot) + sig_bot for stress in stresses]

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.plot(sig, vel, marker='o')

ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$v_f / c_s$')

gplt.savefig('supershear-transition/sigma-velocity.pdf')

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(np.array(sig) - 0.15490721596789805, a_ss, marker='o')

ax.set_xlabel(r'$\sigma - \sigma_c$')
ax.set_ylabel(r'$A^\star / h$')

gplt.savefig('supershear-transition/sigma-Astar.pdf')

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim([2e1, 6e2])
ax.set_ylim([7e1, 3e2])

ax.plot(a_c, a_ss, marker='o')

gplt.plot_powerlaw    (        0.5, 0.07, 0.0, 1.0, axis=ax, ls='--', c='k')
gplt.annotate_powerlaw(r'0.5', 0.5, 0.07, 0.0, 1.0, axis=ax, rx=.45, ry=.5, ha='right', va='bottom')

ax.set_xlabel(r'$A_c / h$')
ax.set_ylabel(r'$A^\star / h$')

gplt.savefig('supershear-transition/Ac-Astar.pdf')
