import sys, os, re, subprocess, shutil, h5py, HDF5pp, matplotlib

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

from collections import defaultdict

plt.style.use(['goose', 'goose-latex'])

from definitions import *

# ==================================================================================================
# Read data
# ==================================================================================================

Cutoff = defaultdict(dict)
ErrorBar = defaultdict(dict)

for nx in list_nx():

  for delta in list_delta():

    fname = os.path.join(dbase, nx, 'AvalancheAfterPush_{0:s}.hdf5'.format(delta))

    if not os.path.isfile(fname): continue

    data = h5py.File(fname, 'r')

    A = data['A'][...]
    S = data['S'][...]

    idx = np.where(A>0)[0]

    A = A[idx]
    S = S[idx]

    idx = np.where(A<num_nx(nx))[0]

    S = S[idx]
    A = A[idx]

    p = 4.

    Ac = np.mean(A**(p+1)) / np.mean(A**p)

    Ec = np.sqrt( Ac**2. * ( (np.std(A**(p+1.))/np.mean(A**(p+1.)))**2. + (np.std(A**p)/np.mean(A**p))**2.) ) / np.sqrt(float(len(A)))

    Cutoff[nx][delta] = Ac

    ErrorBar[nx][delta] = Ec

    data.close()

# ==================================================================================================

DeltaSigma = defaultdict(dict)

for nx in list_nx():

  fname = os.path.join(dbase, nx, 'EnsembleInfo.hdf5')

  data = h5py.File(fname, 'r')

  bot = float(data['/averages/sigd_bottom'][...])
  top = float(data['/averages/sigd_top'   ][...])

  data.close()

  for delta in list_delta():

    if   delta == 'stress=0d6': offset = 0.15490721596789805
    elif delta == 'stress=1d6': offset = 0.14998521525303932
    elif delta == 'stress=2d6': offset = 0.14840373943914173
    elif delta == 'stress=3d6': offset = 0.13998807437965494
    elif delta == 'stress=4d6': offset = 0.13181176453793328
    elif delta == 'stress=5d6': offset = 0.14326478188785668
    elif delta == 'stress=6d6': offset = 0.11904209626971712
    else: offset = 0.15490721596789805

    DeltaSigma[nx][delta] = num_delta(delta) * (top-bot) + bot - offset

# ==================================================================================================
# Plot
# ==================================================================================================

if True:

  for nx in ['nx=3^6x2']:

    fig, ax = gplt.subplots()

    gplt.diagonal_powerlaw(-2., tl=(4e-2, 2e3), width=1e1, axis=ax)

    ax.set_xlabel(r'$\sigma - \sigma_c(\Delta_\sigma)$')
    ax.set_ylabel(r'$A_c$')

    gplt.plot_powerlaw(            -2., 0, 1., 1, axis=ax, color='k', units='relative', linestyle='--', linewidth=1)
    gplt.annotate_powerlaw('$-2$', -2., 0, 1., 1, axis=ax, color='k', units='relative', rx=.1, ry=.1)

    e = [ErrorBar[nx][delta]   for delta in list_stress()]
    y = [Cutoff[nx][delta]     for delta in list_stress()]
    x = [DeltaSigma[nx][delta] for delta in list_stress()]

    ax.errorbar(x, y, yerr=e, **label_nx(nx), **color_nx(nx), marker='o')

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    plt.savefig('Ac-stress.pdf'.format(nx=nx))
    plt.close()

