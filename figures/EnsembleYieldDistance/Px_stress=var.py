import sys, os, re, subprocess, shutil, h5py, HDF5pp

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

from collections import defaultdict

plt.style.use(['goose', 'goose-latex'])

from definitions import *

# ==================================================================================================
# Plot
# ==================================================================================================

if True:

  for bins in ['bins=15001']:

    for nx in ['nx=3^6', 'nx=3^6x2']:

      fig, ax = gplt.subplots()

      ax.set_xlabel(r'$x_\varepsilon / \varepsilon_0$')
      ax.set_ylabel(r'$\rho(x_\varepsilon / \varepsilon_0)$')

      ax.set_xscale('log')
      ax.set_yscale('log')

      ax.set_xlim([1e-2, 1e+1])
      ax.set_ylim([1e-4, 1e+1])

      for delta in list_stress()[::-1]:

        fname = os.path.join(dbase, nx, 'EnsembleYieldDistance_{0:s}.hdf5'.format(delta))

        if not os.path.isfile(fname): continue

        data = h5py.File(fname, 'r')

        P = data[bins]['P'][...]
        x = data[bins]['x'][...]

        data.close()

        ax.plot(x, P, **color(nx,nu,delta), label=label_delta(delta), rasterized=True)

      legend = ax.legend(loc='lower center', ncol=2)

      plt.savefig('{nx:s}/Px_stress=var_{bins:s}.pdf'.format(bins=bins, nx=nx), dpi=300)
      plt.close()

