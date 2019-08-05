import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np
import GooseFEM          as gf

from matplotlib.patches import BoxStyle

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

def applyFilter(xn):

  from scipy import signal

  # Create an order 3 lowpass butterworth filter:
  b, a = signal.butter(3, 0.05)

  # Apply the filter to xn. Use lfilter_zi to choose the initial condition of the filter:
  zi   = signal.lfilter_zi(b, a)
  z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

  # Apply the filter again, to have a result filtered at an order the same as filtfilt:
  z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

  # Use filtfilt to apply the filter:
  y = signal.filtfilt(b, a, xn)

  # Return data
  return y

# ==================================================================================================

from definitions import *

nx = 'nx=3^6x2'
N  = num_nx(nx)

# ==================================================================================================

with h5py.File(path(key='data', nx=nx, fname='EnsembleInfo.hdf5'), 'r') as data:
  h = data['/normalisation/l0'][...]

# --------------------------------------------------------------------------------------------------

if N % 2 == 0: mid = int( N      / 2)
else         : mid = int((N - 1) / 2)

mesh = gf.Mesh.Quad4.FineLayer(N, N, h)

plastic = mesh.elementsMiddleLayer()

mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)

regular = mapping.getRegularMesh()

coor  = regular.coor()
conn  = regular.conn()
elmat = regular.elementMatrix()

elmap   = mapping.getMap()
plastic = [elmap[i] for i in plastic]
plastic = np.array(plastic).reshape(-1)

# --------------------------------------------------------------------------------------------------
# theta = [0, pi/4, pi/2], constant A, varying stress
# --------------------------------------------------------------------------------------------------

if True:

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6'][::-1]:

    for theta in ['theta=0', 'theta=pi-4', 'theta=pi-2']:

      fig, axes = gplt.subplots(ncols=3)

      for ax in axes:
        ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

      axes[0].set_ylabel(r'$\sigma_{xx} / \sqrt{A}$')
      axes[1].set_ylabel(r'$\sigma_{yy} / \sqrt{A}$')
      axes[2].set_ylabel(r'$(\sigma_{xy} - \sigma_c^\star) / \sqrt{A}$')

      for ax in axes:
        ax.set_xlim([-500,500])

      axes[0].set_ylim([-0.01, +0.01])
      axes[1].set_ylim([-0.01, +0.01])
      axes[2].set_ylim([-0.01, +0.01])

      if   theta == 'theta=0'   : label = r'$\theta = 0$, '       + label_stress(stress)['label']
      elif theta == 'theta=pi-4': label = r'$\theta = \pi / 4$, ' + label_stress(stress)['label']
      elif theta == 'theta=pi-2': label = r'$\theta = \pi / 2$, ' + label_stress(stress)['label']
      else: raise IOError('Unknown theta')

      gplt.text(.05, .9, label, units='relative', axis=axes[1], bbox=dict(edgecolor='black', boxstyle=BoxStyle("Round, pad=0.3"), facecolor='white'))

      cmap = plt.get_cmap('jet', 14)

      for ia, a in enumerate(np.linspace(50, 700, 14).astype(np.int)):

        # --

        if a % 2 == 0: a_half = int( a      / 2)
        else         : a_half = int((a - 1) / 2)

        # --

        if theta == 'theta=0':

          tip   = plastic[np.arange(N)[:mid][a_half]]
          delem = np.arange(N)[:mid] - a_half
          elem  = tip + delem
          dr    = delem

        elif theta == 'theta=pi-4':

          tip   = plastic[np.arange(N)[:mid][a_half]]
          ny    = int((regular.nely() - 1)/2)
          delem = (np.linspace(-ny, ny, regular.nely()) * (N + 1)).astype(np.int)
          elem  = tip + delem
          dr    = np.linspace(-ny, ny, regular.nely()) * np.sqrt(2.)
          idx   = np.where(elem >= 0)[0]
          elem  = elem[idx]
          dr    = dr[idx]

        elif theta == 'theta=pi-2':

          tip   = plastic[np.arange(N)[:mid][a_half]]
          ny    = int((regular.nely() - 1)/2)
          delem = (np.linspace(-ny, ny, regular.nely()) * N).astype(np.int)
          elem  = tip + delem
          dr    = np.linspace(-ny, ny, regular.nely())

        else:

          raise IOError('Unknown theta')

        # --

        with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element-components.hdf5'), 'r') as data:

          sig_xx = data['/sig_xx/{0:d}'.format(a)][...]
          sig_yy = data['/sig_yy/{0:d}'.format(a)][...]
          sig_xy = data['/sig_xy/{0:d}'.format(a)][...]

          sig_xx = sig_xx[elem]
          sig_yy = sig_yy[elem]
          sig_xy = sig_xy[elem]

          idx = np.where(np.abs(dr) < h * np.sqrt(2.))[0]

          sig_xx[idx] = np.NaN
          sig_yy[idx] = np.NaN
          sig_xy[idx] = np.NaN

          sigma_c = np.mean(sig_xy[:100])

          if a == 50 or a == 700: l = {'label': r'$A = {0:d}$'.format(a)}
          else                  : l = {}

          axes[0].plot(
            dr,
            sig_xx / np.sqrt(1. * a),
            color=cmap(ia), **l)

          axes[1].plot(
            dr,
            sig_yy / np.sqrt(1. * a),
            color=cmap(ia))

          axes[2].plot(
            dr,
            (sig_xy - sigma_c) / np.sqrt(1. * a),
            color=cmap(ia))

      axes[0].legend()

      gplt.savefig('lefm_collapse/const-theta/{0:s}_{1:s}_A=var.pdf'.format(stress, theta))
      plt.close()

