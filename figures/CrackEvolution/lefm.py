import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np
import GooseFEM          as gf

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

for dh in [0, 4, 30, 60]:

  for a in [100, 300, 500, 700]:

    fig, axes = gplt.subplots(ncols=3)

    for ax in axes:
      ax.set_xlabel(r'$(x - x_\mathrm{tip}) / h$')

    axes[0].set_ylabel(r'$\sigma_{xx}$')
    axes[1].set_ylabel(r'$\sigma_{xy}$')
    axes[2].set_ylabel(r'$\sigma_{yy}$')

    for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

      with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element-components.hdf5'), 'r') as data:

        sig_xx = data['/sig_xx/{0:d}'.format(a)][...]

        axes[0].plot(
          np.arange(N)[:mid] - a/2,
          sig_xx[plastic+dh*N][:mid],
          **color_stress(nx, stress), **label_stress_minimal(stress))

        sig_xy = data['/sig_xy/{0:d}'.format(a)][...]

        axes[1].plot(
          np.arange(N)[:mid] - a/2,
          sig_xy[plastic+dh*N][:mid],
          **color_stress(nx, stress))

        sig_yy = data['/sig_yy/{0:d}'.format(a)][...]

        axes[2].plot(
          np.arange(N)[:mid] - a/2,
          sig_yy[plastic+dh*N][:mid],
          **color_stress(nx, stress))

    axes[0].legend()

    gplt.savefig('lefm/stress=var_dh={0:d}_A={1:d}.pdf'.format(dh, a))
    plt.close()


