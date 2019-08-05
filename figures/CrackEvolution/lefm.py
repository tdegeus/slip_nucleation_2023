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
# constant dy
# --------------------------------------------------------------------------------------------------

if True:

  for dh in [0, 4, 30, 60]:

    for a in [100, 300, 500, 700]:

      fig, axes = gplt.subplots(ncols=3)

      for ax in axes:
        ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

      axes[0].set_ylabel(r'$\sigma_{xx}$')
      axes[1].set_ylabel(r'$\sigma_{yy}$')
      axes[2].set_ylabel(r'$\sigma_{xy}$')

      for ax in axes:
        ax.set_xlim([-400, +400])

      axes[0].set_ylim([-0.1, +0.1])
      axes[1].set_ylim([-0.1, +0.1])
      axes[2].set_ylim([+0.0, +0.2])

      gplt.text(.05, .9, r'$\Delta y / h = %d$' % dh, units='relative', axis=axes[1], bbox=dict(edgecolor='black', boxstyle=BoxStyle("Round, pad=0.3"), facecolor='white'))

      for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

        with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element-components.hdf5'), 'r') as data:

          sig_xx = data['/sig_xx/{0:d}'.format(a)][...]
          sig_yy = data['/sig_yy/{0:d}'.format(a)][...]
          sig_xy = data['/sig_xy/{0:d}'.format(a)][...]

          axes[0].plot(
            np.arange(N)[:mid] - a/2,
            sig_xx[plastic+dh*N][:mid],
            **color_stress(nx, stress), **label_stress_minimal(stress))

          axes[1].plot(
            np.arange(N)[:mid] - a/2,
            sig_yy[plastic+dh*N][:mid],
            **color_stress(nx, stress))

          axes[2].plot(
            np.arange(N)[:mid] - a/2,
            sig_xy[plastic+dh*N][:mid],
            **color_stress(nx, stress))

      axes[0].legend()

      gplt.savefig('lefm/const-dy/dh={0:d}_A={1:d}_stress=var.pdf'.format(dh, a))
      plt.close()

# --------------------------------------------------------------------------------------------------
# theta = 0, constant A, varying stress
# --------------------------------------------------------------------------------------------------

if True:

  for a in [100, 300, 500, 700]:

    if a % 2 == 0: a_half = int( a      / 2)
    else         : a_half = int((a - 1) / 2)

    tip   = plastic[np.arange(N)[:mid][a_half]]
    delem = np.arange(N)[:mid] - a_half
    elem  = tip + delem
    dr    = delem

    fig, axes = gplt.subplots(ncols=3)

    for ax in axes:
      ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

    axes[0].set_ylabel(r'$\sigma_{xx}$')
    axes[1].set_ylabel(r'$\sigma_{yy}$')
    axes[2].set_ylabel(r'$\sigma_{xy}$')

    for ax in axes:
      ax.set_xlim([-500,500])

    axes[0].set_ylim([-0.1, +0.1])
    axes[1].set_ylim([-0.1, +0.1])
    axes[2].set_ylim([+0.0, +0.2])

    gplt.text(.05, .9, r'$\theta = 0$', units='relative', axis=axes[1], bbox=dict(edgecolor='black', boxstyle=BoxStyle("Round, pad=0.3"), facecolor='white'))

    for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6'][::-1]:

      with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element-components.hdf5'), 'r') as data:

        sig_xx = data['/sig_xx/{0:d}'.format(a)][...]
        sig_yy = data['/sig_yy/{0:d}'.format(a)][...]
        sig_xy = data['/sig_xy/{0:d}'.format(a)][...]

        axes[0].plot(
          dr,
          sig_xx[elem],
          **color_stress(nx, stress), **label_stress_minimal(stress))

        axes[1].plot(
          dr,
          sig_yy[elem],
          **color_stress(nx, stress))

        axes[2].plot(
          dr,
          sig_xy[elem],
          **color_stress(nx, stress))

    axes[0].legend()

    gplt.savefig('lefm/const-theta/theta=0_A={0:d}_stress=var.pdf'.format(a))
    plt.close()

# --------------------------------------------------------------------------------------------------
# theta = pi/4, constant A, varying stress
# --------------------------------------------------------------------------------------------------

if True:

  for a in [100, 300, 500, 700]:

    if a % 2 == 0: a_half = int( a      / 2)
    else         : a_half = int((a - 1) / 2)

    tip   = plastic[np.arange(N)[:mid][a_half]]
    ny    = int((regular.nely() - 1)/2)
    delem = (np.linspace(-ny, ny, regular.nely()) * (N + 1)).astype(np.int)
    elem  = tip + delem
    dr    = np.linspace(-ny, ny, regular.nely()) * np.sqrt(2.)
    idx   = np.where(elem >= 0)[0]
    elem  = elem[idx]
    dr    = dr[idx]

    fig, axes = gplt.subplots(ncols=3)

    for ax in axes:
      ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

    axes[0].set_ylabel(r'$\sigma_{xx}$')
    axes[1].set_ylabel(r'$\sigma_{yy}$')
    axes[2].set_ylabel(r'$\sigma_{xy}$')

    for ax in axes:
      ax.set_xlim([-500,500])

    axes[0].set_ylim([-0.1, +0.1])
    axes[1].set_ylim([-0.1, +0.1])
    axes[2].set_ylim([+0.0, +0.2])

    gplt.text(.05, .9, r'$\theta = \pi / 4$', units='relative', axis=axes[1], bbox=dict(edgecolor='black', boxstyle=BoxStyle("Round, pad=0.3"), facecolor='white'))

    for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6'][::-1]:

      with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element-components.hdf5'), 'r') as data:

        sig_xx = data['/sig_xx/{0:d}'.format(a)][...]
        sig_yy = data['/sig_yy/{0:d}'.format(a)][...]
        sig_xy = data['/sig_xy/{0:d}'.format(a)][...]

        axes[0].plot(
          dr,
          sig_xx[elem],
          **color_stress(nx, stress), **label_stress_minimal(stress))

        axes[1].plot(
          dr,
          sig_yy[elem],
          **color_stress(nx, stress))

        axes[2].plot(
          dr,
          sig_xy[elem],
          **color_stress(nx, stress))

    axes[0].legend()

    gplt.savefig('lefm/const-theta/theta=pi-4_A={0:d}_stress=var.pdf'.format(a))
    plt.close()

# --------------------------------------------------------------------------------------------------
# theta = pi/2, constant A, varying stress
# --------------------------------------------------------------------------------------------------

if True:

  for a in [100, 300, 500, 700]:

    if a % 2 == 0: a_half = int( a      / 2)
    else         : a_half = int((a - 1) / 2)

    tip   = plastic[np.arange(N)[:mid][a_half]]
    ny    = int((regular.nely() - 1)/2)
    delem = (np.linspace(-ny, ny, regular.nely()) * N).astype(np.int)
    elem  = tip + delem
    dr    = np.linspace(-ny, ny, regular.nely())

    fig, axes = gplt.subplots(ncols=3)

    for ax in axes:
      ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

    axes[0].set_ylabel(r'$\sigma_{xx}$')
    axes[1].set_ylabel(r'$\sigma_{yy}$')
    axes[2].set_ylabel(r'$\sigma_{xy}$')

    for ax in axes:
      ax.set_xlim([-500,500])

    axes[0].set_ylim([-0.1, +0.1])
    axes[1].set_ylim([-0.1, +0.1])
    axes[2].set_ylim([+0.0, +0.2])

    gplt.text(.05, .9, r'$\theta = \pi / 2$', units='relative', axis=axes[1], bbox=dict(edgecolor='black', boxstyle=BoxStyle("Round, pad=0.3"), facecolor='white'))

    for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6'][::-1]:

      with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element-components.hdf5'), 'r') as data:

        sig_xx = data['/sig_xx/{0:d}'.format(a)][...]
        sig_yy = data['/sig_yy/{0:d}'.format(a)][...]
        sig_xy = data['/sig_xy/{0:d}'.format(a)][...]

        axes[0].plot(
          dr,
          sig_xx[elem],
          **color_stress(nx, stress), **label_stress_minimal(stress))

        axes[1].plot(
          dr,
          sig_yy[elem],
          **color_stress(nx, stress))

        axes[2].plot(
          dr,
          sig_xy[elem],
          **color_stress(nx, stress))

    axes[0].legend()

    gplt.savefig('lefm/const-theta/theta=pi-2_A={0:d}_stress=var.pdf'.format(a))
    plt.close()



