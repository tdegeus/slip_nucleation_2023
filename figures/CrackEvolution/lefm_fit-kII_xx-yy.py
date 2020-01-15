import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np
import GooseFEM          as gf

from matplotlib.patches import BoxStyle
from scipy.optimize import curve_fit

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
  sig_bot = data['/averages/sigd_bottom'][...]
  sig_top = data['/averages/sigd_top'][...]

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

ny = int((regular.nely() - 1)/2)

# --------------------------------------------------------------------------------------------------
# return LEFM solution
# --------------------------------------------------------------------------------------------------

def lefm(RData, *args):

  assert len(RData) % 4 * 2 == 0

  RData = RData.reshape(4, 2, -1)

  kII = args[0]

  SigData = np.empty(RData.shape)

  for itheta, theta in enumerate([np.pi/2.0, -np.pi/2.0, np.pi/4.0, -np.pi/4.0]):

    r = RData[itheta, 0, :]

    sig_xx = -kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) * (2.0 + np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0))
    sig_yy = +kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) *        np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0)
    sig_xy = +kII / np.sqrt(2.0 * np.pi * r) * np.cos(theta / 2.0) * (1.0 - np.sin(theta / 2.0) * np.sin(3.0 * theta / 2.0))

    SigData[itheta, 0, :] = sig_xx
    SigData[itheta, 1, :] = sig_yy
    # SigData[itheta, 2, :] = sig_xy

  RData = RData.reshape(-1)
  SigData = SigData.reshape(-1)

  return SigData

# --------------------------------------------------------------------------------------------------
# read stress
# --------------------------------------------------------------------------------------------------

def read_stress(nx, stress, a, elem):

  with h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element-components.hdf5'), 'r') as data:

    sig_xx = data['/sig_xx/{0:d}'.format(a)][...]
    sig_yy = data['/sig_yy/{0:d}'.format(a)][...]
    sig_xy = data['/sig_xy/{0:d}'.format(a)][...]

    sig_xx = sig_xx[elem]
    sig_yy = sig_yy[elem]
    sig_xy = sig_xy[elem]

  return (sig_xx, sig_yy, sig_xy)

# --------------------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------------------

Ameasure = np.linspace(50, 700, 14).astype(np.int)
Stresses = ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']

kII_data = {stress: np.empty((len(Ameasure))) for stress in Stresses}
kII_error = {stress: np.empty((len(Ameasure))) for stress in Stresses}

skipL = 10
skipR = 50

for stress in Stresses:

  for ia, a in enumerate(Ameasure):

    if a % 2 == 0: a_half = int( a      / 2)
    else         : a_half = int((a - 1) / 2)

    RData   = np.empty((4, 2, ny - skipL - skipR))
    SigData = np.empty((4, 2, ny - skipL - skipR))

    for itheta, theta in enumerate([np.pi/2.0, -np.pi/2.0, np.pi/4.0, -np.pi/4.0]):

      if   itheta == 0: # pi / 2

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = (np.linspace(1 + skipL, ny - skipR, ny - skipL - skipR) * N).astype(np.int)
        elem  = tip + delem
        dr    = np.linspace(1 + skipL, ny - skipR, ny - skipL - skipR)

      elif itheta == 1: # - pi / 2

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = (-1 * np.linspace(1 + skipL, ny - skipR, ny - skipL - skipR) * N).astype(np.int)
        elem  = tip + delem
        dr    = np.linspace(1 + skipL, ny - skipR, ny - skipL - skipR)

      elif itheta == 2: # pi / 4

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = (np.linspace(1 + skipL, ny - skipR, ny - skipL - skipR) * (N + 1)).astype(np.int)
        elem  = tip + delem
        dr    = np.linspace(1 + skipL, ny - skipR, ny - skipL - skipR) * np.sqrt(2.)

      elif itheta == 3: # - pi / 4

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = (-1 * np.linspace(1 + skipL, ny - skipR, ny - skipL - skipR) * (N + 1)).astype(np.int)
        elem  = tip + delem
        dr    = np.linspace(1 + skipL, ny - skipR, ny - skipL - skipR) * np.sqrt(2.)

      sig_xx, sig_yy, sig_xy = read_stress(nx, stress, a, elem)

      RData[itheta, 0, :] = dr
      RData[itheta, 1, :] = dr
      # RData[itheta, 2, :] = dr

      SigData[itheta, 0, :] = sig_xx
      SigData[itheta, 1, :] = sig_yy
      # SigData[itheta, 2, :] = sig_xy

    RData = RData.reshape(-1)
    SigData = SigData.reshape(-1)

    d, _ = curve_fit(lefm, RData, SigData, p0 = (1.0,))

    kII_data[stress][ia] = d
    kII_error[stress][ia] = np.linalg.norm(SigData - lefm(RData, d)) / np.sqrt(SigData.size)

# --------------------------------------------------------------------------------------------------

fig, ax = gplt.subplots()

for stress in Stresses:

  ax.errorbar(Ameasure, kII_data[stress], yerr=kII_error[stress],
    marker='o', **color_stress(nx, stress), **label_stress_minimal(stress))

ax.set_xlabel(r'$A$')
ax.set_ylabel(r'$K_{II}$')

ax.legend()

gplt.savefig('lefm_fit-kII/xx-yy/K-A.pdf')
plt.close()

# --------------------------------------------------------------------------------------------------

fig, ax = gplt.subplots()

for stress in Stresses:

  ax.errorbar(Ameasure, kII_data[stress], yerr=kII_error[stress],
    marker='o', **color_stress(nx, stress), **label_stress_minimal(stress))

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim([4e1, 1e3])
ax.set_ylim([1e-2, 1e0])

ax.set_xlabel(r'$A$')
ax.set_ylabel(r'$K_{II}$')

ax.legend()

gplt.plot_powerlaw(0.5, 0, 0.6, 1, units='relative', axis=ax, ls='--', lw=1)

gplt.savefig('lefm_fit-kII/xx-yy/K-A_log-log.pdf')
plt.close()

# --------------------------------------------------------------------------------------------------

fig, ax = gplt.subplots()

for stress in Stresses:

  if stress == 'stress=0d6':
    continue

  sig_inf = (sig_top - sig_bot) * num_stress(stress)

  ax.errorbar(Ameasure, kII_data[stress]/sig_inf, yerr=kII_error[stress],
    marker='o', **color_stress(nx, stress), **label_stress_minimal(stress))

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim([4e1, 1e3])
ax.set_ylim([1e-1, 1e1])

ax.set_xlabel(r'$A$')
ax.set_ylabel(r'$K_{II} / (\sigma - \sigma_c)$')

ax.legend()

gplt.plot_powerlaw(0.5, 0, 0.5, 1, units='relative', axis=ax, ls='--', lw=1)

gplt.savefig('lefm_fit-kII/xx-yy/K-A_log-log_scale.pdf')
plt.close()

# --------------------------------------------------------------------------------------------------
# theta = [0, pi/4, pi/2, -pi/4, -pi/2], constant A, varying stress
# --------------------------------------------------------------------------------------------------

if True:

  for theta in ['theta=0', 'theta=pi?4', 'theta=pi?2', 'theta=-pi?4', 'theta=-pi?2']:

    for a in [100, 300, 500, 700]:

      # --

      if a % 2 == 0: a_half = int( a      / 2)
      else         : a_half = int((a - 1) / 2)

      # --

      if theta == 'theta=0':

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = np.arange(N)[:mid] - a_half
        elem  = tip + delem
        dr    = delem
        label = r'$\theta = 0$'
        T     = 0.0

      elif theta == 'theta=pi?4':

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = (np.linspace(0, ny, ny+1) * (N + 1)).astype(np.int)
        elem  = tip + delem
        dr    = np.linspace(0, ny, ny+1) * np.sqrt(2.)
        label = r'$\theta = \pi / 4$'
        T     = np.pi / 4.0

      elif theta == 'theta=pi?2':

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = (np.linspace(0, ny, ny+1) * N).astype(np.int)
        elem  = tip + delem
        dr    = np.linspace(0, ny, ny+1)
        label = r'$\theta = \pi / 2$'
        T     = np.pi / 2.0

      elif theta == 'theta=-pi?4':

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = (-1 * np.linspace(0, ny, ny+1) * (N + 1)).astype(np.int)
        elem  = tip + delem
        dr    = np.linspace(0, ny, ny+1) * np.sqrt(2.)
        label = r'$\theta = - \pi / 4$'
        T     = - np.pi / 4.0

      elif theta == 'theta=-pi?2':

        tip   = plastic[np.arange(N)[:mid][a_half]]
        delem = (-1 * np.linspace(0, ny, ny+1) * N).astype(np.int)
        elem  = tip + delem
        dr    = np.linspace(0, ny, ny+1)
        label = r'$\theta = - \pi / 2$'
        T     = - np.pi / 2.0

      else:

        raise IOError('Unknown theta')

      # --

      fig, axes = gplt.subplots(ncols=3)

      for ax in axes:
        ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

      axes[0].set_ylabel(r'$\sigma_{xx}$')
      axes[1].set_ylabel(r'$\sigma_{yy}$')
      axes[2].set_ylabel(r'$\sigma_{xy}$')

      for ax in axes:
        ax.set_xlim([0, 500])

      axes[0].set_ylim([-0.1, +0.1])
      axes[1].set_ylim([-0.1, +0.1])
      axes[2].set_ylim([+0.0, +0.2])

      gplt.text(.05, .9, label, units='relative', axis=axes[1], bbox=dict(edgecolor='black', boxstyle=BoxStyle("Round, pad=0.3"), facecolor='white'))

      for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6'][::-1]:

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

          axes[0].plot(dr, sig_xx, **color_stress(nx, stress), **label_stress_minimal(stress))
          axes[1].plot(dr, sig_yy, **color_stress(nx, stress))
          axes[2].plot(dr, sig_xy, **color_stress(nx, stress))

          idx = np.where(a == Ameasure)[0]
          kII = kII_data[stress][idx]
          r = np.array(dr, copy=True)
          r[dr <= 0] = 1.0

          sig_xx = -kII / np.sqrt(2.0 * np.pi * r) * np.sin(T / 2.0) * (2.0 + np.cos(T / 2.0) * np.cos(3.0 * T / 2.0))
          sig_yy = +kII / np.sqrt(2.0 * np.pi * r) * np.sin(T / 2.0) *        np.cos(T / 2.0) * np.cos(3.0 * T / 2.0)
          sig_xy = +kII / np.sqrt(2.0 * np.pi * r) * np.cos(T / 2.0) * (1.0 - np.sin(T / 2.0) * np.sin(3.0 * T / 2.0))

          sig_xy += sig_bot
          print(sig_bot, sig_top)

          sig_xx[dr <= 0] = np.NaN
          sig_yy[dr <= 0] = np.NaN
          sig_xy[dr <= 0] = np.NaN

          # print(sig_xy[-10:])

          axes[0].plot(dr, sig_xx, **color_stress(nx, stress), lw=1)
          axes[1].plot(dr, sig_yy, **color_stress(nx, stress), lw=1)
          axes[2].plot(dr, sig_xy, **color_stress(nx, stress), lw=1)

      axes[0].legend()

      gplt.savefig('lefm_fit-kII/xx-yy/const-theta/{0:s}_A={1:d}_stress=var.pdf'.format(theta, a))
      plt.close()
