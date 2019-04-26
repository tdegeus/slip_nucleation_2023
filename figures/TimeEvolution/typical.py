import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# --------------------------------------------------------------------------------------------------

from definitions import *

nx = 'nx=3^6x2'

N = num_nx(nx)

# --------------------------------------------------------------------------------------------------

def getRenumIndex(old, new, N, center=False):

  idx = np.tile(np.arange(N), (3))

  shift = 0

  if center:
    shift = int(new / 2.)

  return idx[old+N-(new-shift) : old+2*N-(new-shift)]

# --------------------------------------------------------------------------------------------------

def getIncrementAc(data, Ac):

  # get stored increments
  incs = data['stored'][...]

  # get reference yield index
  idx0 = data['idx']['0'][...]

  # loop over increments
  for inc in incs:

    # - get current yield index
    idx = data['idx'][str(inc)][...]

    # - avalanche area
    A = np.sum(idx0 != idx)

    if A >= Ac:
      return inc

  # last increment
  if Ac == idx.size:
    return incs[-1]

  raise IOError('Increment not found')

# --------------------------------------------------------------------------------------------------

def applyFilter(xn):

  from scipy import signal

  # Size of the measurement
  N = xn.size

  # Periodic repetitions
  xn = np.tile(xn, (3))

  # Create an order 3 lowpass butterworth filter:
  b, a = signal.butter(3, 0.05)

  # Apply the filter to xn. Use lfilter_zi to choose the initial condition of the filter:
  zi   = signal.lfilter_zi(b, a)
  z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

  # Apply the filter again, to have a result filtered at an order the same as filtfilt:
  z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

  # Use filtfilt to apply the filter:
  y = signal.filtfilt(b, a, xn)

  # Select middle of the periodic repetitions
  y = y[N: 2*N]

  # Return data
  return y

# --------------------------------------------------------------------------------------------------

data = h5py.File(path(nx=nx, fname='EnsembleInfo.hdf5'), 'r')

sig0  = data['/normalisation/sigy'][...]
sig_c = data['/averages/sigd_bottom'][...]

data.close()

# --------------------------------------------------------------------------------------------------

data = h5py.File(path(nx=nx, fname='TimeEvolution_stress=6d6.hdf5'), 'r')

# set crack size to consider
Ac = 600

# allocate averages
Sig_xx = np.zeros((5, N)) # middle layer + two layers above and below
Sig_xy = np.zeros((5, N)) # middle layer + two layers above and below
Sig_yy = np.zeros((5, N)) # middle layer + two layers above and below
S      = np.zeros((   N)) # middle layer
norm   = 0.

# loop over all pushes
for file in sorted([f for f in data]):

  # - find the increment for which the "crack" is first bigger than "Ac"
  inc = getIncrementAc(data[file], Ac)

  # - get reference and current yield-index
  idx0 = data[file]['idx']['0'     ][...]
  idx  = data[file]['idx'][str(inc)][...]

  # - get stress (of the middle layer + two layers above and below)
  Sig = data[file]['Sig'][str(inc)][...]
  # - stress indices
  sig_xx = Sig[:, 0, 0].reshape(-1, N)
  sig_xy = Sig[:, 0, 1].reshape(-1, N)
  sig_yx = Sig[:, 1, 0].reshape(-1, N)
  sig_yy = Sig[:, 1, 1].reshape(-1, N)

  # - indices of blocks where yielding took place
  icell = np.argwhere(idx0 != idx).ravel()
  icell = np.hstack((icell, icell[0]+N))
  # - right-most-block containing the crack
  iright = icell[np.argmax(np.diff(icell))] + 1
  # - renumber such that the first "Ac" items are in the crack
  renum = getRenumIndex(iright, Ac, N)

  # - align stress to overlapping right side of the crack
  sig_xx = sig_xx[:, renum]
  sig_xy = sig_xy[:, renum]
  sig_yy = sig_yy[:, renum]
  # - align number of times yielded to overlapping right side of the crack
  idx  = idx .astype(np.int)
  idx0 = idx0.astype(np.int)
  s    = (idx - idx0)[renum]

  # - add stress to ensemble average
  Sig_xx += sig_xx
  Sig_xy += sig_xy
  Sig_yy += sig_yy
  # - add number of times yielded to ensemble average
  S += s
  # - update number of measurement in ensemble average
  norm += 1.

# close the data-file
data.close()

# normalise stress tensor
Sig_xx /= norm
Sig_xy /= norm
Sig_yy /= norm

# normalise number of times yielded
S /= norm

# hydrostatic stress
Sig_m = (Sig_xx + Sig_yy) / 2.

# deviatoric stress
Sigd_xx = Sig_xx - Sig_m
Sigd_xy = Sig_xy
Sigd_yy = Sig_yy - Sig_m

# equivalent stress
Sig_eq = np.sqrt(2.0 * (Sigd_xx**2.0 + Sigd_yy**2.0 + 2.0 * Sigd_xy**2.0))

# normalise stress
Sig_eq /= sig0

# get (filtered measurement)
xn = Sig_eq[2, :]
y  = applyFilter(xn)

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.set_ylim([0.0, 0.6])
ax.set_xlim([0, N])

ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$\sigma$')

ax.plot(np.arange(N), xn, lw=.5)
ax.plot(np.arange(N), y)

ax.plot([0, N], [sig_c, sig_c], c='b', ls='--')

ax.plot([0 , 0 ], ax.get_ylim(), c='g')
ax.plot([Ac, Ac], ax.get_ylim(), c='g')

plt.savefig('typical_stress.pdf')
plt.close()

# --------------------------------------------------------------------------------------------------

xn = S
y  = applyFilter(xn)

fig, ax = plt.subplots()

ax.set_xlim([0, N])

ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$S$')

ax.plot(np.arange(N), xn, lw=.5)
ax.plot(np.arange(N), y)

ax.plot([0 , 0 ], ax.get_ylim(), c='g')
ax.plot([Ac, Ac], ax.get_ylim(), c='g')

plt.savefig('typical_S.pdf')
plt.close()



