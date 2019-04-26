
import numpy as np
import os, h5py
import GooseHDF5 as g5

# --------------------------------------------------------------------------------------------------

def getRenumIndex(old, new, N):

  idx = np.tile(np.arange(N), (3))

  return idx[old+N-new : old+2*N-new]

# --------------------------------------------------------------------------------------------------

def _getCrackInfoFile(data, max_skip=5):
  r'''
Get basic information of all increments in a single file.

:options:

  **max_skip** (``<int>``)
    Remove right-most-block of the crack when it is ``max_skip`` blocks away from the former
    block that yielded (this eliminates individual yield events that can occur by chance).
  '''

  # get stored increments
  incs = data['stored'][...]

  # allocate output
  out = {
    # - time since pushing
    'dt'   : data['iiter'][...] * data['dt'][...],
    # - raw measurement
    'A'    : np.zeros(incs.shape, dtype=int), # avalanche area
    'R'    : np.zeros(incs.shape, dtype=int), # avalanche extension
    'RIGHT': np.zeros(incs.shape, dtype=int), # index of the right-most-block of "A"
    'LEFT' : np.zeros(incs.shape, dtype=int), # index of the left -most-block of "A"
    # - corrected measurement (removed isolated yielding events, measured by "max_skip")
    'a'    : np.zeros(incs.shape, dtype=int), # crack area
    'r'    : np.zeros(incs.shape, dtype=int), # crack extension
    'right': np.zeros(incs.shape, dtype=int), # index of the right-most-block of "a"
    'left' : np.zeros(incs.shape, dtype=int), # index of the left -most-block of "a"
  }

  # get reference yield index
  idx0 = data['idx']['0'][...]

  # get system size
  N = idx0.size

  # loop over increments
  for inc in incs:

    # - get current yield index
    idx = data['idx'][str(inc)][...]

    # - indices of blocks where yielding took place during the quench
    icell = np.argwhere(idx0 != idx).ravel()

    # - skip increment without any yielding no yielding
    if len(icell) == 0:
      continue

    # - add periodicity
    icell = np.hstack((icell, icell[0]+N))

    # - extract information
    out['A'    ][inc] = np.sum(idx0 != idx)
    out['R'    ][inc] = N - (np.max(np.diff(icell)) - 1)
    out['LEFT' ][inc] = icell[np.argmax(np.diff(icell)) + 1]
    out['RIGHT'][inc] = icell[np.argmax(np.diff(icell))] + 1

    # - correct for periodicity
    if out['LEFT' ][inc] >= N: out['LEFT' ][inc] -= N
    if out['RIGHT'][inc] >= N: out['RIGHT'][inc] -= N

    # - default correct data: no correction
    out['a'    ][inc] = out['A'    ][inc]
    out['r'    ][inc] = out['R'    ][inc]
    out['left' ][inc] = out['LEFT' ][inc]
    out['right'][inc] = out['RIGHT'][inc]

    # - skip correction for too small avalanches
    if out['A'][inc] < 10:
      continue
    # - skip correction if no correction is needed
    if np.max(np.unique(np.diff(icell))[:-1]) < max_skip:
      continue

    # - renumber left-most-block as the zeroth block (to avoid difficulties with periodicity)
    renum = getRenumIndex(out['LEFT'][inc], 0, N)

    # - renumbered yield-index
    jdx0 = idx0[renum]
    jdx  = idx [renum]

    # - renumbered left- and right-most-block
    l = np.argwhere(renum == out['LEFT' ][inc]).ravel()[0]
    r = np.argwhere(renum == out['RIGHT'][inc]).ravel()[0]

    # - check to move extremities inwards, if they are too isolated events
    while True:

      # - move right-most-block to the left
      if np.sum((jdx0 != jdx)[r-max_skip: r-1]) == 0:

        # -- get index of all other yielding blocks, left of "r""
        i = np.argwhere((jdx0 != jdx)[:r-1])

        # -- no blocks remaining: do nothing
        if len(i) == 0:
          break

        # -- move right-most-block to the next block to the left
        j = np.max(i)
        jdx[j:r] = jdx0[j:r]
        r = j

        # -- continue to check once again
        continue

      # - move left-most-block to the right
      if np.sum((jdx0 != jdx)[l+1: l+max_skip]) == 0:

        # -- get index of all other yielding blocks, right of "r""
        i = np.argwhere((jdx0 != jdx)[l+1:])

        # -- no blocks remaining: do nothing
        if len(i) == 0:
          break

        # -- move left-most-block to the next block to the right
        j = np.min(i) + l
        jdx[l:j] = jdx0[l:j]
        l = j

        # -- continue to check once again
        continue

      # - no changes made: stop checking
      break

    # - revert renumbering
    tmp0 = np.array(jdx0, copy=True)
    tmp  = np.array(jdx , copy=True)
    jdx0[renum] = tmp0
    jdx [renum] = tmp

    # - indices of blocks where yielding took place
    jcell = np.argwhere(jdx0 != jdx).ravel()

    # - skip no yielding
    if len(jcell) == 0:
      continue

    # - add periodicity
    jcell = np.hstack((jcell, jcell[0]+N))

    # - store information
    out['a'    ][inc] = np.sum(jdx0 != jdx)
    out['r'    ][inc] = N - (np.max(np.diff(jcell)) - 1)
    out['left' ][inc] = jcell[np.argmax(np.diff(jcell)) + 1]
    out['right'][inc] = jcell[np.argmax(np.diff(jcell))] + 1

    # - correct for periodicity
    if out['left' ][inc] >= N: out['left' ][inc] -= N
    if out['right'][inc] >= N: out['right'][inc] -= N

  # return output
  return out

# --------------------------------------------------------------------------------------------------

def getCrackInfo(data):

  # allocate output
  out = {}

  # loop over all measurements in the ensemble
  for file in sorted([f for f in data]):

    print(file)

    out[file] = _getCrackInfoFile(data[file])

  # return output
  return out

# --------------------------------------------------------------------------------------------------

from definitions import *

# create folder
if not os.path.isdir('meta-data'): os.makedirs('meta-data')

# loop over data
for nx in list_nx():

  for stress in list_stress():

    data = h5py.File(path(nx=nx, fname='TimeEvolution_{stress:s}.hdf5'.format(stress=stress)), 'r')

    meta = getCrackInfo(data)

    data.close()

    data = h5py.File(os.path.join('meta-data', '{nx:s}_{stress:s}.hdf5'.format(nx=nx, stress=stress)), 'w')

    for file in meta:
      for field in meta[file]:
        data[g5.join(file, field)] = meta[file][field]

    data.close()
