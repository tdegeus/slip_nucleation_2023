
import os, subprocess, h5py
import numpy      as np
import GooseHDF5  as g5
import GooseSLURM as gs

# ==================================================================================================
# get all simulation files, split in ensembles
# ==================================================================================================

files = subprocess.check_output("find . -iname '*.hdf5'", shell=True).decode('utf-8')
files = list(filter(None, files.split('\n')))
files = [os.path.relpath(file) for file in files]

ensembles = {}

for ensemble in list(set([file.split('id=')[0] for file in files])):

  name = ensemble[:-1]

  ensembles[name] = sorted([file for file in files if len(file.split(ensemble))>1])

# ==================================================================================================
# extract data
# ==================================================================================================

def getInfo(data):

  # get stored increments
  incs = data['stored'][...]

  # allocate output
  out = {
    'dt'      : data['iiter'][...] * data['dt'][...],
    'A'       : np.zeros(incs.shape, dtype=int),
    'sigbar'  : data['sigbar'][...],
    'sigweak' : np.zeros(incs.shape, dtype=float),
  }

  # get reference yield index
  idx0 = data['idx']['0'][...]

  # get system size
  N = idx0.size

  # loop over increments
  for inc in incs:

    # - get current yield index
    idx = data['idx'][str(inc)][...]

    # - extract information
    out['A'][inc] = np.sum(idx0 != idx)

    # - read stress tensor
    Sig = data['Sig'][str(inc)][...]

    # - decompose stress
    sig_xx = Sig[:, 0, 0].reshape(-1, N)
    sig_xy = Sig[:, 0, 1].reshape(-1, N)
    sig_yy = Sig[:, 1, 1].reshape(-1, N)

    # - compute averages
    sig_xx = np.mean(sig_xx, axis=1)
    sig_xy = np.mean(sig_xy, axis=1)
    sig_yy = np.mean(sig_yy, axis=1)

    # - hydrostatic stress
    sig_m = (sig_xx + sig_yy) / 2.

    # - deviatoric stress
    sigd_xx = sig_xx - sig_m
    sigd_xy = sig_xy
    sigd_yy = sig_yy - sig_m

    # - equivalent stress
    sig_eq = np.sqrt(2.0 * (sigd_xx**2.0 + sigd_yy**2.0 + 2.0 * sigd_xy**2.0))

    # - extract information
    out['sigweak'][inc] = sig_eq[2]

  return out

# ==================================================================================================
# combine files
# ==================================================================================================

for ensemble in sorted(ensembles):

  dest = h5py.File(ensemble + '.hdf5', 'w')

  for file in ensembles[ensemble]:

    root = os.path.splitext(file)[0]

    data = h5py.File(file, 'r')

    out = getInfo(data)

    for key in out:

      dest[g5.join(file, key)] = out[key]

    data.close()

  dest.close()



