
import os, subprocess, h5py, HDF5pp
import numpy      as np
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
# combine files
# ==================================================================================================

for ensemble in sorted(ensembles):

  dest = h5py.File(ensemble + '.hdf5', 'w')

  for file in ensembles[ensemble]:

    source = h5py.File(file, 'r')

    datasets = list(HDF5pp.getdatasets(source))

    HDF5pp.copydatasets(source, dest, datasets, datasets)

    source.close()

  dest.close()



