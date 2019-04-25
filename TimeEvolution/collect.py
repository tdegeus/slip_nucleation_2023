
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
# combine files
# ==================================================================================================

for ensemble in sorted(ensembles):

  dest = h5py.File(ensemble + '.hdf5', 'w')

  for file in ensembles[ensemble]:

    root = os.path.splitext(file)[0]

    source = h5py.File(file, 'r')

    source_datasets = list(g5.getdatasets(source))

    dest_datasets = ['/' + root + dataset for dataset in source_datasets]

    g5.copydatasets(source, dest, source_datasets, dest_datasets)

    source.close()

  dest.close()



