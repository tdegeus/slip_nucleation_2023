
import sys
import os
import re
import subprocess
import h5py
import numpy as np

dbase = '../../../data/nx=3^6x2'
N = (3**6) * 2

keys = [
    '/conn',
    '/coor',
    '/cusp/G',
    '/cusp/K',
    '/cusp/elem',
    '/cusp/epsy',
    '/damping/alpha',
    '/damping/eta_d',
    '/damping/eta_v',
    '/dofs',
    '/dofsP',
    '/elastic/G',
    '/elastic/K',
    '/elastic/elem',
    '/rho',
    '/run/dt',
    '/run/epsd/kick',
    '/run/epsd/max',
    '/uuid',
]

with h5py.File(os.path.join(dbase, 'EnsembleInfo.hdf5'), 'r') as data:

    A = data['/avalanche/A'][...]
    idx = np.argwhere(A == N).ravel()
    incs = data['/avalanche/inc'][idx]
    files = data['/files'][...][data['/avalanche/file'][idx]]
    stresses = data['/avalanche/sigd'][idx]
    sig0 = float(data['/normalisation/sig0'][...])

for stress, inc, file in zip(stresses, incs, files):

    outfilename = '{0:s}_inc={1:d}.hdf5'.format(file.split('.hdf5')[0], inc)

    with h5py.File(os.path.join(dbase, file), 'r') as data:

        with h5py.File(outfilename, 'w') as output:

            for key in keys:
                output[key] = data[key][...]

            output['/disp/0'] = data['disp'][str(inc)][...]

            dset = output.create_dataset('/stored', (1, ), maxshape=(None, ), dtype=np.int)
            dset[0] = 0

            dset = output.create_dataset('/sigd', (1, ), maxshape=(None, ), dtype=np.float)
            dset[0] = stress * sig0

            dset = output.create_dataset('/t', (1, ), maxshape=(None, ), dtype=np.float)
            dset[0] = float(data['/t'][inc])



