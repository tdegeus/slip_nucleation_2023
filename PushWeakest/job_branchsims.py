
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
    '/t',
    '/uuid',
]

with h5py.File(os.path.join(dbase, 'EnsembleInfo.hdf5'), 'r') as data:

    A = data['/avalanche/A'][...]
    idx = np.argwhere(A == N).ravel()
    inc = data['/avalanche/inc'][idx] + 1
    files = data['/files'][...][data['/avalanche/file'][idx]]

    # for file, i in zip(files, inc):
    #     s = data['full'][file]['sigd'][...]
    #     i = int(i)
    #     print(s[i - 1], s[i], s[i + 1])

for i, f in zip(inc, files):

    filename = '{0:s}_inc={1:d}.hdf5'.format(f.split('.hdf5')[0], i)

    with h5py.File(os.path.join(dbase, f), 'r') as data:

        with h5py.File(filename, 'w') as output:

            for key in keys:
                output[key] = data[key][...]

            output['/disp/0'] = data['disp'][str(i)][...]

            dset = output.create_dataset('/stored', (1, ), maxshape=(None, ), dtype=np.int)
            dset[0] = 0



