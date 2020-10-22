
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

with h5py.File(os.path.join(dbase, 'AvalancheAfterPush_strain=00d10.hdf5'), 'r') as data:

    p_files = data['files'  ][...]
    p_file  = data['file'   ][...]
    p_elem  = data['element'][...]
    p_A     = data['A'      ][...]
    p_S     = data['S'      ][...]
    p_sig   = data['sigd0'  ][...]
    p_sigc  = data['sig_c'  ][...]
    p_incc  = data['inc_c'  ][...]

idx = np.argwhere(p_A == N).ravel()

p_file = p_file[idx]
p_elem = p_elem[idx]
p_A    = p_A   [idx]
p_S    = p_S   [idx]
p_sig  = p_sig [idx]
p_sigc = p_sigc[idx]
p_incc = p_incc[idx]

for i in range(len(p_file)):

    file = p_files[p_file[i]]
    inc = p_incc[i]
    element = p_elem[i]
    stress = p_sig[i]

    outfilename = '{0:s}_element={1:d}_inc={2:d}.hdf5'.format(file.split('.hdf5')[0], element, inc)

    print(outfilename, p_S[i])

    with h5py.File(os.path.join(dbase, file), 'r') as data:

        with h5py.File(outfilename, 'w') as output:

            for key in keys:
                output[key] = data[key][...]

            output['/push/element'] = element
            output['/push/inc'] = inc
            output['/disp/0'] = data['disp'][str(inc)][...]

            dset = output.create_dataset('/stored', (1, ), maxshape=(None, ), dtype=np.int)
            dset[0] = 0

            dset = output.create_dataset('/sigd', (1, ), maxshape=(None, ), dtype=np.float)
            dset[0] = stress

            dset = output.create_dataset('/t', (1, ), maxshape=(None, ), dtype=np.float)
            dset[0] = float(data['/t'][inc])



