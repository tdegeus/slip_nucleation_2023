import subprocess
import h5py
import os
import numpy as np

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

dbase = '../../../data/nx=3^6x2'
sourcedir = os.path.join(dbase, 'PushRecursiveQuasiThermal')

files = sorted(list(filter(None, subprocess.check_output(
    "find {0:s} -iname 'id*.hdf5'".format(sourcedir), shell=True).decode('utf-8').split('\n'))))

def is_completed(file):
    try:
        with h5py.File(file, 'r') as data:
            if 'completed' in data:
                return data['/completed'][...]
        return False
    except:
        return False

files = [file for file in files if is_completed(file)]

N = int(3 ** 6 * 2)
itrigger = [0, 100, 200, int(N / 2), int(N / 2) + 100, int(N / 2) + 200, N - 201, N - 101, N - 1]

for file in files:

    for i in itrigger:

        outfilename = '{0:s}_itrigger={1:d}.hdf5'.format(os.path.split(file.split('.hdf5')[0])[1], i)

        print(outfilename)

        with h5py.File(file, 'r') as data:

            with h5py.File(outfilename, 'w') as output:

                for key in keys:
                    output[key] = data[key][...]

                inc = data['/stored'][-1]
                output['/disp/0'] = data['disp'][str(inc)][...]
                output['/trigger/i'] = i

                dset = output.create_dataset('/stored', (1, ), maxshape=(None, ), dtype=np.int)
                dset[0] = 0

                dset = output.create_dataset('/sigd', (1, ), maxshape=(None, ), dtype=np.float)
                dset[0] = float(data['/sigd'][inc])

                dset = output.create_dataset('/t', (1, ), maxshape=(None, ), dtype=np.float)
                dset[0] = float(data['/t'][inc])
