r'''
Collect velocities form data synchronised at avalanche area `A`

Usage:
    collect_sync-A_velocity.py [options] <files>...

Arguments:
    <files>     Files from which to collect data.

Options:
    -o, --output=<N>    Output file. [default: output.hdf5]
    -i, --info=<N>      Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
'''

import os
import sys
import docopt
import click
import h5py
import numpy as np
import GooseFEM as gf

# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args['<files>']
info = args['--info']
output = args['--output']

for file in files + [info]:
    if not os.path.isfile(file):
        raise IOError('"{0:s}" does not exist'.format(file))

if os.path.isfile(output):
    print('"{0:s}" exists'.format(output))
    if not click.confirm('Proceed?'):
        sys.exit(1)

# ==================================================================================================
# get normalisation
# ==================================================================================================

with h5py.File(info, 'r') as data:
  dt = data['/normalisation/dt'][...]
  t0 = data['/normalisation/t0'][...]
  sig0 = data['/normalisation/sig0'][...]
  nx = int(data['/normalisation/N'][...])

v = np.zeros((len(files)), dtype='float')

for ifile, file in enumerate(files):

    with h5py.File(file, 'r') as data:

        A = data["/sync-A/stored"][...].astype(np.int)

        if A[-1] != nx:
            print('Skipping {0:s}'.format(file))
            continue
        else:
            print('Reading {0:s}'.format(file))

        i0 = np.argmin(np.abs(A - 400))
        i1 = np.argmin(np.abs(A - 800))

        iiter = data["/sync-A/global/iiter"][...][A].astype(np.int)
        Dt = (iiter[i1] - iiter[i0]) * dt / t0
        Da = (A[i1] - A[i0])

        v[ifile] = Da / Dt

with h5py.File(output, 'w') as data:
    data['/v'] = v
