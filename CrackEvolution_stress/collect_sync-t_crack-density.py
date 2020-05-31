r'''
For avalanches synchronised at avalanche area `A`,
compute how dense a crack is on average: how many of the cracks yielded at a certain position
along the crack.

Usage:
    collect_sync-A_crack-density.py [options] <files>...

Arguments:
    <files>     Files from which to collect data.

Options:
    -o, --output=<N>    Output file. [default: output.hdf5]
    -i, --info=<N>      Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
    -f, --force         Overwrite existing output-file.
    -h, --help          Print help.
'''

import os
import sys
import docopt
import click
import h5py
import numpy as np

# ==================================================================================================
# horizontal shift
# ==================================================================================================

def getRenumIndex(old, new, N):
    idx = np.tile(np.arange(N), (3))
    return idx[old+N-new: old+2*N-new]

# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args['<files>']
output = args['--output']
info = args['--info']

for file in files:
    if not os.path.isfile(file):
        raise IOError('"{0:s}" does not exist'.format(file))

if not args['--force']:
    if os.path.isfile(output):
        print('"{0:s}" exists'.format(output))
        if not click.confirm('Proceed?'):
            sys.exit(1)

# ==================================================================================================
# get constants
# ==================================================================================================

with h5py.File(info, 'r') as data:
    dt = data['/normalisation/dt'][...]
    t0 = data['/normalisation/t0'][...]
    nx = int(data['/normalisation/N'][...])
    mid = int((nx - nx % 2) / 2)

# ==================================================================================================
# build histogram
# ==================================================================================================

niter = 100000
count = np.zeros((niter, nx), dtype='int')  # [t, r]
norm = np.zeros((niter), dtype='int')  # [t]

for file in files:

    cracked = np.zeros((niter, nx), dtype='int')  # [A, r]

    with h5py.File(file, 'r') as data:

        A = data["/sync-A/stored"][...]
        idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(A))][...]
        iiter = data['/sync-A/global/iiter'][...]
        a_n = 0

        if A[-1] != nx:
            print('Skipping {0:s}'.format(file))
            continue
        else:
            print('Reading {0:s}'.format(file))

        for a in A:

            i = iiter[a_n]
            j = iiter[a]
            if i >= niter or i >= niter:
                print('Range exceeded {0:s}'.format(file))
                break
            idx = data['/sync-A/plastic/{0:d}/idx'.format(a)][...]
            cracked[i:j, :] = (idx != idx0).astype(np.int)
            norm[i:j] += 1
            a_n = a

        # center
        a = np.argmin(np.abs(np.sum(cracked, axis=1) - mid))
        icell = np.argwhere(cracked[a, :]).ravel()
        icell[icell > mid] -= nx
        center = np.mean(icell)
        renum = getRenumIndex(int(center), 0, nx)
        count += cracked[:, renum]

# ==================================================================================================
# save data
# ==================================================================================================

with h5py.File(output, 'w') as data:

    data['/t'] = np.arange(niter) * dt / t0
    data['/P'] = count / np.where(norm > 0, norm, 1).reshape(-1, 1)
    data['/norm'] = norm

    data['/t'].attrs['desc'] = 'Time at which at which /P is stored'
    data['/P'].attrs['desc'] = 'Probability that a block yielded: realisations are centered before averaging'
    data['/P'].attrs['shape'] = '[len(/t), N] or [t, x]'
    data['/norm'].attrs['desc'] = 'Number of measurements per A'
