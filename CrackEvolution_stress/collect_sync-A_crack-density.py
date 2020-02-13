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

with h5py.File(files[0], 'r') as data:
    plastic = data['/meta/plastic'][...]
    nx = len(plastic)

if nx % 2 == 0:
    mid = nx / 2
else:
    mid = (nx - 1) / 2

# ==================================================================================================
# build histogram
# ==================================================================================================

count = np.zeros((nx + 1, nx), dtype='int')  # [A, r]
norm = np.zeros((nx + 1), dtype='int')  # [A]

for file in files:

    print(file)

    with h5py.File(file, 'r') as data:

        A = data["/sync-A/stored"][...]
        idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(A))][...]
        cracked = np.zeros(idx0.shape, dtype='int')

        for a in A:

            idx = data['/sync-A/plastic/{0:d}/idx'.format(a)][...]
            yielded = (idx != idx0).astype(np.int)
            cracked = np.where(yielded + cracked, 1, 0)

            icell = np.argwhere(cracked).ravel()

            if len(icell) == 0:
                continue

            icell[icell > mid] -= nx
            center = np.mean(icell)
            renum = getRenumIndex(int(center), 0, nx)
            count[a, :] += cracked[renum]
            norm[a] += 1

# ==================================================================================================
# save data
# ==================================================================================================

with h5py.File(output, 'w') as data:

    data['/A'] = np.arange(nx + 1)
    data['/P'] = count / np.where(norm > 0, norm, 1).reshape(-1, 1)
    data['/norm'] = norm
