r'''
For avalanches synchronised at avalanche area `A`,
compute the distance between a new yielding event and the largest connected block

Usage:
    collect_sync-A_connect.py [options] <files>...

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
import GooseEYE as eye

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

count_clusters = np.zeros((nx + 1), dtype='int')  # [A]
count_distance = np.zeros((nx + 1, nx + 1), dtype='int')  # [A, distance]
count_size     = np.zeros((nx + 1, nx + 1), dtype='int')  # [A, distance]
count_maxsize  = np.zeros((nx + 1), dtype='int')  # [A]
norm_clusters  = np.zeros((nx + 1), dtype='int')  # [A]
norm_distance  = np.zeros((nx + 1), dtype='int')  # [A]
norm_size      = np.zeros((nx + 1), dtype='int')  # [A]

for file in files:

    print(file)

    with h5py.File(file, 'r') as data:

        A = data["/sync-A/stored"][...]
        idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(A))][...]
        cracked = np.zeros(idx0.shape, dtype='int')

        for a in A:

            idx = data['/sync-A/plastic/{0:d}/idx'.format(a)][...]
            yielded = (idx != idx0).astype(np.int)
            active = np.where(yielded - cracked == 1, 1, 0)

            if np.sum(cracked) == nx:
                continue

            if np.sum(cracked) > 0:

                clusters = eye.Clusters(cracked, periodic=True)
                labels = clusters.labels()
                sizes = clusters.sizes()
                n, _ = np.histogram(sizes[1:], bins=(nx + 1), range=(0, nx + 1), density=False)
                count_clusters[a] += sizes.size - 1
                count_size[a, :] += n
                count_maxsize[a] += np.max(sizes[1:])
                norm_clusters[a] += 1

            if np.sum(cracked) > 0 and np.sum(active) > 0:

                l = np.argmax(sizes[1:]) + 1
                icell = np.argwhere(labels == l)
                icell[icell > mid] -= nx
                left = np.min(icell)
                right = np.max(icell)

                iactive = np.argwhere(active)[0]
                iactive[iactive > mid] -= nx
                d = np.where(iactive < left, left - iactive, iactive - right)

                count_distance[a, d] += 1
                norm_distance[a] += 1

            cracked = np.where(yielded + cracked, 1, 0)

# ==================================================================================================
# save data
# ==================================================================================================

with h5py.File(output, 'w') as data:

    data['/A'] = np.arange(nx + 1)
    data['/clusters'] = count_clusters / np.where(norm_clusters > 0, norm_clusters, 1)
    data['/distance'] = count_distance / np.where(norm_distance > 0, norm_distance, 1).reshape(-1, 1)
    data['/size']     = count_size     / np.where(norm_clusters > 0, norm_clusters, 1).reshape(-1, 1)
    data['/maxsize']  = count_maxsize  / np.where(norm_clusters > 0, norm_clusters, 1)
    data['/norm_clusters'] = norm_clusters
    data['/norm_distance'] = norm_distance
    data['/norm_size']     = norm_clusters
    data['/norm_maxsize']  = norm_clusters
