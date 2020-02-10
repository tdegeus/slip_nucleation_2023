r'''
For avalanches synchronised at avalanche area `A`,
build a histogram of the connectedness of the avalanche,
given by the distance of the next block that yielded as least once.

Usage:
  collect_sync-A_connect.py [options] <files>...

Arguments:
  <files>   Files from which to collect data.

Options:
  -o, --output=<N>  Output file. [default: output.hdf5]
'''

import os
import sys
import docopt
import click
import h5py
import numpy as np

# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args['<files>']
output = args['--output']

for file in files:
  if not os.path.isfile(file):
    raise IOError('"{0:s}" does not exist'.format(file))

if os.path.isfile(output):
  print('"{0:s}" exists'.format(output))
  if not click.confirm('Proceed?'):
    sys.exit(1)

# ==================================================================================================
# get constants
# ==================================================================================================

with h5py.File(files[0], 'r') as data:
  plastic = data['/meta/plastic'][...]
  nx      = len(plastic)

# ==================================================================================================
# build histogram
# ==================================================================================================

count = np.zeros((nx + 1, nx + 1), dtype='int') # (A, x)
norm = np.zeros((nx + 1), dtype='int') # (A)

for file in files:

  print(file)

  with h5py.File(file, 'r') as data:

    # get stored "A"
    A = data["/sync-A/stored"][...]

    # get the reference configuration
    idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(A))][...]

    # loop over cracks
    for a in A:

      # get current configuration
      idx = data['/sync-A/plastic/{0:d}/idx'.format(a)][...]

      # indices of blocks where yielding took place
      icell = np.argwhere(idx0 != idx).ravel()

      if len(icell) == 0:
        continue

      icell = np.pad(icell, [1, 0], 'wrap')
      icell[0] -= nx

      n, _ = np.histogram(np.diff(icell), bins=count.shape[1], range=(0, nx+1), density=False)
      count[a, :] += n
      norm[a] += 1

# ==================================================================================================
# save data
# ==================================================================================================

with h5py.File(output, 'w') as data:

  data['/A'] = np.arange(nx + 1)
  data['/norm'] = norm
  data['/count'] = count

