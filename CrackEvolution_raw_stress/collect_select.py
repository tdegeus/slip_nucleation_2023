r'''
    Select part of the data for future processing.

Usage:
    collect_forces.py [options] <source> <output-dir>

Arguments:
    <source>        Files from which to collect data.
    <output-dir>    Directory to store selected data.

Options:
    -f, --force     Overwrite existing output-file.
    -h, --help      Print help.
'''

import os
import sys
import docopt
import click
import h5py
import numpy as np
import GooseHDF5 as g5

args = docopt.docopt(__doc__)

source = args['<source>']
outdir = args['<output-dir>']
output = os.path.join(outdir, source)

if not os.path.isfile(source):
    raise IOError('"{0:s}" does not exist'.format(source))

if not args['--force']:
    if not os.path.isdir(outdir):
        print('mkdir -p "{0:s}"'.format(outdir))
        if not click.confirm('Proceed?'):
            sys.exit(1)
        os.makedirs(outdir)

if not args['--force']:
    if os.path.isfile(output):
        print('"{0:s}" exists'.format(output))
        if not click.confirm('Proceed?'):
            sys.exit(1)

with h5py.File(output, 'w') as out:

    with h5py.File(source, 'r') as data:

        plastic = data["/meta/plastic"][...]
        N = len(plastic)

        A = np.sort((N - np.arange((N - N % 100) / 100 + 1) * 100).astype(np.int64))
        A = A[np.in1d(A, np.sort(data["/sync-A/stored"][...]))]

        out["/sync-A/stored"] = A
        out["/sync-A/global/iiter"] = data["/sync-A/global/iiter"][...][A]

        for a in A:
            out["/sync-A/{0:d}/u".format(a)] = data["/sync-A/{0:d}/u".format(a)][...]
            out["/sync-A/{0:d}/v".format(a)] = data["/sync-A/{0:d}/v".format(a)][...]

        T = np.sort(data["/sync-t/stored"][...])[::100]

        out["/sync-t/stored"] = T
        out["/sync-t/global/iiter"] = data["/sync-t/global/iiter"][...][T]

        for t in T:
            out["/sync-t/{0:d}/u".format(t)] = data["/sync-t/{0:d}/u".format(t)][...]
            out["/sync-t/{0:d}/v".format(t)] = data["/sync-t/{0:d}/v".format(t)][...]

        g5.copydatasets(data, out, list(g5.getdatasets(data, "/git")))
        g5.copydatasets(data, out, list(g5.getdatasets(data, "/meta")))
