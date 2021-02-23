r'''
    Select part of the data for future processing.
    Common practice::

        shelephant_dump *.hdf5
        python collect_forces shelephant_dump.yaml reduced

Usage:
    collect_forces.py [options] <files.yaml> <output-root>

Arguments:
    <files.yaml>    Files from which to collect data.
    <output-root>   Directory to store selected data (file-structure preserved).

Options:
    -k, --key=N     Path in the YAML-file, separated by "/". [default: /]
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
import shelephant
import tqdm
from setuptools_scm import get_version

args = docopt.docopt(__doc__)

source = args['<files.yaml>']
key = list(filter(None, args['--key'].split('/')))
sources = shelephant.YamlGetItem(source, key)
root = args['<output-root>']
destinations = shelephant.PrefixPaths(root, sources)

shelephant.CheckAllIsFile(sources)
shelephant.OverWrite(destinations)
shelephant.MakeDirs(shelephant.DirNames(destinations))

for source, destination in zip(tqdm.tqdm(sources), destinations):

    with h5py.File(destination, 'w') as out:

        with h5py.File(source, 'r') as data:

            plastic = data["/meta/plastic"][...]
            N = len(plastic)

            A = np.sort((N - np.arange((N - N % 100) / 100 + 1) * 100).astype(np.int64))
            A = A[np.in1d(A, np.sort(data["/sync-A/stored"][...]))]

            out["/sync-A/stored"] = A
            out["/sync-A/global/iiter"] = data["/sync-A/global/iiter"][...]

            for a in A:
                out["/sync-A/{0:d}/u".format(a)] = data["/sync-A/{0:d}/u".format(a)][...]
                out["/sync-A/{0:d}/v".format(a)] = data["/sync-A/{0:d}/v".format(a)][...]

            T = np.sort(data["/sync-t/stored"][...])[::100]

            out["/sync-t/stored"] = T
            out["/sync-t/global/iiter"] = data["/sync-t/global/iiter"][...][T]

            for t in T:
                out["/sync-t/{0:d}/u".format(t)] = data["/sync-t/{0:d}/u".format(t)][...]
                out["/sync-t/{0:d}/v".format(t)] = data["/sync-t/{0:d}/v".format(t)][...]

            g5.copydatasets(data, out, list(g5.getdatasets(data, "/meta")))
            out["/meta/versions/CrackEvolution_raw_stress"] = data["/git/run"][...]
            out["/meta/versions/collect_select.py"] = get_version(root='..', relative_to=__file__)
