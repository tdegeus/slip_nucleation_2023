'''list_status
    List the status of simulations.

Usage:
    list_status [options] <dirname>

Arguments:
    <dirname>   Directory name where the simulations are stored.

Options:
    -o, --output=N      Output file [default: list_status.yaml]
    -h, --help          Show help.
        --version       Show version.
'''

import h5py
import subprocess
import os
import numpy as np
import docopt
import yaml


def eventIsCompleted(filename):
    with h5py.File(filename, 'r') as data:
        if 'meta' not in data:
            return False
        if 'completed' not in data['meta']:
            return False
        return int(data['/meta/completed'][...])


def isCompleted(filename):
    with h5py.File(filename, 'r') as data:
        if 'completed' not in data:
            return False
        return int(data['completed'][...])


def hasRun(filename):
    with h5py.File(filename, 'r') as data:
        if np.max(data['stored']) > 0:
            return True


def getStored(filename):
    with h5py.File(filename, 'r') as data:
        stored = data['stored'][...]
        for i in stored:
            if str(i) not in data['disp']:
                return 0
            if data['disp'][str(i)].shape != data['coor'].shape:
                return 0
    return stored[1:]


def getOnDisk(eventfiles):
    stored = [int(file.split('_ipush=')[1].split('.hdf5')[0]) for file in eventfiles]
    return np.sort(np.array(stored))


def toPushName(basename, ipush):
    return basename.replace('.hdf5', '') + '_ipush={0:d}.hdf5'.format(ipush)


args = docopt.docopt(__doc__, version='universal')

files = sorted(list(filter(None, subprocess.check_output(
    "find {0:s} -iname '*.hdf5'".format(args['<dirname>']), shell=True).decode('utf-8').split('\n'))))

files = [os.path.relpath(file) for file in files]

basefiles = {file: [] for file in files if len(file.split('ipush')) == 1}
eventfiles = [file for file in files if len(file.split('ipush')) > 1]

for file in eventfiles:
    basename = file.split('_ipush')[0]
    basefiles[basename + '.hdf5'] += [file]

output = {
    'dirname' : os.path.abspath(args['<dirname>']),
    'completed_base' : [],
    'completed_event' : [],
    'partial_base' : [],
    'partial_event' : [],
    'remove_event' : [],
    'error' : [],
    'new' : [],
}

for file in basefiles:

    if hasRun(file):

        stored = getStored(file)
        ondisk = getOnDisk(basefiles[file])
        completed = isCompleted(file)

        if completed and np.array_equal(ondisk, stored):
            output['completed_base'] += [file]
            output['completed_event'] += basefiles[file]
        elif not completed and np.all(np.isin(stored, ondisk)):
            output['partial_base'] += [file]
            output['partial_event'] += basefiles[file]
        elif completed and np.all(np.isin(stored, ondisk)):
            copy = [toPushName(file, i) for i in stored]
            remove = [f for f in basefiles[file] if f not in copy]
            for f in remove:
                if eventIsCompleted(f):
                    raise IOError('This should never happen')
            output['remove_event'] += remove
            output['completed_base'] += [file]
            output['completed_event'] += copy
        else:
            output['error'] += [file] + basefiles[file]

    else:

        output['new'] += [file] + basefiles[file]

with open(args['--output'], 'w') as file:
    documents = yaml.dump(output, file)

