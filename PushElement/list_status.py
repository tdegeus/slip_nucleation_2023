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
        if 'corrupt' in data['meta']:
            return not int(data['/meta/corrupt'][...])
        return int(data['/meta/completed'][...])


def isCompleted(filename):
    with h5py.File(filename, 'r') as data:
        if 'completed' not in data:
            return False
        if 'finished' in data['completed']:
            return int(data['/completed/finished'][...])


def hasRun(filename):
    with h5py.File(filename, 'r') as data:
        if 'stored' in data:
            if data['stored'].size > 0:
                return True
    return False


def getStored(filename):
    with h5py.File(filename, 'r') as data:
        if 'stored' not in data:
            return []
        return data['stored'][1:]


def getOnDisk(eventfiles):
    stored = [int(file.split('_push=')[1].split('.hdf5')[0]) for file in eventfiles]
    return np.sort(np.array(stored))


def toPushName(basename, push):
    return basename.replace('.hdf5', '') + '_push={0:d}.hdf5'.format(push)


args = docopt.docopt(__doc__, version='universal')

output = {
    'dirname' : os.path.abspath(args['<dirname>']),
    'completed_base' : [],
    'completed_event' : [],
    'partial_base' : [],
    'partial_event' : [],
    'error' : [],
    'new' : [],
}

files = sorted(list(filter(None, subprocess.check_output(
    "find {0:s} -iname '*.hdf5'".format(args['<dirname>']), shell=True).decode('utf-8').split('\n'))))

files = [os.path.relpath(file) for file in files]

basefiles = {file: [] for file in files if len(file.split('push')) == 1}
eventfiles = [file for file in files if len(file.split('push')) > 1]
output['error'] = [file for file in eventfiles if not eventIsCompleted(file)]
eventfiles = [file for file in eventfiles if file not in output['error']]

print('Basic error:', output['error'])

for file in eventfiles:
    basename = file.split('_push')[0]
    basefiles[basename + '.hdf5'] += [file]

for file in basefiles:

    ondisk = getOnDisk(basefiles[file])

    if len(ondisk) > 0:

        stored = getStored(file)
        completed = isCompleted(file)

        print('hasRun', file, completed, stored, ondisk)

        if completed and np.array_equal(ondisk, stored):
            output['completed_base'] += [file]
            output['completed_event'] += basefiles[file]
        elif not completed and np.all(np.isin(stored, ondisk)):
            output['partial_base'] += [file]
            output['partial_event'] += basefiles[file]
        else:
            output['error'] += [file] + basefiles[file]

    else:

        print('new', file)

        output['new'] += [file] + basefiles[file]

with open(args['--output'], 'w') as file:
    documents = yaml.dump(output, file)

