import h5py
import os
import tqdm
import argparse
from shelephant.yaml import dump

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='*', type=str)
args = parser.parse_args()
assert np.all([os.path.isfile(file) for file in args.files])

def is_completed(file):
    with h5py.File(file, 'r') as data:
        try:
            return data['/meta/EventEvolution_stress/completed'][...]
        except:
            return False

completed = []
partial = []

for file in tqdm.tqdm(args.files):
    if is_completed(file):
        completed += [file]
    else:
        partial += [file]

ret = {
    'completed': completed,
    'failed': failed,
}

dump('completed.yaml', ret)
