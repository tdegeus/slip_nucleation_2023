import subprocess
import h5py
import os
import tqdm
from shelephant.yaml import dump

def is_completed(file):
    with h5py.File(file, 'r') as data:
        if 'completed' in data:
            return data['/completed'][...]
    return False

def is_completed_push(file):
    with h5py.File(file, 'r') as data:
        if 'meta' in data:
            if 'completed' in data['meta']:
                return data['/meta/completed'][...]
    return False

files = sorted(list(filter(None, subprocess.check_output(
    "find . -maxdepth 1 -iname 'id*.hdf5'", shell=True).decode('utf-8').split('\n'))))

sims = [file for file in files if len(file.split('push')) == 1]
pushes = [file for file in files if len(file.split('push')) > 1]

sims = [os.path.relpath(file) for file in tqdm.tqdm(sims) if is_completed(file)]
pushes = [os.path.relpath(file) for file in tqdm.tqdm(pushes) if is_completed_push(file)]

ret = {
    'sims': sims,
    'pushes': pushes,
}

dump('completed.yaml', ret)
