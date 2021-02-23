import subprocess
import h5py
import os
import tqdm
from shelephant import YamlDump

def is_completed(file):
    with h5py.File(file, 'r') as data:
        if 'meta' in data:
            if 'completed' in data['meta']:
                return data['/meta/completed'][...]
    return False

files = sorted(list(filter(None, subprocess.check_output(
    "find . -iname 'id*.hdf5'", shell=True).decode('utf-8').split('\n'))))

files = [os.path.relpath(file) for file in tqdm.tqdm(files) if is_completed(file)]

YamlDump('completed.yaml', files)
