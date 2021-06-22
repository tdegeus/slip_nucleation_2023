import subprocess
import h5py
import os
import tqdm
from shelephant.yaml import dump

def is_completed(file):
    with h5py.File(file, 'r') as data:
        try:
            return data['/meta/EventEvolution_stress/completed'][...]
        except:
            return False

files = sorted(list(filter(None, subprocess.check_output(
    "find . -maxdepth 1 -iname 'id*.hdf5'", shell=True).decode('utf-8').split('\n'))))

files = [os.path.relpath(file) for file in tqdm.tqdm(files) if is_completed(file)]

dump('completed.yaml', files)
