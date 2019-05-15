import sys, os, re, subprocess, shutil, h5py

import numpy as np

files = sorted(list(filter(None, subprocess.check_output("find . -iname '*id*.hdf5'",shell=True).decode('utf-8').split('\n'))))

for file in files:

  data = h5py.File(file, 'r')

  if 'completed' not in data['meta']:
    print(file, 'Error: "/meta/completed" not found.')
    continue

  if 'sync-t' not in data:
    print(file, 'Error: "/sync-t/..." not found.')
    continue

  inc = data['/sync-A/stored'][...]

  idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(inc))][...]
  idx  = data['/sync-A/plastic/{0:d}/idx'.format(np.max(inc))][...]

  if np.sum(idx0 != idx) != len(idx):
    print(file, 'Error: "/sync-A/..." not system spanning.')
    continue

  inc = data['/sync-t/stored'][...]

  idx = data['/sync-t/plastic/{0:d}/idx'.format(np.max(inc))][...]

  if np.sum(idx0 != idx) != len(idx):
    print(file, 'Error: "/sync-t/..." not system spanning.')
    continue

  print(file)

  data.close()

