import h5py
import numpy as np
from scipy.io.wavfile import write

with h5py.File('id=747_elem=0219_incc=047.hdf5', 'r') as f:
  data = f['/sig_xy/close'][...]


data = data[:240000]

data -= np.min(data)
data /= np.max(data)
data *= 100.
data -= data[0]

data = np.hstack((data, data[-1] * np.cos(np.linspace(0, np.pi/2., 100000))))

write('test.wav', int(data.size/10.), data)


import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

fig, ax = plt.subplots()

ax.plot(np.arange(data.size), data)

plt.show()
