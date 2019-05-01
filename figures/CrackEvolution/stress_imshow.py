import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# ==================================================================================================

from definitions import *

# ==================================================================================================

if True:

  velocity = {
    'stress=0d6' : 0.7482585623998137,
    'stress=1d6' : 1.315656157225776,
    'stress=2d6' : 1.6000690643891013,
    'stress=3d6' : 1.7860898113087171,
    'stress=4d6' : 1.91797578256221,
    'stress=5d6' : 2.0091308317549643,
    'stress=6d6' : 2.09902506699992,
  }

  clim = {
    'stress=0d6' : 0.20,
    'stress=1d6' : 0.23,
    'stress=2d6' : 0.27,
    'stress=3d6' : 0.30,
    'stress=4d6' : 0.33,
    'stress=5d6' : 0.37,
    'stress=6d6' : 0.40,
  }

  nx = 'nx=3^6x2'

  for stress in ['stress=0d6', 'stress=1d6', 'stress=2d6', 'stress=3d6', 'stress=4d6', 'stress=5d6', 'stress=6d6']:

    data = h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element.hdf5'), 'r')

    sig_eq = data['sig_eq/700'][...].reshape(-1, num_nx(nx))

    data.close()

    fig, ax = plt.subplots()

    ax.imshow(sig_eq, clim=[0.0, clim[stress]], cmap='bone_r')

    if velocity[stress] > 1.:

      theta = np.arcsin(1./velocity[stress])

      dx = 300
      dy = np.tan(theta) * dx

      ax.plot(350 - np.array([0, dx]), int(sig_eq.shape[0]/2.) + np.array([0, dy]), c='w')
      ax.plot(350 - np.array([0, dx]), int(sig_eq.shape[0]/2.) - np.array([0, dy]), c='w')

    plt.axis('off')

    plt.savefig('stress_imshow/{0:s}_A=700.png'.format(stress))

# --------------------------------------------------------------------------------------------------

if True:

  nx = 'nx=3^6x2'

  for stress in ['stress=6d6']:

    data = h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-A_element.hdf5'), 'r')

    for A in sorted(data['/sig_eq']):

      sig_eq = data['sig_eq'][A][...].reshape(-1, num_nx(nx))

      fig, ax = plt.subplots()

      ax.imshow(sig_eq, clim=[0.0, 0.6], cmap='bone_r')

      plt.axis('off')

      plt.savefig('stress_imshow/{0:s}/sync-A/A={1:04d}.png'.format(stress, int(A)))
      plt.close()

    data.close()

# --------------------------------------------------------------------------------------------------

if True:

  nx = 'nx=3^6x2'

  for stress in ['stress=6d6']:

    data = h5py.File(path(key='CrackEvolution_stress', nx=nx, stress=stress, fname='data_sync-t_element.hdf5'), 'r')

    for t in sorted(data['/sig_eq']):

      sig_eq = data['sig_eq'][t][...].reshape(-1, num_nx(nx))

      fig, ax = plt.subplots()

      ax.imshow(sig_eq, clim=[0.0, 0.6], cmap='bone_r')

      plt.axis('off')

      plt.savefig('stress_imshow/{0:s}/sync-t/t={1:04d}.png'.format(stress, int(t)))
      plt.close()

    data.close()


