import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

plt.style.use(['goose', 'goose-latex'])

# --------------------------------------------------------------------------------------------------

from definitions import *

rho  = 1.
h    = np.pi
G    = 1.0
mu   = G / 2.
cs   = np.sqrt(mu/rho)

# --------------------------------------------------------------------------------------------------

def hdf2dict(data):

  out = {}

  for file in data:

    out[file] = {}

    for field in data[file]:

      out[file][field] = data[file][field][...]

  return out

# --------------------------------------------------------------------------------------------------

def average(nx, stress, Ac):

  # ensemble info
  # -------------

  data = h5py.File(path(nx=nx, fname='EnsembleInfo.hdf5'), 'r')

  N     = int(data['/normalisation/N'][...])
  sig0  = float(data['/normalisation/sigy'][...])
  sig_n = float(data['/averages/sigd_top'   ][...])
  sig_c = float(data['/averages/sigd_bottom'][...])

  data.close()

  # meta-data
  # ---------

  data = h5py.File(os.path.join('meta-data', '{nx:s}_{stress:s}.hdf5'.format(nx=nx, stress=stress)), 'r')

  meta = hdf2dict(data)

  data.close()

  # raw results & average
  # ---------------------

  # open raw-results
  data = h5py.File(path(nx=nx, fname='TimeEvolution_{stress:s}.hdf5'.format(stress=stress)), 'r')

  # allocate averages
  out = {
    'sigbar'   : [],
    'sigeq:-2' : [],
    'sigeq:-1' : [],
    'sigeq:+0' : [],
    'sigeq:+1' : [],
    'sigeq:+2' : [],
  }

  # loop over measurements
  for file in sorted([f for f in data]):

    # - check if a crack of "Ac" is reached
    if not np.any(meta[file]['A'] >= Ac):
      continue

    # - find the increment for which the "crack" is first bigger than "Ac"
    inc = np.argmin(np.abs(meta[file]['A'] - Ac))

    # - store macroscopic stress
    out['sigbar'] += [data[file]['sigbar'][...][inc]]

    # - read stress tensor
    Sig = data[file]['Sig'][str(inc)][...]

    # - decompose stress
    sig_xx = Sig[:, 0, 0].reshape(-1, N)
    sig_xy = Sig[:, 0, 1].reshape(-1, N)
    sig_yy = Sig[:, 1, 1].reshape(-1, N)

    # - compute averages
    sig_xx = np.mean(sig_xx, axis=1)
    sig_xy = np.mean(sig_xy, axis=1)
    sig_yy = np.mean(sig_yy, axis=1)

    # - hydrostatic stress
    sig_m = (sig_xx + sig_yy) / 2.

    # - deviatoric stress
    sigd_xx = sig_xx - sig_m
    sigd_xy = sig_xy
    sigd_yy = sig_yy - sig_m

    # - equivalent stress
    sig_eq = np.sqrt(2.0 * (sigd_xx**2.0 + sigd_yy**2.0 + 2.0 * sigd_xy**2.0))

    # - store
    out['sigeq:-2'] += [sig_eq[0]]
    out['sigeq:-1'] += [sig_eq[1]]
    out['sigeq:+0'] += [sig_eq[2]]
    out['sigeq:+1'] += [sig_eq[3]]
    out['sigeq:+2'] += [sig_eq[4]]

  # close raw-data
  data.close()

  return (\
    {key: np.mean(np.array(out[key]) / sig0) for key in out},
    {key: np.std (np.array(out[key]) / sig0) for key in out},
    sig_c,
    sig_n,
  )

# --------------------------------------------------------------------------------------------------

if not os.path.isdir('stress'): os.makedirs('stress')

for nx in list_nx()[::-1]:

  N = num_nx(nx)

  fig, ax = gplt.subplots(scale_x=1.2)

  ax.set_ylim([0.0, 0.6])
  ax.set_xlim([0, N])

  ax.set_xlabel(r'$A$')
  ax.set_ylabel(r'$\sigma$')

  for stress in list_stress()[::-1]:

    if nx == 'nx=3^6': Ac_list = np.linspace(50,  700, 15).astype(np.int)
    else             : Ac_list = np.linspace(50, 1400, 29).astype(np.int)

    for i, Ac in enumerate(Ac_list):

      mean, std, sig_c, sig_n = average(nx, stress, Ac)

      if i == 0:
        Mean = {key:[] for key in mean}
        Std  = {key:[] for key in std }

      for key in Mean:
        Mean[key] += [mean[key]]
        Std [key] += [std [key]]

    ax.errorbar(Ac_list, Mean['sigbar'  ], yerr=Std['sigbar'  ], **color_stress(nx, stress), **label_stress(stress), ls='-' , zorder=0)
    ax.errorbar(Ac_list, Mean['sigeq:+0'], yerr=Std['sigeq:+0'], **color_stress(nx, stress)                        , ls='--', zorder=0)

  ax.plot(ax.get_xlim(), sig_c * np.ones(2), c='k', ls='--')
  ax.plot(ax.get_xlim(), sig_n * np.ones(2), c='k', ls='-.')

  # Shrink current axis by 20%
  plt.subplots_adjust(right=0.8)

  # Put a legend to the right of the current axis
  legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.savefig('stress/{nx:s}.pdf'.format(nx=nx))
  plt.close()

