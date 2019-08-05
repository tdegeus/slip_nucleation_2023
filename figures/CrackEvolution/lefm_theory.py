import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

from matplotlib.patches import BoxStyle

plt.style.use(['goose', 'goose-latex'])

# --------------------------------------------------------------------------------------------------

kII = 1.0
sigma_c = 0.1

# --------------------------------------------------------------------------------------------------

for dy in [0, 4, 30, 60]:

  fig, axes = gplt.subplots(ncols=3)

  for ax in axes:
    ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

  axes[0].set_ylabel(r'$\sigma_{xx}$')
  axes[1].set_ylabel(r'$\sigma_{yy}$')
  axes[2].set_ylabel(r'$\sigma_{xy}$')

  for ax in axes:
    ax.set_xlim([-400, +400])

  axes[0].set_ylim([-0.1, +0.1])
  axes[1].set_ylim([-0.1, +0.1])
  axes[2].set_ylim([+0.0, +0.2])

  gplt.text(.05, .9, r'$\Delta y / h = %d$' % dy, units='relative', axis=axes[1], bbox=dict(edgecolor='black', boxstyle=BoxStyle("Round, pad=0.3"), facecolor='white'))

  # --

  dx = np.linspace(0, 400, 4 * 400 + 1)[1:]

  r = np.sqrt(dx**2.0 + dy**2.0)
  theta = np.arctan(dy / dx)

  sig_xx = -kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) * (2.0 + np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0))
  sig_yy = +kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) *        np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0)
  sig_xy = +kII / np.sqrt(2.0 * np.pi * r) * np.cos(theta / 2.0) * (1.0 - np.sin(theta / 2.0) * np.sin(3.0 * theta / 2.0))

  sig_xy += sigma_c

  axes[0].plot(dx, sig_xx, c='k')
  axes[1].plot(dx, sig_yy, c='k')
  axes[2].plot(dx, sig_xy, c='k')

  # --

  dx = np.linspace(0, 400, 4 * 400 + 1)[1:]

  r = np.sqrt(dx**2.0 + dy**2.0)
  theta = np.pi - np.arctan(dy / dx)

  sig_xx = -kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) * (2.0 + np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0))
  sig_yy = +kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) *        np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0)
  sig_xy = +kII / np.sqrt(2.0 * np.pi * r) * np.cos(theta / 2.0) * (1.0 - np.sin(theta / 2.0) * np.sin(3.0 * theta / 2.0))

  sig_xy += sigma_c

  axes[0].plot(-dx, sig_xx, c='k')
  axes[1].plot(-dx, sig_yy, c='k')
  axes[2].plot(-dx, sig_xy, c='k')

  # --

  gplt.savefig('lefm_theory/const-dy/dy={0:d}.pdf'.format(dy))
  plt.close()

# --------------------------------------------------------------------------------------------------

for theta, theta_name in zip([0., np.pi/4., np.pi/2.], ['theta=0', 'theta=pi-4', 'theta=pi-2']):

  if   theta_name == 'theta=0'   : label = r'$\theta = 0$'
  elif theta_name == 'theta=pi-4': label = r'$\theta = \pi / 4$'
  elif theta_name == 'theta=pi-2': label = r'$\theta = \pi / 2$'
  else: raise IOError('Unknown theta')

  fig, axes = gplt.subplots(ncols=3)

  for ax in axes:
    ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

  axes[0].set_ylabel(r'$\sigma_{xx}$')
  axes[1].set_ylabel(r'$\sigma_{yy}$')
  axes[2].set_ylabel(r'$\sigma_{xy}$')

  for ax in axes:
    ax.set_xlim([-500,500])

  axes[0].set_ylim([-0.1, +0.1])
  axes[1].set_ylim([-0.1, +0.1])
  axes[2].set_ylim([+0.0, +0.2])

  gplt.text(.05, .9, label, units='relative', axis=axes[1], bbox=dict(edgecolor='black', boxstyle=BoxStyle("Round, pad=0.3"), facecolor='white'))

  r = np.linspace(0, 500, 1001)[1:]

  sig_xx = -kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) * (2.0 + np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0))
  sig_yy = +kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) *        np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0)
  sig_xy = +kII / np.sqrt(2.0 * np.pi * r) * np.cos(theta / 2.0) * (1.0 - np.sin(theta / 2.0) * np.sin(3.0 * theta / 2.0))

  sig_xy += sigma_c

  axes[0].plot(r, sig_xx, c='k')
  axes[1].plot(r, sig_yy, c='k')
  axes[2].plot(r, sig_xy, c='k')

  theta += np.pi

  sig_xx = -kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) * (2.0 + np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0))
  sig_yy = +kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) *        np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0)
  sig_xy = +kII / np.sqrt(2.0 * np.pi * r) * np.cos(theta / 2.0) * (1.0 - np.sin(theta / 2.0) * np.sin(3.0 * theta / 2.0))

  sig_xy += sigma_c

  axes[0].plot(-r, sig_xx, c='k')
  axes[1].plot(-r, sig_yy, c='k')
  axes[2].plot(-r, sig_xy, c='k')

  gplt.savefig('lefm_theory/const-theta/{0:s}.pdf'.format(theta_name))
  plt.close()
