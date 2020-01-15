import sys, os, re, subprocess, shutil, h5py

import matplotlib.pyplot as plt
import GooseMPL          as gplt
import numpy             as np

from matplotlib.patches import BoxStyle
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use(['goose', 'goose-latex'])

# --------------------------------------------------------------------------------------------------

kII = 1.0
sigma_c = 0.1
nu = 0.5

# --------------------------------------------------------------------------------------------------

x, y = np.meshgrid(np.linspace(-400, 400, 801), np.linspace(-400, 400, 801))

x = x.reshape(-1)
y = y.reshape(-1)

r = np.sqrt(x**2.0 + y**2.0)
theta = np.empty(r.shape)

idx = np.where(np.all([x == 0, y >= 0], axis=0))[0]
theta[idx] = +np.pi / 2.0

idx = np.where(np.all([x == 0, y <= 0], axis=0))[0]
theta[idx] = -np.pi / 2.0

idx = np.where(np.all([x <= 0, y == 0], axis=0))[0]
theta[idx] = np.pi

idx = np.where(np.all([x >= 0, y == 0], axis=0))[0]
theta[idx] = 0.0

idx = np.where(np.all([x > 0, y > 0], axis=0))[0]
theta[idx] = np.arctan(y[idx] / x[idx])

idx = np.where(np.all([x < 0, y > 0], axis=0))[0]
theta[idx] = np.pi - np.arctan(y[idx] / -x[idx])

idx = np.where(np.all([x > 0, y < 0], axis=0))[0]
theta[idx] = - np.arctan(-y[idx] / x[idx])

idx = np.where(np.all([x < 0, y < 0], axis=0))[0]
theta[idx] = -np.pi + np.arctan(-y[idx] / -x[idx])

r = r.reshape(801, 801)
theta = theta.reshape(801, 801)

# ---

r_orig = np.array(r, copy=True)
r[r == 0.0] = 1.0

sig_xx  = -kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) * (2.0 + np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0))
sig_yy  = +kII / np.sqrt(2.0 * np.pi * r) * np.sin(theta / 2.0) *        np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0)
sig_xy  = +kII / np.sqrt(2.0 * np.pi * r) * np.cos(theta / 2.0) * (1.0 - np.sin(theta / 2.0) * np.sin(3.0 * theta / 2.0))
sig_zz  = nu * (sig_xx + sig_yy)
sigm    = (sig_xx + sig_yy + sig_zz) / 3.0
sigd_xx = sig_xx - sigm
sigd_yy = sig_yy - sigm
sigd_zz = sig_zz - sigm
sigd_xy = sig_xy
sigeq   = np.sqrt(0.5 * (sigd_xx**2.0 + sigd_yy**2.0 + sigd_zz**2.0 + 2.0 * sigd_xy**2.0))

r = r_orig

# ---

fig, axes = plt.subplots(ncols=2)

axes[0].set_title(r'$r$')
axes[1].set_title(r'$\theta$')

ax   = axes[0]
im   = ax.imshow(r, cmap='jet', clim=[0, np.max(r)])
div  = make_axes_locatable(ax)
cax  = div.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)

ax   = axes[1]
im   = ax.imshow(theta, cmap='RdBu_r', clim=[-np.pi, np.pi])
div  = make_axes_locatable(ax)
cax  = div.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)

for ax in axes:
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

gplt.savefig('lefm_theory/stress-field/theta-r.pdf', bbox_inches='tight')

# ----

fig, axes = plt.subplots(ncols=3)

axes[0].set_title(r'$\sigma_{xx}$')
axes[1].set_title(r'$\sigma_{yy}$')
axes[2].set_title(r'$\sigma_{xy}$')

ax   = axes[0]
im   = ax.imshow(sig_xx, cmap='RdBu_r', clim=[-0.1, +0.1])
div  = make_axes_locatable(ax)
cax  = div.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)

ax   = axes[1]
im   = ax.imshow(sig_yy, cmap='RdBu_r', clim=[-0.1, +0.1])
div  = make_axes_locatable(ax)
cax  = div.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)

ax   = axes[2]
im   = ax.imshow(sig_xy, cmap='Reds', clim=[0.0, +0.1])
div  = make_axes_locatable(ax)
cax  = div.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)

for ax in axes:
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

gplt.savefig('lefm_theory/stress-field/stress.pdf', bbox_inches='tight')

# ----

fig, axes = plt.subplots(ncols=2)

axes[0].set_title(r'$\sigma_{m}$')
axes[1].set_title(r'$\sigma_{eq}$')

ax   = axes[0]
im   = ax.imshow(sigm, cmap='RdBu_r', clim=[-0.1, +0.1])
div  = make_axes_locatable(ax)
cax  = div.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)

ax   = axes[1]
im   = ax.imshow(sigeq, cmap='bone_r', clim=[0.0, 0.1])
div  = make_axes_locatable(ax)
cax  = div.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)

for ax in axes:
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

gplt.savefig('lefm_theory/stress-field/stress-equivalent.pdf', bbox_inches='tight')

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

for theta, theta_name in zip([0., np.pi/4., np.pi/2., -np.pi/4., -np.pi/2.], ['theta=0', 'theta=pi?4', 'theta=pi?2', 'theta=-pi?4', 'theta=-pi?2']):

  if   theta_name == 'theta=0'    : label = r'$\theta = 0$'
  elif theta_name == 'theta=pi?4' : label = r'$\theta = \pi / 4$'
  elif theta_name == 'theta=pi?2' : label = r'$\theta = \pi / 2$'
  elif theta_name == 'theta=-pi?4': label = r'$\theta = - \pi / 4$'
  elif theta_name == 'theta=-pi?2': label = r'$\theta = - \pi / 2$'
  else: raise IOError('Unknown theta')

  fig, axes = gplt.subplots(ncols=3)

  for ax in axes:
    ax.set_xlabel(r'$(r - r_\mathrm{tip}) / h$')

  axes[0].set_ylabel(r'$\sigma_{xx}$')
  axes[1].set_ylabel(r'$\sigma_{yy}$')
  axes[2].set_ylabel(r'$\sigma_{xy}$')

  for ax in axes:
    ax.set_xlim([0, 500])

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

  gplt.savefig('lefm_theory/const-theta/{0:s}.pdf'.format(theta_name))
  plt.close()
