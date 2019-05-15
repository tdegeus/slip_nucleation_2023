
import matplotlib.pyplot as plt
import numpy             as np
import re, os, h5py

from collections import defaultdict

# --------------------------------------------------------------------------------------------------

def list_nx():

  return [
    'nx=3^6',
    'nx=3^6x2',
  ]

# --------------------------------------------------------------------------------------------------

def num_nx(nx):

  if nx == 'nx=3^6'  : return 3**6
  if nx == 'nx=3^6x2': return 2*(3**6)

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def color_nx(nx):

  if nx == 'nx=3^6'  : return {'color': plt.get_cmap('Greys'  , 10)(7)}
  if nx == 'nx=3^6x2': return {'color': plt.get_cmap('Oranges', 10)(7)}

  raise IOError('Unknown input: ' + nx)

# --------------------------------------------------------------------------------------------------

def linestyle_nx(nx):

  if nx == 'nx=3^6'  : return {'linestyle': '--'}
  if nx == 'nx=3^6x2': return {'linestyle': '-'}

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def marker_nx(nx):

  if nx == 'nx=3^6'  : return {'marker': 's', 'markeredgecolor': 'none'}
  if nx == 'nx=3^6x2': return {'marker': 'h', 'markeredgecolor': 'none'}

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def label_nx(nx):

  if nx == 'nx=3^6'  : return {'label': r'$N = 3^6$'}
  if nx == 'nx=3^6x2': return {'label': r'$N = 3^6 \times 2$'}

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def list_esemble():

  return [
    'nx=3^6',
    'nx=3^6x2',
  ]

# --------------------------------------------------------------------------------------------------

def list_stress():

  return [
    'stress=0d6',
    'stress=1d6',
    'stress=2d6',
    'stress=3d6',
    'stress=4d6',
    'stress=5d6',
    'stress=6d6',
  ]

# --------------------------------------------------------------------------------------------------

def num_stress(stress):

  if stress == 'stress=0d6': return 0./6.
  if stress == 'stress=1d6': return 1./6.
  if stress == 'stress=2d6': return 2./6.
  if stress == 'stress=3d6': return 3./6.
  if stress == 'stress=4d6': return 4./6.
  if stress == 'stress=5d6': return 5./6.
  if stress == 'stress=6d6': return 6./6.

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def color_stress(nx, stress):

  if nx == 'nx=3^6':
    if stress == 'stress=0d6': return {'color':plt.get_cmap('Greys'  , 7+5)( 6+5)}
    if stress == 'stress=1d6': return {'color':plt.get_cmap('Greys'  , 7+5)( 5+5)}
    if stress == 'stress=2d6': return {'color':plt.get_cmap('Greys'  , 7+5)( 4+5)}
    if stress == 'stress=3d6': return {'color':plt.get_cmap('Greys'  , 7+5)( 3+5)}
    if stress == 'stress=4d6': return {'color':plt.get_cmap('Greys'  , 7+5)( 2+5)}
    if stress == 'stress=5d6': return {'color':plt.get_cmap('Greys'  , 7+5)( 1+5)}
    if stress == 'stress=6d6': return {'color':plt.get_cmap('Greys'  , 7+5)( 0+5)}

  if nx == 'nx=3^6x2':
    if stress == 'stress=0d6': return {'color':plt.get_cmap('Oranges', 7+5)( 6+5)}
    if stress == 'stress=1d6': return {'color':plt.get_cmap('Oranges', 7+5)( 5+5)}
    if stress == 'stress=2d6': return {'color':plt.get_cmap('Oranges', 7+5)( 4+5)}
    if stress == 'stress=3d6': return {'color':plt.get_cmap('Oranges', 7+5)( 3+5)}
    if stress == 'stress=4d6': return {'color':plt.get_cmap('Oranges', 7+5)( 2+5)}
    if stress == 'stress=5d6': return {'color':plt.get_cmap('Oranges', 7+5)( 1+5)}
    if stress == 'stress=6d6': return {'color':plt.get_cmap('Oranges', 7+5)( 0+5)}

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def label_stress(stress):

  if stress == 'stress=0d6': return {'label': r'$\Delta_\sigma = 0$'  }
  if stress == 'stress=1d6': return {'label': r'$\Delta_\sigma = 1/6$'}
  if stress == 'stress=2d6': return {'label': r'$\Delta_\sigma = 2/6$'}
  if stress == 'stress=3d6': return {'label': r'$\Delta_\sigma = 3/6$'}
  if stress == 'stress=4d6': return {'label': r'$\Delta_\sigma = 4/6$'}
  if stress == 'stress=5d6': return {'label': r'$\Delta_\sigma = 5/6$'}
  if stress == 'stress=6d6': return {'label': r'$\Delta_\sigma = 1$'  }

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def path(key='data', nx='nx=3^6x2', stress=None, fname='EnsembleInfo.hdf5'):

  if key == 'data':
    return os.path.join('../../data', '_'.join([nx]), fname)

  if key == 'CrackEvolution_strain':
    return os.path.join('../../CrackEvolution_strain/data', '_'.join([nx]), fname)

  if key == 'CrackEvolution_stress':
    return os.path.join('../../CrackEvolution_stress/data', '_'.join([nx, stress]), fname)

