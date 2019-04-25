
import matplotlib.pyplot as plt
import numpy             as np
import re, os

from collections import defaultdict

# --------------------------------------------------------------------------------------------------

dbase = '../../data'

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

def dirname(nx='nx=3^6x2'):

  path = '_'.join([nx])

  return os.path.join(dbase, path)

# --------------------------------------------------------------------------------------------------

def path(nx='nx=3^6x2', fname='EnsembleInfo.hdf5'):

  path = '_'.join([nx])

  return os.path.join(dbase, path, fname)

# --------------------------------------------------------------------------------------------------
