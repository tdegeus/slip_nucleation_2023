
import matplotlib.pyplot as plt
import numpy             as np
import re

from collections import defaultdict

# --------------------------------------------------------------------------------------------------

dbase = '../../data'

G    = 1.0
eps0 = 5.e-4
sig0 = 2. * G * eps0

nu = 'nu=0,c=1'

# --------------------------------------------------------------------------------------------------

def n_nx():
  return 6

# --------------------------------------------------------------------------------------------------

def n_nu():
  return 1

# --------------------------------------------------------------------------------------------------

def n_stress():
  return 9

# --------------------------------------------------------------------------------------------------

def n_strain():
  return 11

# --------------------------------------------------------------------------------------------------

def n_bins():
  return 9

# --------------------------------------------------------------------------------------------------

def list_bins():

  return [
    'bins=501',
    'bins=1001',
    'bins=2001',
    'bins=5001',
    'bins=10001',
    'bins=15001',
    'bins=20001',
    'bins=25001',
    'bins=30001',
  ]

# --------------------------------------------------------------------------------------------------

def num(name):

  return int(name.split('=')[1])

# --------------------------------------------------------------------------------------------------

def list_nx():

  return [
    'nx=3^2',
    'nx=3^3',
    'nx=3^4',
    'nx=3^5',
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

def list_strain():

  return [
    'strain=00d10',
    'strain=01d10',
    'strain=02d10',
    'strain=03d10',
    'strain=04d10',
    'strain=05d10',
    'strain=06d10',
    'strain=07d10',
    'strain=08d10',
    'strain=09d10',
    'strain=10d10',
  ]

# --------------------------------------------------------------------------------------------------

def list_stress_all():

  return [
    'stress=-3d6',
    'stress=-2d6',
    'stress=-1d6',
    'stress=0d6',
    'stress=1d6',
    'stress=2d6',
    'stress=3d6',
    'stress=4d6',
    'stress=5d6',
    'stress=6d6',
  ]

# --------------------------------------------------------------------------------------------------

def list_strain_all():
  return list_strain()

# --------------------------------------------------------------------------------------------------

def list_delta():

  return list_stress() + list_strain()

# --------------------------------------------------------------------------------------------------

def list_delta_all():

  return list_stress_all() + list_strain_all()

# --------------------------------------------------------------------------------------------------

def color(nx, nu, delta):

  if nx == 'nx=3^2' and nu == 'nu=0,c=1':
    if delta == 'stress=-3d6' : return {'color':plt.get_cmap('Purples', 7+5)( 0+5)}
    if delta == 'stress=-2d6' : return {'color':plt.get_cmap('Purples', 7+5)( 2+5)}
    if delta == 'stress=-1d6' : return {'color':plt.get_cmap('Purples', 7+5)( 4+5)}
    if delta == 'stress=0d6'  : return {'color':plt.get_cmap('Purples', 7+5)( 6+5)}
    if delta == 'stress=1d6'  : return {'color':plt.get_cmap('Purples', 7+5)( 5+5)}
    if delta == 'stress=2d6'  : return {'color':plt.get_cmap('Purples', 7+5)( 4+5)}
    if delta == 'stress=3d6'  : return {'color':plt.get_cmap('Purples', 7+5)( 3+5)}
    if delta == 'stress=4d6'  : return {'color':plt.get_cmap('Purples', 7+5)( 2+5)}
    if delta == 'stress=5d6'  : return {'color':plt.get_cmap('Purples', 7+5)( 1+5)}
    if delta == 'stress=6d6'  : return {'color':plt.get_cmap('Purples', 7+5)( 0+5)}
  if nx == 'nx=3^3' and nu == 'nu=0,c=1':
    if delta == 'stress=-3d6' : return {'color':plt.get_cmap('Blues'  , 7+5)( 0+5)}
    if delta == 'stress=-2d6' : return {'color':plt.get_cmap('Blues'  , 7+5)( 2+5)}
    if delta == 'stress=-1d6' : return {'color':plt.get_cmap('Blues'  , 7+5)( 4+5)}
    if delta == 'stress=0d6'  : return {'color':plt.get_cmap('Blues'  , 7+5)( 6+5)}
    if delta == 'stress=1d6'  : return {'color':plt.get_cmap('Blues'  , 7+5)( 5+5)}
    if delta == 'stress=2d6'  : return {'color':plt.get_cmap('Blues'  , 7+5)( 4+5)}
    if delta == 'stress=3d6'  : return {'color':plt.get_cmap('Blues'  , 7+5)( 3+5)}
    if delta == 'stress=4d6'  : return {'color':plt.get_cmap('Blues'  , 7+5)( 2+5)}
    if delta == 'stress=5d6'  : return {'color':plt.get_cmap('Blues'  , 7+5)( 1+5)}
    if delta == 'stress=6d6'  : return {'color':plt.get_cmap('Blues'  , 7+5)( 0+5)}
  if nx == 'nx=3^4' and nu == 'nu=0,c=1':
    if delta == 'stress=-3d6' : return {'color':plt.get_cmap('Greens' , 7+5)( 0+5)}
    if delta == 'stress=-2d6' : return {'color':plt.get_cmap('Greens' , 7+5)( 2+5)}
    if delta == 'stress=-1d6' : return {'color':plt.get_cmap('Greens' , 7+5)( 4+5)}
    if delta == 'stress=0d6'  : return {'color':plt.get_cmap('Greens' , 7+5)( 6+5)}
    if delta == 'stress=1d6'  : return {'color':plt.get_cmap('Greens' , 7+5)( 5+5)}
    if delta == 'stress=2d6'  : return {'color':plt.get_cmap('Greens' , 7+5)( 4+5)}
    if delta == 'stress=3d6'  : return {'color':plt.get_cmap('Greens' , 7+5)( 3+5)}
    if delta == 'stress=4d6'  : return {'color':plt.get_cmap('Greens' , 7+5)( 2+5)}
    if delta == 'stress=5d6'  : return {'color':plt.get_cmap('Greens' , 7+5)( 1+5)}
    if delta == 'stress=6d6'  : return {'color':plt.get_cmap('Greens' , 7+5)( 0+5)}
  if nx == 'nx=3^5' and nu == 'nu=0,c=1':
    if delta == 'stress=-3d6' : return {'color':plt.get_cmap('Reds'   , 7+5)( 0+5)}
    if delta == 'stress=-2d6' : return {'color':plt.get_cmap('Reds'   , 7+5)( 2+5)}
    if delta == 'stress=-1d6' : return {'color':plt.get_cmap('Reds'   , 7+5)( 4+5)}
    if delta == 'stress=0d6'  : return {'color':plt.get_cmap('Reds'   , 7+5)( 6+5)}
    if delta == 'stress=1d6'  : return {'color':plt.get_cmap('Reds'   , 7+5)( 5+5)}
    if delta == 'stress=2d6'  : return {'color':plt.get_cmap('Reds'   , 7+5)( 4+5)}
    if delta == 'stress=3d6'  : return {'color':plt.get_cmap('Reds'   , 7+5)( 3+5)}
    if delta == 'stress=4d6'  : return {'color':plt.get_cmap('Reds'   , 7+5)( 2+5)}
    if delta == 'stress=5d6'  : return {'color':plt.get_cmap('Reds'   , 7+5)( 1+5)}
    if delta == 'stress=6d6'  : return {'color':plt.get_cmap('Reds'   , 7+5)( 0+5)}
  if nx == 'nx=3^6' and nu == 'nu=0,c=1':
    if delta == 'stress=-3d6' : return {'color':plt.get_cmap('Greys'  , 7+5)( 0+5)}
    if delta == 'stress=-2d6' : return {'color':plt.get_cmap('Greys'  , 7+5)( 2+5)}
    if delta == 'stress=-1d6' : return {'color':plt.get_cmap('Greys'  , 7+5)( 4+5)}
    if delta == 'stress=0d6'  : return {'color':plt.get_cmap('Greys'  , 7+5)( 6+5)}
    if delta == 'stress=1d6'  : return {'color':plt.get_cmap('Greys'  , 7+5)( 5+5)}
    if delta == 'stress=2d6'  : return {'color':plt.get_cmap('Greys'  , 7+5)( 4+5)}
    if delta == 'stress=3d6'  : return {'color':plt.get_cmap('Greys'  , 7+5)( 3+5)}
    if delta == 'stress=4d6'  : return {'color':plt.get_cmap('Greys'  , 7+5)( 2+5)}
    if delta == 'stress=5d6'  : return {'color':plt.get_cmap('Greys'  , 7+5)( 1+5)}
    if delta == 'stress=6d6'  : return {'color':plt.get_cmap('Greys'  , 7+5)( 0+5)}
  if nx == 'nx=3^6x2' and nu == 'nu=0,c=1':
    if delta == 'stress=-3d6' : return {'color':plt.get_cmap('Oranges', 7+5)( 0+5)}
    if delta == 'stress=-2d6' : return {'color':plt.get_cmap('Oranges', 7+5)( 2+5)}
    if delta == 'stress=-1d6' : return {'color':plt.get_cmap('Oranges', 7+5)( 4+5)}
    if delta == 'stress=0d6'  : return {'color':plt.get_cmap('Oranges', 7+5)( 6+5)}
    if delta == 'stress=1d6'  : return {'color':plt.get_cmap('Oranges', 7+5)( 5+5)}
    if delta == 'stress=2d6'  : return {'color':plt.get_cmap('Oranges', 7+5)( 4+5)}
    if delta == 'stress=3d6'  : return {'color':plt.get_cmap('Oranges', 7+5)( 3+5)}
    if delta == 'stress=4d6'  : return {'color':plt.get_cmap('Oranges', 7+5)( 2+5)}
    if delta == 'stress=5d6'  : return {'color':plt.get_cmap('Oranges', 7+5)( 1+5)}
    if delta == 'stress=6d6'  : return {'color':plt.get_cmap('Oranges', 7+5)( 0+5)}

  if nx == 'nx=3^2' and nu == 'nu=0,c=1':
    if delta == 'strain=00d10': return {'color':plt.get_cmap('Purples', 11+5)(10+5)}
    if delta == 'strain=01d10': return {'color':plt.get_cmap('Purples', 11+5)( 9+5)}
    if delta == 'strain=02d10': return {'color':plt.get_cmap('Purples', 11+5)( 8+5)}
    if delta == 'strain=03d10': return {'color':plt.get_cmap('Purples', 11+5)( 7+5)}
    if delta == 'strain=04d10': return {'color':plt.get_cmap('Purples', 11+5)( 6+5)}
    if delta == 'strain=05d10': return {'color':plt.get_cmap('Purples', 11+5)( 5+5)}
    if delta == 'strain=06d10': return {'color':plt.get_cmap('Purples', 11+5)( 4+5)}
    if delta == 'strain=07d10': return {'color':plt.get_cmap('Purples', 11+5)( 3+5)}
    if delta == 'strain=08d10': return {'color':plt.get_cmap('Purples', 11+5)( 2+5)}
    if delta == 'strain=09d10': return {'color':plt.get_cmap('Purples', 11+5)( 1+5)}
    if delta == 'strain=10d10': return {'color':plt.get_cmap('Purples', 11+5)( 0+5)}
  if nx == 'nx=3^3' and nu == 'nu=0,c=1':
    if delta == 'strain=00d10': return {'color':plt.get_cmap('Blues'  , 11+5)(10+5)}
    if delta == 'strain=01d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 9+5)}
    if delta == 'strain=02d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 8+5)}
    if delta == 'strain=03d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 7+5)}
    if delta == 'strain=04d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 6+5)}
    if delta == 'strain=05d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 5+5)}
    if delta == 'strain=06d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 4+5)}
    if delta == 'strain=07d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 3+5)}
    if delta == 'strain=08d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 2+5)}
    if delta == 'strain=09d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 1+5)}
    if delta == 'strain=10d10': return {'color':plt.get_cmap('Blues'  , 11+5)( 0+5)}
  if nx == 'nx=3^4' and nu == 'nu=0,c=1':
    if delta == 'strain=00d10': return {'color':plt.get_cmap('Greens' , 11+5)(10+5)}
    if delta == 'strain=01d10': return {'color':plt.get_cmap('Greens' , 11+5)( 9+5)}
    if delta == 'strain=02d10': return {'color':plt.get_cmap('Greens' , 11+5)( 8+5)}
    if delta == 'strain=03d10': return {'color':plt.get_cmap('Greens' , 11+5)( 7+5)}
    if delta == 'strain=04d10': return {'color':plt.get_cmap('Greens' , 11+5)( 6+5)}
    if delta == 'strain=05d10': return {'color':plt.get_cmap('Greens' , 11+5)( 5+5)}
    if delta == 'strain=06d10': return {'color':plt.get_cmap('Greens' , 11+5)( 4+5)}
    if delta == 'strain=07d10': return {'color':plt.get_cmap('Greens' , 11+5)( 3+5)}
    if delta == 'strain=08d10': return {'color':plt.get_cmap('Greens' , 11+5)( 2+5)}
    if delta == 'strain=09d10': return {'color':plt.get_cmap('Greens' , 11+5)( 1+5)}
    if delta == 'strain=10d10': return {'color':plt.get_cmap('Greens' , 11+5)( 0+5)}
  if nx == 'nx=3^5' and nu == 'nu=0,c=1':
    if delta == 'strain=00d10': return {'color':plt.get_cmap('Reds'   , 11+5)(10+5)}
    if delta == 'strain=01d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 9+5)}
    if delta == 'strain=02d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 8+5)}
    if delta == 'strain=03d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 7+5)}
    if delta == 'strain=04d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 6+5)}
    if delta == 'strain=05d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 5+5)}
    if delta == 'strain=06d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 4+5)}
    if delta == 'strain=07d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 3+5)}
    if delta == 'strain=08d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 2+5)}
    if delta == 'strain=09d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 1+5)}
    if delta == 'strain=10d10': return {'color':plt.get_cmap('Reds'   , 11+5)( 0+5)}
  if nx == 'nx=3^6' and nu == 'nu=0,c=1':
    if delta == 'strain=00d10': return {'color':plt.get_cmap('Greys'  , 11+5)(10+5)}
    if delta == 'strain=01d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 9+5)}
    if delta == 'strain=02d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 8+5)}
    if delta == 'strain=03d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 7+5)}
    if delta == 'strain=04d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 6+5)}
    if delta == 'strain=05d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 5+5)}
    if delta == 'strain=06d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 4+5)}
    if delta == 'strain=07d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 3+5)}
    if delta == 'strain=08d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 2+5)}
    if delta == 'strain=09d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 1+5)}
    if delta == 'strain=10d10': return {'color':plt.get_cmap('Greys'  , 11+5)( 0+5)}
  if nx == 'nx=3^6x2' and nu == 'nu=0,c=1':
    if delta == 'strain=00d10': return {'color':plt.get_cmap('Oranges', 11+5)(10+5)}
    if delta == 'strain=01d10': return {'color':plt.get_cmap('Oranges', 11+5)( 9+5)}
    if delta == 'strain=02d10': return {'color':plt.get_cmap('Oranges', 11+5)( 8+5)}
    if delta == 'strain=03d10': return {'color':plt.get_cmap('Oranges', 11+5)( 7+5)}
    if delta == 'strain=04d10': return {'color':plt.get_cmap('Oranges', 11+5)( 6+5)}
    if delta == 'strain=05d10': return {'color':plt.get_cmap('Oranges', 11+5)( 5+5)}
    if delta == 'strain=06d10': return {'color':plt.get_cmap('Oranges', 11+5)( 4+5)}
    if delta == 'strain=07d10': return {'color':plt.get_cmap('Oranges', 11+5)( 3+5)}
    if delta == 'strain=08d10': return {'color':plt.get_cmap('Oranges', 11+5)( 2+5)}
    if delta == 'strain=09d10': return {'color':plt.get_cmap('Oranges', 11+5)( 1+5)}
    if delta == 'strain=10d10': return {'color':plt.get_cmap('Oranges', 11+5)( 0+5)}

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def color_nx(nx):

  if nx == 'nx=3^2'  : return {'color':plt.get_cmap('Purples', 10)(7)}
  if nx == 'nx=3^3'  : return {'color':plt.get_cmap('Blues'  , 10)(7)}
  if nx == 'nx=3^4'  : return {'color':plt.get_cmap('Greens' , 10)(7)}
  if nx == 'nx=3^5'  : return {'color':plt.get_cmap('Reds'   , 10)(7)}
  if nx == 'nx=3^6'  : return {'color':plt.get_cmap('Greys'  , 10)(7)}
  if nx == 'nx=3^6x2': return {'color':plt.get_cmap('Oranges', 10)(7)}

  raise IOError('Unknown input: ' + nx)

# --------------------------------------------------------------------------------------------------

def linestyle_nx(nx):

  if nx == 'nx=3^2'  : return {'linestyle': '-'}
  if nx == 'nx=3^3'  : return {'linestyle': '-'}
  if nx == 'nx=3^4'  : return {'linestyle': '-'}
  if nx == 'nx=3^5'  : return {'linestyle': '-'}
  if nx == 'nx=3^6'  : return {'linestyle': '--'}
  if nx == 'nx=3^6x2': return {'linestyle': '-'}

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def marker_nx(nx):

  if nx == 'nx=3^2'  : return 'P'
  if nx == 'nx=3^3'  : return 'X'
  if nx == 'nx=3^4'  : return 'v'
  if nx == 'nx=3^5'  : return 'd'
  if nx == 'nx=3^6'  : return 's'
  if nx == 'nx=3^6x2': return 'h'

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def marker_stress(stress):

  if stress == 'stress=-3d6' : return '*'
  if stress == 'stress=-2d6' : return 'P'
  if stress == 'stress=-1d6' : return 'h'
  if stress == 'stress=0d6'  : return 'o'
  if stress == 'stress=1d6'  : return '>'
  if stress == 'stress=2d6'  : return 'v'
  if stress == 'stress=3d6'  : return 's'
  if stress == 'stress=4d6'  : return 'h'
  if stress == 'stress=5d6'  : return 'P'
  if stress == 'stress=6d6'  : return '*'

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def marker_strain(strain):

  if strain == 'strain=00d10' : return 'o'
  if strain == 'strain=01d10' : return '^'
  if strain == 'strain=02d10' : return '>'
  if strain == 'strain=03d10' : return 'v'
  if strain == 'strain=04d10' : return '<'
  if strain == 'strain=05d10' : return 's'
  if strain == 'strain=06d10' : return 'd'
  if strain == 'strain=07d10' : return 'h'
  if strain == 'strain=08d10' : return 'P'
  if strain == 'strain=09d10' : return 'X'
  if strain == 'strain=10d10' : return '*'

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def marker_delta(delta):

  if re.match(r'stress.*', delta): return marker_stress(delta)
  else                           : return marker_strain(delta)

# --------------------------------------------------------------------------------------------------

def num_nx(nx):

  if nx == 'nx=3^2'  : return 3**2
  if nx == 'nx=3^3'  : return 3**3
  if nx == 'nx=3^4'  : return 3**4
  if nx == 'nx=3^5'  : return 3**5
  if nx == 'nx=3^6'  : return 3**6
  if nx == 'nx=3^6x2': return 2*(3**6)

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def num_strain(strain):

  if strain == 'strain=00d10': return  0./10.
  if strain == 'strain=01d10': return  1./10.
  if strain == 'strain=02d10': return  2./10.
  if strain == 'strain=03d10': return  3./10.
  if strain == 'strain=04d10': return  4./10.
  if strain == 'strain=05d10': return  5./10.
  if strain == 'strain=06d10': return  6./10.
  if strain == 'strain=07d10': return  7./10.
  if strain == 'strain=08d10': return  8./10.
  if strain == 'strain=09d10': return  9./10.
  if strain == 'strain=10d10': return 10./10.

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def num_stress(stress):

  if stress == 'stress=-3d6' : return -3./6.
  if stress == 'stress=-2d6' : return -2./6.
  if stress == 'stress=-1d6' : return -1./6.
  if stress == 'stress=0d6' : return 0./6.
  if stress == 'stress=1d6' : return 1./6.
  if stress == 'stress=2d6' : return 2./6.
  if stress == 'stress=3d6' : return 3./6.
  if stress == 'stress=4d6' : return 4./6.
  if stress == 'stress=5d6' : return 5./6.
  if stress == 'stress=6d6' : return 6./6.

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def num_delta(delta):

  if re.match(r'stress.*', delta): return num_stress(delta)
  else                           : return num_strain(delta)

# --------------------------------------------------------------------------------------------------

def label_nx(nx):

  if nx == 'nx=3^2'  : return r'$N = 3^2$'
  if nx == 'nx=3^3'  : return r'$N = 3^3$'
  if nx == 'nx=3^4'  : return r'$N = 3^4$'
  if nx == 'nx=3^5'  : return r'$N = 3^5$'
  if nx == 'nx=3^6'  : return r'$N = 3^6$'
  if nx == 'nx=3^6x2': return r'$N = 3^6 \times 2$'

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def label_strain(strain):

  if strain == 'strain=00d10': return r'$\Delta_\varepsilon = 0$'
  if strain == 'strain=01d10': return r'$\Delta_\varepsilon = 1/10$'
  if strain == 'strain=02d10': return r'$\Delta_\varepsilon = 2/10$'
  if strain == 'strain=03d10': return r'$\Delta_\varepsilon = 3/10$'
  if strain == 'strain=04d10': return r'$\Delta_\varepsilon = 4/10$'
  if strain == 'strain=05d10': return r'$\Delta_\varepsilon = 5/10$'
  if strain == 'strain=06d10': return r'$\Delta_\varepsilon = 6/10$'
  if strain == 'strain=07d10': return r'$\Delta_\varepsilon = 7/10$'
  if strain == 'strain=08d10': return r'$\Delta_\varepsilon = 8/10$'
  if strain == 'strain=09d10': return r'$\Delta_\varepsilon = 9/10$'
  if strain == 'strain=10d10': return r'$\Delta_\varepsilon = 1$'

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def label_stress(stress):

  if stress == 'stress=-3d6' : return r'$\Delta_\sigma = -3/6$'
  if stress == 'stress=-2d6' : return r'$\Delta_\sigma = -2/6$'
  if stress == 'stress=-1d6' : return r'$\Delta_\sigma = -1/6$'
  if stress == 'stress=0d6'  : return r'$\Delta_\sigma =  0$'
  if stress == 'stress=1d6'  : return r'$\Delta_\sigma =  1/6$'
  if stress == 'stress=2d6'  : return r'$\Delta_\sigma =  2/6$'
  if stress == 'stress=3d6'  : return r'$\Delta_\sigma =  3/6$'
  if stress == 'stress=4d6'  : return r'$\Delta_\sigma =  4/6$'
  if stress == 'stress=5d6'  : return r'$\Delta_\sigma =  5/6$'
  if stress == 'stress=6d6'  : return r'$\Delta_\sigma =  1$'

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def label_delta(delta):

  if re.match(r'stress.*', delta): return label_stress(delta)
  else                           : return label_strain(delta)

# --------------------------------------------------------------------------------------------------

def label(nx, nu, delta):

  return ', '.join([label_nx(nx), label_delta(delta)])

# --------------------------------------------------------------------------------------------------

def style_line(nx, nu, delta):

  settings = {}

  settings['color'] = color_nx(nx)['color']

  if re.match(r'stress.*', delta):
    settings['linestyle'] = '-'
  else:
    settings['linestyle'] = '--'

  return settings

# --------------------------------------------------------------------------------------------------

def shearwave(nu):

  if nu == 'nu=10^{+2}': return np.sqrt( 10.**(+2) / 2. )
  if nu == 'nu=10^{+1}': return np.sqrt( 10.**(+1) / 2. )
  if nu == 'nu=10^{+0}': return np.sqrt( 10.**(+0) / 2. )
  if nu == 'nu=10^{-1}': return np.sqrt( 10.**(-1) / 2. )
  if nu == 'nu=10^{-2}': return np.sqrt( 10.**(-2) / 2. )
  if nu == 'nu=10^{-3}': return np.sqrt( 10.**(-3) / 2. )
  if nu == 'nu=10^{-4}': return np.sqrt( 10.**(-4) / 2. )
  if nu == 'nu=10^{-5}': return np.sqrt( 10.**(-5) / 2. )
  if nu == 'nu=10^{-6}': return np.sqrt( 10.**(-6) / 2. )
  if nu == 'nu=0,c=1'  : return 1./np.sqrt(2.)

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------
