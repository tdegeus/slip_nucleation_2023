
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

def alternative_color_stress(nx, stress):

  if nx == 'nx=3^6':
    if stress == 'stress=0d6': return {'color':plt.get_cmap('Reds'   , 7+5)( 6+5)}
    if stress == 'stress=1d6': return {'color':plt.get_cmap('Reds'   , 7+5)( 5+5)}
    if stress == 'stress=2d6': return {'color':plt.get_cmap('Reds'   , 7+5)( 4+5)}
    if stress == 'stress=3d6': return {'color':plt.get_cmap('Reds'   , 7+5)( 3+5)}
    if stress == 'stress=4d6': return {'color':plt.get_cmap('Reds'   , 7+5)( 2+5)}
    if stress == 'stress=5d6': return {'color':plt.get_cmap('Reds'   , 7+5)( 1+5)}
    if stress == 'stress=6d6': return {'color':plt.get_cmap('Reds'   , 7+5)( 0+5)}

  if nx == 'nx=3^6x2':
    if stress == 'stress=0d6': return {'color':plt.get_cmap('Greens' , 7+5)( 6+5)}
    if stress == 'stress=1d6': return {'color':plt.get_cmap('Greens' , 7+5)( 5+5)}
    if stress == 'stress=2d6': return {'color':plt.get_cmap('Greens' , 7+5)( 4+5)}
    if stress == 'stress=3d6': return {'color':plt.get_cmap('Greens' , 7+5)( 3+5)}
    if stress == 'stress=4d6': return {'color':plt.get_cmap('Greens' , 7+5)( 2+5)}
    if stress == 'stress=5d6': return {'color':plt.get_cmap('Greens' , 7+5)( 1+5)}
    if stress == 'stress=6d6': return {'color':plt.get_cmap('Greens' , 7+5)( 0+5)}

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

def label_stress_minimal(stress):

  if stress == 'stress=0d6': return {'label': r'$\Delta_\sigma = 0$'}
  if stress == 'stress=1d6': return {}
  if stress == 'stress=2d6': return {}
  if stress == 'stress=3d6': return {}
  if stress == 'stress=4d6': return {}
  if stress == 'stress=5d6': return {}
  if stress == 'stress=6d6': return {'label': r'$\Delta_\sigma = 1$'}

  raise IOError('Undefined behaviour')

# --------------------------------------------------------------------------------------------------

def path(key='data', nx='nx=3^6x2', stress=None, fname='EnsembleInfo.hdf5'):

  if key == 'data':
    return os.path.join('../../data', '_'.join([nx]), fname)

  if key == 'CrackEvolution_strain':
    return os.path.join('../../CrackEvolution_strain/data', '_'.join([nx]), fname)

  if key == 'CrackEvolution_stress':
    return os.path.join('../../CrackEvolution_stress/data', '_'.join([nx, stress]), fname)

# --------------------------------------------------------------------------------------------------

def read_Ac(nx, stress):

  if nx == 'nx=3^6':

    if stress == 'stress=0d6'  : return 579.94053891264
    if stress == 'stress=1d6'  : return 468.4266846307872
    if stress == 'stress=2d6'  : return 278.3416529148491
    if stress == 'stress=3d6'  : return 181.6644088180627
    if stress == 'stress=4d6'  : return 142.3413022708431
    if stress == 'stress=5d6'  : return  55.08619716070644
    if stress == 'stress=6d6'  : return  36.70793280955525
    if stress == 'strain=00d10': return 573.9742678102573
    if stress == 'strain=01d10': return 549.962607594159
    if stress == 'strain=02d10': return 543.5918854419907
    if stress == 'strain=03d10': return 500.2618633287878
    if stress == 'strain=04d10': return 411.81796288763445
    if stress == 'strain=05d10': return 261.731123742007
    if stress == 'strain=06d10': return 169.15822901405554
    if stress == 'strain=07d10': return 161.44899085347944
    if stress == 'strain=08d10': return 182.04321219909482
    if stress == 'strain=09d10': return  59.56836237092701
    if stress == 'strain=10d10': return  89.6869816057923

  if nx == 'nx=3^6x2':

    if stress == 'stress=0d6'  : return 932.3844723939063
    if stress == 'stress=1d6'  : return 467.3199738145631
    if stress == 'stress=2d6'  : return 305.8574308533365
    if stress == 'stress=3d6'  : return 147.71061575923957
    if stress == 'stress=4d6'  : return  86.65425420195837
    if stress == 'stress=5d6'  : return  57.32524892440282
    if stress == 'stress=6d6'  : return  29.4690945768259
    if stress == 'strain=00d10': return 1102.3234436207288
    if stress == 'strain=01d10': return 985.7220701495067
    if stress == 'strain=02d10': return 786.516388540789
    if stress == 'strain=03d10': return 459.87728558915177
    if stress == 'strain=04d10': return 283.7869985066645
    if stress == 'strain=05d10': return 140.85333442625722
    if stress == 'strain=06d10': return 152.8787377601085
    if stress == 'strain=07d10': return 157.93041477971647
    if stress == 'strain=08d10': return 233.43577735028143
    if stress == 'strain=09d10': return  81.8469043944492
    if stress == 'strain=10d10': return  43.2792140905199

  raise IOError('Undefined behaviour')
