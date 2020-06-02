r'''
Collected data at synchronised avalanche area `A`,
for "plastic" blocks along the weak layer.

Usage:
  collect_sync-A_plastic.py [options] <files>...

Arguments:
  <files>   Files from which to collect data.

Options:
  -o, --output=<N>  Output file. [default: output.hdf5]
  -i, --info=<N>    Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
  -f, --force       Overwrite existing output-file.
  -h, --help        Print help.
'''

import os
import sys
import docopt
import click
import h5py
import numpy as np
import GooseFEM as gf

# ==================================================================================================
# compute center of mass
# https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
# ==================================================================================================

def center_of_mass(x, L):
    if np.allclose(x, 0):
        return 0
    theta = 2.0 * np.pi * x / L
    xi = np.cos(theta)
    zeta = np.sin(theta)
    xi_bar = np.mean(xi)
    zeta_bar = np.mean(zeta)
    theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
    return L * theta_bar / (2.0 * np.pi)

def renumber(x, L):
    center = center_of_mass(x, L)
    N = int(L)
    M = int((N - N % 2) / 2)
    C = int(center)
    return np.roll(np.arange(N), M - C)

# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args['<files>']
info = args['--info']
output = args['--output']

for file in files + [info]:
    if not os.path.isfile(file):
        raise IOError('"{0:s}" does not exist'.format(file))

if not args['--force']:
    if os.path.isfile(output):
        print('"{0:s}" exists'.format(output))
        if not click.confirm('Proceed?'):
            sys.exit(1)

# ==================================================================================================
# get constants
# ==================================================================================================

with h5py.File(files[0], 'r') as data:
    plastic = data['/meta/plastic'][...]
    nx = len(plastic)
    h = np.pi

# ==================================================================================================
# get normalisation
# ==================================================================================================

with h5py.File(info, 'r') as data:
    dt = float(data['/normalisation/dt'][...])
    t0 = float(data['/normalisation/t0'][...])
    sig0 = float(data['/normalisation/sig0'][...])
    eps0 = float(data['/normalisation/eps0'][...])

# ==================================================================================================
# get mapping
# ==================================================================================================

mesh = gf.Mesh.Quad4.FineLayer(nx, nx, h)

assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

if nx % 2 == 0:
    mid = nx / 2
else:
    mid = (nx - 1) / 2

mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)
regular = mapping.getRegularMesh()
elmat = regular.elementMatrix()

# ==================================================================================================
# ensemble average
# ==================================================================================================

# -------------------------------------------------------------------------
# initialise normalisation, and sum of first and second statistical moments
# -------------------------------------------------------------------------

norm = np.zeros((nx + 1), dtype='uint')
norm_x = np.zeros((nx + 1), dtype='uint')

out = {
    '1st': {},
    '2nd': {},
}

for key in out:

    out[key]['sig_xx'] = np.zeros((nx + 1, nx), dtype='float') # (A, x)
    out[key]['sig_xy'] = np.zeros((nx + 1, nx), dtype='float')
    out[key]['sig_yy'] = np.zeros((nx + 1, nx), dtype='float')
    out[key]['epsp'  ] = np.zeros((nx + 1, nx), dtype='float')
    out[key]['depsp' ] = np.zeros((nx + 1, nx), dtype='float')
    out[key]['x'     ] = np.zeros((nx + 1, nx), dtype='float')
    out[key]['S'     ] = np.zeros((nx + 1, nx), dtype='int')

# ---------------
# loop over files
# ---------------

for ifile, file in enumerate(files):

    print('({0:3d}/{1:3d}) {2:s}'.format(ifile + 1, len(files), file))

    with h5py.File(file, 'r') as data:

        A = data["/sync-A/stored"][...]
        idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(A))][...]

        if '/sync-A/plastic/{0:d}/epsp'.format(np.min(A)) in data:
            epsp0 = data['/sync-A/plastic/{0:d}/epsp'.format(np.min(A))][...]
            norm_x[A] += 1
            store_x = True
        else:
            store_x = False

        norm[A] += 1

        for a in A:

            if "/sync-A/element/{0:d}/sig_xx".format(a) in data:
                sig_xx = data["/sync-A/element/{0:d}/sig_xx".format(a)][...][plastic]
                sig_xy = data["/sync-A/element/{0:d}/sig_xy".format(a)][...][plastic]
                sig_yy = data["/sync-A/element/{0:d}/sig_yy".format(a)][...][plastic]
            else:
                sig_xx = data["/sync-A/plastic/{0:d}/sig_xx".format(a)][...]
                sig_xy = data["/sync-A/plastic/{0:d}/sig_xy".format(a)][...]
                sig_yy = data["/sync-A/plastic/{0:d}/sig_yy".format(a)][...]

            idx = data['/sync-A/plastic/{0:d}/idx'.format(a)][...]

            if store_x:
                epsp = data['/sync-A/plastic/{0:d}/epsp'.format(a)][...]
                x = data['/sync-A/plastic/{0:d}/x'.format(a)][...]

            renum = renumber(np.argwhere(idx0 != idx).ravel(), nx)

            out['1st']['sig_xx'][a, :] += sig_xx[renum]
            out['1st']['sig_xy'][a, :] += sig_xy[renum]
            out['1st']['sig_yy'][a, :] += sig_yy[renum]
            out['1st']['S'     ][a, :] += (idx - idx0)[renum].astype(np.int)
            if store_x:
                out['1st']['epsp' ][a, :] += epsp[renum]
                out['1st']['depsp'][a, :] += (epsp - epsp0)[renum]
                out['1st']['x'    ][a, :] += x[renum]

            out['2nd']['sig_xx'][a, :] += (sig_xx[renum]) ** 2.0
            out['2nd']['sig_xy'][a, :] += (sig_xy[renum]) ** 2.0
            out['2nd']['sig_yy'][a, :] += (sig_yy[renum]) ** 2.0
            out['2nd']['S'     ][a, :] += ((idx - idx0)[renum].astype(np.int)) ** 2
            if store_x:
                out['2nd']['epsp' ][a, :] += (epsp[renum]) ** 2.0
                out['2nd']['depsp'][a, :] += ((epsp - epsp0)[renum]) ** 2.0
                out['2nd']['x'    ][a, :] += (x[renum]) ** 2.0

# ---------------------------------------------
# select only measurements with sufficient data
# ---------------------------------------------

idx = np.argwhere(norm > 30).ravel()

A = np.arange(nx + 1)
norm = norm[idx].astype(np.float)
norm_x = norm_x[idx].astype(np.float)
A = A[idx].astype(np.float)

for key in out:
    for field in out[key]:
        out[key][field] = out[key][field][idx, :]

# ----------------------
# store support function
# ----------------------


def store(data, key,
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_depsp, m_x, m_S,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_depsp, v_x, v_S):

    # hydrostatic stress
    m_sig_m = (m_sig_xx + m_sig_yy) / 2.

    # variance
    v_sig_m = v_sig_xx * (m_sig_xx / 2.0)**2.0 + v_sig_yy * (m_sig_yy / 2.0)**2.0

    # deviatoric stress
    m_sigd_xx = m_sig_xx - m_sig_m
    m_sigd_xy = m_sig_xy
    m_sigd_yy = m_sig_yy - m_sig_m

    # equivalent stress
    m_sig_eq = np.sqrt(2.0 * (m_sigd_xx**2.0 + m_sigd_yy**2.0 + 2.0 * m_sigd_xy**2.0))

    # correct for division
    sig_eq = np.where(m_sig_eq != 0.0, m_sig_eq, 1.0)

    # variance
    v_sig_eq = v_sig_xx * ((m_sig_xx - 0.5 * (m_sig_xx + m_sig_yy)) / sig_eq)**2.0 +\
               v_sig_yy * ((m_sig_yy - 0.5 * (m_sig_xx + m_sig_yy)) / sig_eq)**2.0 +\
               v_sig_xy * (4.0 * m_sig_xy / sig_eq)**2.0

    # store mean
    data['/{0:s}/avr/sig_eq'.format(key)] = m_sig_eq / sig0
    data['/{0:s}/avr/sig_m'.format(key)] = m_sig_m / sig0
    data['/{0:s}/avr/epsp'.format(key)] = m_epsp / eps0
    data['/{0:s}/avr/depsp'.format(key)] = m_depsp / eps0
    data['/{0:s}/avr/x'.format(key)] = m_x / eps0
    data['/{0:s}/avr/S'.format(key)] = m_S

    # store variance
    data['/{0:s}/std/sig_eq'.format(key)] = np.sqrt(np.abs(v_sig_eq)) / sig0
    data['/{0:s}/std/sig_m'.format(key)] = np.sqrt(np.abs(v_sig_m)) / sig0
    data['/{0:s}/std/epsp'.format(key)] = np.sqrt(np.abs(v_epsp)) / eps0
    data['/{0:s}/std/depsp'.format(key)] = np.sqrt(np.abs(v_depsp)) / eps0
    data['/{0:s}/std/x'.format(key)] = np.sqrt(np.abs(v_x)) / eps0
    data['/{0:s}/std/S'.format(key)] = np.sqrt(np.abs(v_S))

# -----
# store
# -----

def compute_average(first, norm):
    r'''
first: sum(a)
norm: number of items in sum
    '''
    return first / norm

def compute_variance(first, second, norm):
    r'''
first: sum(a)
second: sum(a ** 2)
norm: number of items in sum
    '''
    return (second / norm - (first / norm) ** 2.0) * norm / (norm - 1.0)


# open output file
with h5py.File(output, 'w') as data:

    data['/A'] = A

    # ---------

    # allow broadcasting
    norm = norm.reshape((-1, 1))
    norm_x = norm_x.reshape((-1, 1))
    A = A.reshape((-1, 1))

    # compute mean
    m_sig_xx = compute_average(out['1st']['sig_xx'], norm)
    m_sig_xy = compute_average(out['1st']['sig_xy'], norm)
    m_sig_yy = compute_average(out['1st']['sig_yy'], norm)
    m_S      = compute_average(out['1st']['S'     ], norm)
    m_epsp   = compute_average(out['1st']['epsp'  ], norm_x)
    m_depsp  = compute_average(out['1st']['depsp' ], norm_x)
    m_x      = compute_average(out['1st']['x'     ], norm_x)

    # compute variance
    v_sig_xx = compute_variance(out['1st']['sig_xx'], out['2nd']['sig_xx'], norm)
    v_sig_xy = compute_variance(out['1st']['sig_xy'], out['2nd']['sig_xy'], norm)
    v_sig_yy = compute_variance(out['1st']['sig_yy'], out['2nd']['sig_yy'], norm)
    v_S      = compute_variance(out['1st']['S'     ], out['2nd']['S'     ], norm)
    v_epsp   = compute_variance(out['1st']['epsp'  ], out['2nd']['epsp'  ], norm_x)
    v_depsp  = compute_variance(out['1st']['depsp' ], out['2nd']['depsp' ], norm_x)
    v_x      = compute_variance(out['1st']['x'     ], out['2nd']['x'     ], norm_x)

    # store
    store(data, 'element',
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_depsp, m_x, m_S,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_depsp, v_x, v_S)

    # ---------

    # disable broadcasting
    norm = norm.ravel()
    norm_x = norm_x.ravel()
    A = A.ravel()

    # compute mean
    m_sig_xx = compute_average(np.sum(out['1st']['sig_xx'], axis=1), norm * nx)
    m_sig_xy = compute_average(np.sum(out['1st']['sig_xy'], axis=1), norm * nx)
    m_sig_yy = compute_average(np.sum(out['1st']['sig_yy'], axis=1), norm * nx)
    m_S      = compute_average(np.sum(out['1st']['S'     ], axis=1), norm * nx)
    m_epsp   = compute_average(np.sum(out['1st']['epsp'  ], axis=1), norm_x * nx)
    m_depsp  = compute_average(np.sum(out['1st']['depsp' ], axis=1), norm_x * nx)
    m_x      = compute_average(np.sum(out['1st']['x'     ], axis=1), norm_x * nx)

    # compute variance
    v_sig_xx = compute_variance(np.sum(out['1st']['sig_xx'], axis=1), np.sum(out['2nd']['sig_xx'], axis=1), norm * nx)
    v_sig_xy = compute_variance(np.sum(out['1st']['sig_xy'], axis=1), np.sum(out['2nd']['sig_xy'], axis=1), norm * nx)
    v_sig_yy = compute_variance(np.sum(out['1st']['sig_yy'], axis=1), np.sum(out['2nd']['sig_yy'], axis=1), norm * nx)
    v_S      = compute_variance(np.sum(out['1st']['S'     ], axis=1), np.sum(out['2nd']['S'     ], axis=1), norm * nx)
    v_epsp   = compute_variance(np.sum(out['1st']['epsp'  ], axis=1), np.sum(out['2nd']['epsp'  ], axis=1), norm_x * nx)
    v_depsp  = compute_variance(np.sum(out['1st']['depsp' ], axis=1), np.sum(out['2nd']['depsp' ], axis=1), norm_x * nx)
    v_x      = compute_variance(np.sum(out['1st']['x'     ], axis=1), np.sum(out['2nd']['x'     ], axis=1), norm_x * nx)

    # store
    store(data, 'layer',
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_depsp, m_x, m_S,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_depsp, v_x, v_S)

    # ---------

    # remove data outside crack
    for i, a in enumerate(A):
        a = int(a)
        mid = int((nx - nx % 2) / 2)
        il = int(mid - a / 2)
        iu = int(mid + a / 2)
        for key in out:
            for field in out[key]:
                out[key][field] = out[key][field].astype(np.float)
                out[key][field][i, :il] = 0.
                out[key][field][i, iu:] = 0.

    A[A == 0.0] = 1.0

    # compute mean
    m_sig_xx = compute_average(np.sum(out['1st']['sig_xx'], axis=1), norm * A)
    m_sig_xy = compute_average(np.sum(out['1st']['sig_xy'], axis=1), norm * A)
    m_sig_yy = compute_average(np.sum(out['1st']['sig_yy'], axis=1), norm * A)
    m_S      = compute_average(np.sum(out['1st']['S'     ], axis=1), norm * A)
    m_epsp   = compute_average(np.sum(out['1st']['epsp'  ], axis=1), norm_x * A)
    m_depsp  = compute_average(np.sum(out['1st']['depsp' ], axis=1), norm_x * A)
    m_x      = compute_average(np.sum(out['1st']['x'     ], axis=1), norm_x * A)

    # compute variance
    v_sig_xx = compute_variance(np.sum(out['1st']['sig_xx'], axis=1), np.sum(out['2nd']['sig_xx'], axis=1) , norm * A)
    v_sig_xy = compute_variance(np.sum(out['1st']['sig_xy'], axis=1), np.sum(out['2nd']['sig_xy'], axis=1) , norm * A)
    v_sig_yy = compute_variance(np.sum(out['1st']['sig_yy'], axis=1), np.sum(out['2nd']['sig_yy'], axis=1) , norm * A)
    v_S      = compute_variance(np.sum(out['1st']['S'     ], axis=1), np.sum(out['2nd']['S'     ], axis=1) , norm * A)
    v_epsp   = compute_variance(np.sum(out['1st']['epsp'  ], axis=1), np.sum(out['2nd']['epsp'  ], axis=1) , norm_x * A)
    v_depsp  = compute_variance(np.sum(out['1st']['depsp' ], axis=1), np.sum(out['2nd']['depsp' ], axis=1) , norm_x * A)
    v_x      = compute_variance(np.sum(out['1st']['x'     ], axis=1), np.sum(out['2nd']['x'     ], axis=1) , norm_x * A)

    # store
    store(data, 'crack',
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_depsp, m_x, m_S,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_depsp, v_x, v_S)
