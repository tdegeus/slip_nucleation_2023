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
import tqdm

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
source_dir = os.path.dirname(info)
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
elmat = regular.elementgrid()

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

moving_average = {
    '1st': {},
    '2nd': {},
}

for key in out:

    out[key]['sig_xx' ] = np.zeros((nx + 1, nx), dtype=np.float64) # (A, x)
    out[key]['sig_xy' ] = np.zeros((nx + 1, nx), dtype=np.float64)
    out[key]['sig_yy' ] = np.zeros((nx + 1, nx), dtype=np.float64)
    out[key]['epsp'   ] = np.zeros((nx + 1, nx), dtype=np.float64)
    out[key]['epspdot'] = np.zeros((nx + 1, nx), dtype=np.float64)
    out[key]['S'      ] = np.zeros((nx + 1, nx), dtype=np.int64)
    out[key]['moved'  ] = np.zeros((nx + 1, nx), dtype=np.int64)

for key in moving_average:

    moving_average[key]['sig_xx' ] = np.zeros((nx + 1), dtype=np.float64) # (A, x)
    moving_average[key]['sig_xy' ] = np.zeros((nx + 1), dtype=np.float64)
    moving_average[key]['sig_yy' ] = np.zeros((nx + 1), dtype=np.float64)
    moving_average[key]['epsp'   ] = np.zeros((nx + 1), dtype=np.float64)
    moving_average[key]['epspdot'] = np.zeros((nx + 1), dtype=np.float64)
    moving_average[key]['S'      ] = np.zeros((nx + 1), dtype=np.int64)
    moving_average[key]['moved'  ] = np.zeros((nx + 1), dtype=np.int64)

# ---------------
# loop over files
# ---------------

edx = np.empty((2, nx), dtype='int')
edx[0, :] = np.arange(nx)
dA = 50

for ifile, file in enumerate(tqdm.tqdm(files)):

    idnum = os.path.basename(file).split('_')[0]

    with h5py.File(os.path.join(source_dir, '{0:s}.hdf5'.format(idnum)), 'r') as data:

        epsy = data['/cusp/epsy'][...]
        epsy = np.hstack(( - epsy[:, 0].reshape(-1, 1), epsy ))
        uuid = data["/uuid"].asstr()[...]

    with h5py.File(file, 'r') as data:

        assert uuid == data["/meta/uuid"].asstr()[...]

        A = data["/sync-A/stored"][...]
        T = data["/sync-A/global/iiter"][...]

        assert np.max(A) == nx

        idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(A))][...]

        edx[1, :] = idx0
        i = np.ravel_multi_index(edx, epsy.shape)
        epsy_l = epsy.flat[i]
        epsy_r = epsy.flat[i + 1]
        epsp0 = 0.5 * (epsy_l + epsy_r)
        epspdot = np.zeros_like(epsp0)

        norm[A] += 1

        for ia, a in enumerate(A):

            if "/sync-A/element/{0:d}/sig_xx".format(a) in data:
                sig_xx = data["/sync-A/element/{0:d}/sig_xx".format(a)][...][plastic]
                sig_xy = data["/sync-A/element/{0:d}/sig_xy".format(a)][...][plastic]
                sig_yy = data["/sync-A/element/{0:d}/sig_yy".format(a)][...][plastic]
            else:
                sig_xx = data["/sync-A/plastic/{0:d}/sig_xx".format(a)][...]
                sig_xy = data["/sync-A/plastic/{0:d}/sig_xy".format(a)][...]
                sig_yy = data["/sync-A/plastic/{0:d}/sig_yy".format(a)][...]

            idx = data['/sync-A/plastic/{0:d}/idx'.format(a)][...]

            edx[1, :] = idx
            i = np.ravel_multi_index(edx, epsy.shape)
            epsy_l = epsy.flat[i]
            epsy_r = epsy.flat[i + 1]
            epsp = 0.5 * (epsy_l + epsy_r)

            if '/sync-A/plastic/{0:d}/epsp'.format(a) in data and a > 0:
                assert np.allclose(epsp, data['/sync-A/plastic/{0:d}/epsp'.format(a)][...])

            if ia >= dA:

                a_n = A[ia - dA]
                idx_n = data['/sync-A/plastic/{0:d}/idx'.format(a_n)][...]

                edx[1, :] = idx_n
                i = np.ravel_multi_index(edx, epsy.shape)
                epsy_l = epsy.flat[i]
                epsy_r = epsy.flat[i + 1]
                epsp_n = 0.5 * (epsy_l + epsy_r)

                if '/sync-A/plastic/{0:d}/epsp'.format(a) in data:
                    assert np.allclose(epsp_n, data['/sync-A/plastic/{0:d}/epsp'.format(a_n)][...]) or a_n == 0

            moved = idx0 != idx
            renum = renumber(np.argwhere(moved).ravel(), nx)
            moved = moved.astype(np.int64)

            if ia >= dA:
                epspdot = (epsp - epsp_n) / (T[a] - T[a_n])

            sig_xx  = sig_xx[renum]
            sig_xy  = sig_xy[renum]
            sig_yy  = sig_yy[renum]
            S       = (idx - idx0)[renum].astype(np.int64)
            epsp    = (epsp - epsp0)[renum]
            epspdot = epspdot[renum]
            moved   = moved[renum]

            out['1st']['sig_xx' ][a, :] += sig_xx
            out['1st']['sig_xy' ][a, :] += sig_xy
            out['1st']['sig_yy' ][a, :] += sig_yy
            out['1st']['S'      ][a, :] += S
            out['1st']['epsp'   ][a, :] += epsp
            out['1st']['epspdot'][a, :] += epspdot
            out['1st']['moved'  ][a, :] += moved

            out['2nd']['sig_xx' ][a, :] += sig_xx ** 2.0
            out['2nd']['sig_xy' ][a, :] += sig_xy ** 2.0
            out['2nd']['sig_yy' ][a, :] += sig_yy ** 2.0
            out['2nd']['S'      ][a, :] += S ** 2
            out['2nd']['epsp'   ][a, :] += epsp ** 2.0
            out['2nd']['epspdot'][a, :] += epspdot ** 2.0
            out['2nd']['moved'  ][a, :] += moved ** 2

            moving_average['1st']['sig_xx' ] = np.sum(moved * sig_xx )
            moving_average['1st']['sig_xy' ] = np.sum(moved * sig_xy )
            moving_average['1st']['sig_yy' ] = np.sum(moved * sig_yy )
            moving_average['1st']['S'      ] = np.sum(moved * S      )
            moving_average['1st']['epsp'   ] = np.sum(moved * epsp   )
            moving_average['1st']['epspdot'] = np.sum(moved * epspdot)
            moving_average['1st']['moved'  ] = np.sum(moved * moved  )

            moving_average['2nd']['sig_xx' ] = np.sum((moved * sig_xx) ** 2.0 )
            moving_average['2nd']['sig_xy' ] = np.sum((moved * sig_xy) ** 2.0 )
            moving_average['2nd']['sig_yy' ] = np.sum((moved * sig_yy) ** 2.0 )
            moving_average['2nd']['S'      ] = np.sum((moved * S) ** 2        )
            moving_average['2nd']['epsp'   ] = np.sum((moved * epsp) ** 2.0   )
            moving_average['2nd']['epspdot'] = np.sum((moved * epspdot) ** 2.0)
            moving_average['2nd']['moved'  ] = np.sum((moved * moved) ** 2    )

# ---------------------------------------------
# select only measurements with sufficient data
# ---------------------------------------------

idx = np.argwhere(norm > 30).ravel()

A = np.arange(nx + 1)
norm = norm[idx].astype(np.float64)
norm_x = norm_x[idx].astype(np.float64)
A = A[idx].astype(np.float64)

for key in out:
    for field in out[key]:
        out[key][field] = out[key][field][idx, :]

# ----------------------
# store support function
# ----------------------

def store(data, key,
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_epspdot, m_S, m_moved,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_epspdot, v_S, v_moved):

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
    data['/{0:s}/avr/sig_xx'.format(key)] = m_sig_xx / sig0
    data['/{0:s}/avr/sig_yy'.format(key)] = m_sig_yy / sig0
    data['/{0:s}/avr/sig_xy'.format(key)] = m_sig_xy / sig0
    data['/{0:s}/avr/sig_eq'.format(key)] = m_sig_eq / sig0
    data['/{0:s}/avr/sig_m'.format(key)] = m_sig_m / sig0
    data['/{0:s}/avr/epsp'.format(key)] = m_epsp / eps0
    data['/{0:s}/avr/epspdot'.format(key)] = m_epspdot / eps0 / (dt / t0)
    data['/{0:s}/avr/S'.format(key)] = m_S
    data['/{0:s}/avr/moved'.format(key)] = m_moved

    # store variance
    data['/{0:s}/std/sig_xx'.format(key)] = v_sig_xx / sig0
    data['/{0:s}/std/sig_yy'.format(key)] = v_sig_yy / sig0
    data['/{0:s}/std/sig_xy'.format(key)] = v_sig_xy / sig0
    data['/{0:s}/std/sig_eq'.format(key)] = np.sqrt(np.abs(v_sig_eq)) / sig0
    data['/{0:s}/std/sig_m'.format(key)] = np.sqrt(np.abs(v_sig_m)) / sig0
    data['/{0:s}/std/epsp'.format(key)] = np.sqrt(np.abs(v_epsp)) / eps0
    data['/{0:s}/std/epspdot'.format(key)] = np.sqrt(np.abs(v_epsp)) / eps0 / (dt / t0)
    data['/{0:s}/std/S'.format(key)] = np.sqrt(np.abs(v_S))
    data['/{0:s}/std/moved'.format(key)] = np.sqrt(np.abs(v_moved))

# -----
# store
# -----

def compute_mean(first, norm):
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
    data['/dA'] = dA

    # ---------

    # allow broadcasting
    norm = norm.reshape((-1, 1))
    A = A.reshape((-1, 1))

    # compute mean
    m_sig_xx  = compute_mean(out['1st']['sig_xx' ], norm)
    m_sig_xy  = compute_mean(out['1st']['sig_xy' ], norm)
    m_sig_yy  = compute_mean(out['1st']['sig_yy' ], norm)
    m_S       = compute_mean(out['1st']['S'      ], norm)
    m_epsp    = compute_mean(out['1st']['epsp'   ], norm)
    m_epspdot = compute_mean(out['1st']['epspdot'], norm)
    m_moved   = compute_mean(out['1st']['moved'  ], norm)

    # compute variance
    v_sig_xx  = compute_variance(out['1st']['sig_xx' ], out['2nd']['sig_xx' ], norm)
    v_sig_xy  = compute_variance(out['1st']['sig_xy' ], out['2nd']['sig_xy' ], norm)
    v_sig_yy  = compute_variance(out['1st']['sig_yy' ], out['2nd']['sig_yy' ], norm)
    v_S       = compute_variance(out['1st']['S'      ], out['2nd']['S'      ], norm)
    v_epsp    = compute_variance(out['1st']['epsp'   ], out['2nd']['epsp'   ], norm)
    v_epspdot = compute_variance(out['1st']['epspdot'], out['2nd']['epspdot'], norm)
    v_moved   = compute_variance(out['1st']['moved'  ], out['2nd']['moved'  ], norm)

    # store
    store(data, 'element',
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_epspdot, m_S, m_moved,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_epspdot, v_S, v_moved)

    # ---------

    # disable broadcasting
    norm = norm.ravel()
    A = A.ravel()

    # compute mean
    m_sig_xx  = compute_mean(np.sum(out['1st']['sig_xx' ], axis=1), norm * nx)
    m_sig_xy  = compute_mean(np.sum(out['1st']['sig_xy' ], axis=1), norm * nx)
    m_sig_yy  = compute_mean(np.sum(out['1st']['sig_yy' ], axis=1), norm * nx)
    m_S       = compute_mean(np.sum(out['1st']['S'      ], axis=1), norm * nx)
    m_epsp    = compute_mean(np.sum(out['1st']['epsp'   ], axis=1), norm * nx)
    m_epspdot = compute_mean(np.sum(out['1st']['epspdot'], axis=1), norm * nx)
    m_moved   = compute_mean(np.sum(out['1st']['moved'  ], axis=1), norm * nx)

    # compute variance
    v_sig_xx  = compute_variance(np.sum(out['1st']['sig_xx' ], axis=1), np.sum(out['2nd']['sig_xx' ], axis=1), norm * nx)
    v_sig_xy  = compute_variance(np.sum(out['1st']['sig_xy' ], axis=1), np.sum(out['2nd']['sig_xy' ], axis=1), norm * nx)
    v_sig_yy  = compute_variance(np.sum(out['1st']['sig_yy' ], axis=1), np.sum(out['2nd']['sig_yy' ], axis=1), norm * nx)
    v_S       = compute_variance(np.sum(out['1st']['S'      ], axis=1), np.sum(out['2nd']['S'      ], axis=1), norm * nx)
    v_epsp    = compute_variance(np.sum(out['1st']['epsp'   ], axis=1), np.sum(out['2nd']['epsp'   ], axis=1), norm * nx)
    v_epspdot = compute_variance(np.sum(out['1st']['epspdot'], axis=1), np.sum(out['2nd']['epspdot'], axis=1), norm * nx)
    v_moved   = compute_variance(np.sum(out['1st']['moved'  ], axis=1), np.sum(out['2nd']['moved'  ], axis=1), norm * nx)

    # store
    store(data, 'layer',
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_epspdot, m_S, m_moved,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_epspdot, v_S, v_moved)

    # ---------

    # compute mean
    m_sig_xx  = compute_mean(moving_average['1st']['sig_xx' ], moving_average['1st']['moved'])
    m_sig_xy  = compute_mean(moving_average['1st']['sig_xy' ], moving_average['1st']['moved'])
    m_sig_yy  = compute_mean(moving_average['1st']['sig_yy' ], moving_average['1st']['moved'])
    m_S       = compute_mean(moving_average['1st']['S'      ], moving_average['1st']['moved'])
    m_epsp    = compute_mean(moving_average['1st']['epsp'   ], moving_average['1st']['moved'])
    m_epspdot = compute_mean(moving_average['1st']['epspdot'], moving_average['1st']['moved'])
    m_moved   = compute_mean(moving_average['1st']['moved'  ], moving_average['1st']['moved'])

    # compute variance
    v_sig_xx  = compute_variance(moving_average['1st']['sig_xx' ], moving_average['2nd']['sig_xx' ], moving_average['1st']['moved'])
    v_sig_xy  = compute_variance(moving_average['1st']['sig_xy' ], moving_average['2nd']['sig_xy' ], moving_average['1st']['moved'])
    v_sig_yy  = compute_variance(moving_average['1st']['sig_yy' ], moving_average['2nd']['sig_yy' ], moving_average['1st']['moved'])
    v_S       = compute_variance(moving_average['1st']['S'      ], moving_average['2nd']['S'      ], moving_average['1st']['moved'])
    v_epsp    = compute_variance(moving_average['1st']['epsp'   ], moving_average['2nd']['epsp'   ], moving_average['1st']['moved'])
    v_epspdot = compute_variance(moving_average['1st']['epspdot'], moving_average['2nd']['epspdot'], moving_average['1st']['moved'])
    v_moved   = compute_variance(moving_average['1st']['moved'  ], moving_average['2nd']['moved'  ], moving_average['1st']['moved'])

    # store
    store(data, 'moving',
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_epspdot, m_S, m_moved,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_epspdot, v_S, v_moved)

    # ---------

    # remove data outside crack
    for i, a in enumerate(A):
        a = int(a)
        mid = int((nx - nx % 2) / 2)
        il = int(mid - a / 2)
        iu = int(mid + a / 2)
        for key in out:
            for field in out[key]:
                out[key][field] = out[key][field].astype(np.float64)
                out[key][field][i, :il] = 0.
                out[key][field][i, iu:] = 0.

    A[A == 0.0] = 1.0

    # compute mean
    m_sig_xx  = compute_mean(np.sum(out['1st']['sig_xx' ], axis=1), norm * A)
    m_sig_xy  = compute_mean(np.sum(out['1st']['sig_xy' ], axis=1), norm * A)
    m_sig_yy  = compute_mean(np.sum(out['1st']['sig_yy' ], axis=1), norm * A)
    m_S       = compute_mean(np.sum(out['1st']['S'      ], axis=1), norm * A)
    m_epsp    = compute_mean(np.sum(out['1st']['epsp'   ], axis=1), norm * A)
    m_epspdot = compute_mean(np.sum(out['1st']['epspdot'], axis=1), norm * A)
    m_moved   = compute_mean(np.sum(out['1st']['moved'  ], axis=1), norm * A)

    # compute variance
    v_sig_xx  = compute_variance(np.sum(out['1st']['sig_xx' ], axis=1), np.sum(out['2nd']['sig_xx' ], axis=1) , norm * A)
    v_sig_xy  = compute_variance(np.sum(out['1st']['sig_xy' ], axis=1), np.sum(out['2nd']['sig_xy' ], axis=1) , norm * A)
    v_sig_yy  = compute_variance(np.sum(out['1st']['sig_yy' ], axis=1), np.sum(out['2nd']['sig_yy' ], axis=1) , norm * A)
    v_S       = compute_variance(np.sum(out['1st']['S'      ], axis=1), np.sum(out['2nd']['S'      ], axis=1) , norm * A)
    v_epsp    = compute_variance(np.sum(out['1st']['epsp'   ], axis=1), np.sum(out['2nd']['epsp'   ], axis=1) , norm * A)
    v_epspdot = compute_variance(np.sum(out['1st']['epspdot'], axis=1), np.sum(out['2nd']['epspdot'], axis=1) , norm * A)
    v_moved   = compute_variance(np.sum(out['1st']['moved'  ], axis=1), np.sum(out['2nd']['moved'  ], axis=1) , norm * A)

    # store
    store(data, 'crack',
          m_sig_xx, m_sig_xy, m_sig_yy, m_epsp, m_epspdot, m_S, m_moved,
          v_sig_xx, v_sig_xy, v_sig_yy, v_epsp, v_epspdot, v_S, v_moved)
