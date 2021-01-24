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
# ensemble average
# ==================================================================================================

left = int((nx - nx % 2) / 2 - 100)
right = int((nx - nx % 2) / 2 + 100 + 1)
ret = np.zeros((len(files), nx + 1, nx), dtype='float')
norm = np.zeros((len(files), nx + 1, nx), dtype='float')
pbar = tqdm.tqdm(files)
edx = np.empty((2, nx), dtype='int')
edx[0, :] = np.arange(nx)
dA = 50

for ifile, file in enumerate(pbar):

    pbar.set_description(file)

    sim = os.path.basename(file).split('_')[0]

    with h5py.File('../{0:s}.hdf5'.format(sim), 'r') as data:
        epsy = data['/cusp/epsy'][...]
        epsy = np.hstack(( - epsy[:, 0].reshape(-1, 1), epsy ))

    with h5py.File(file, 'r') as data:

        A = data["/sync-A/stored"][...]
        T = data["/sync-A/global/iiter"][...]

        idx0 = data['/sync-A/plastic/{0:d}/idx'.format(np.min(A))][...]

        norm[ifile, A[dA:], :] += 1

        for a_n, a in zip(A[:-dA], A[dA:]):

            idx = data['/sync-A/plastic/{0:d}/idx'.format(a)][...]
            idx_n = data['/sync-A/plastic/{0:d}/idx'.format(a_n)][...]

            edx[1, :] = idx
            i = np.ravel_multi_index(edx, epsy.shape)
            epsy_l = epsy.flat[i]
            epsy_r = epsy.flat[i + 1]
            epsp = 0.5 * (epsy_l + epsy_r)

            edx[1, :] = idx_n
            i = np.ravel_multi_index(edx, epsy.shape)
            epsy_l = epsy.flat[i]
            epsy_r = epsy.flat[i + 1]
            epsp_n = 0.5 * (epsy_l + epsy_r)

            if '/sync-A/plastic/{0:d}/epsp'.format(a) in data:
                assert np.allclose(epsp, data['/sync-A/plastic/{0:d}/epsp'.format(a)][...])
                assert np.allclose(epsp_n, data['/sync-A/plastic/{0:d}/epsp'.format(a_n)][...]) or a_n == 0

            renum = renumber(np.argwhere(idx0 != idx).ravel(), nx)
            epsp = epsp[renum]
            epsp_n = epsp_n[renum]

            ret[ifile, a, :] = (epsp - epsp_n) / (T[a] - T[a_n])

# ==================================================================================================
# store
# ==================================================================================================

with h5py.File(output, 'w') as data:

    A = np.arange(nx + 1)

    A = A[dA:]
    ret = ret[:, dA:, :]
    norm = norm[:, dA:]

    ret = ret / eps0 / (dt / t0)

    data['/raw/epspdot'] = ret
    data['/raw/norm'] = norm

    data['/epspdot/r'] = np.average(ret, weights=norm, axis=0)
    data['/epspdot/center'] = np.average(ret[:, :, left: right], weights=norm[:, :, left: right], axis=(0, 2))
    data['/epspdot/plastic'] = np.average(ret, weights=norm, axis=(0, 2))
    data['/A'] = A

