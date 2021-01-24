r'''
Collected scatter data at the final increment and at A = N.

Usage:
  collect_final.py [options] <files>...

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
import GMatElastoPlasticQPot.Cartesian2d as gmat

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
    dt = data['/normalisation/dt'][...]
    t0 = data['/normalisation/t0'][...]
    eps0 = data['/normalisation/eps0'][...]
    sig0 = data['/normalisation/sig0'][...]
    nx = int(data['/normalisation/N'][...])

# ==================================================================================================
# ensemble average
# ==================================================================================================

# -------------------------------------------------------------------------
# initialise normalisation, and sum of first and second statistical moments
# -------------------------------------------------------------------------

final_ret = {'global': {}, 'plastic': {}}
final_stored_global = np.zeros(len(files), dtype=np.int)
final_stored_epsp = np.zeros(len(files), dtype=np.int)
final_stored_plastic = np.zeros(len(files), dtype=np.int)

final_ret['global']['sig_xx'] = np.zeros(len(files), dtype=np.float)
final_ret['global']['sig_xy'] = np.zeros(len(files), dtype=np.float)
final_ret['global']['sig_yy'] = np.zeros(len(files), dtype=np.float)
final_ret['global']['iiter'] = np.zeros(len(files), dtype=np.int)

final_ret['plastic']['sig_xx'] = np.zeros(len(files), dtype=np.float)
final_ret['plastic']['sig_xy'] = np.zeros(len(files), dtype=np.float)
final_ret['plastic']['sig_yy'] = np.zeros(len(files), dtype=np.float)
final_ret['plastic']['epsp'] = np.zeros(len(files), dtype=np.float)
final_ret['plastic']['epspdot_full'] = np.zeros(len(files), dtype=np.float)

system_ret = {'global': {}, 'plastic': {}}
system_stored_global = np.zeros(len(files), dtype=np.int)
system_stored_epsp = np.zeros(len(files), dtype=np.int)
system_stored_plastic = np.zeros(len(files), dtype=np.int)

system_ret['global']['sig_xx'] = np.zeros(len(files), dtype=np.float)
system_ret['global']['sig_xy'] = np.zeros(len(files), dtype=np.float)
system_ret['global']['sig_yy'] = np.zeros(len(files), dtype=np.float)
system_ret['global']['iiter'] = np.zeros(len(files), dtype=np.int)

system_ret['plastic']['sig_xx'] = np.zeros(len(files), dtype=np.float)
system_ret['plastic']['sig_xy'] = np.zeros(len(files), dtype=np.float)
system_ret['plastic']['sig_yy'] = np.zeros(len(files), dtype=np.float)
system_ret['plastic']['epsp'] = np.zeros(len(files), dtype=np.float)
system_ret['plastic']['epspdot_full'] = np.zeros(len(files), dtype=np.float)
system_ret['plastic']['epspdot'] = np.zeros(len(files), dtype=np.float)

# ---------------
# loop over files
# ---------------

for ifile, file in enumerate(files):

    print('({0:3d}/{1:3d}) {2:s}'.format(ifile + 1, len(files), file))

    with h5py.File(file, 'r') as data:

        a = data["/sync-A/stored"][...]
        Am = a[int(len(a) / 2.0)]
        A0 = data["/sync-A/stored"][0]
        A = data["/sync-A/stored"][-1]

        if A == nx:

            system_stored_global[ifile] = 1

            system_ret['global']['sig_xx'][ifile] = data["/sync-A/global/sig_xx"][A]
            system_ret['global']['sig_xy'][ifile] = data["/sync-A/global/sig_xy"][A]
            system_ret['global']['sig_yy'][ifile] = data["/sync-A/global/sig_yy"][A]
            system_ret['global']['iiter'][ifile] = data["/sync-A/global/iiter"][A]

            if '/sync-A/plastic/{0:d}/epsp'.format(A) in data:

                epsp = data['/sync-A/plastic/{0:d}/epsp'.format(A)][...]
                epsp0 = data['/sync-A/plastic/{0:d}/epsp'.format(A0)][...]
                epspm = data['/sync-A/plastic/{0:d}/epsp'.format(Am)][...]
                t = data["/sync-A/global/iiter"][A]
                tm = data["/sync-A/global/iiter"][Am]

                system_ret['plastic']['epsp'][ifile] = np.mean(epsp - epsp0)
                system_ret['plastic']['epspdot_full'][ifile] = np.mean(epsp - epsp0) / t
                system_ret['plastic']['epspdot'][ifile] = np.mean(epsp - epspm) / (t - tm)

                system_stored_epsp[ifile] = 1

            if "/sync-A/element/{0:d}/sig_xx".format(A) in data:

                sig_xx = data["/sync-A/element/{0:d}/sig_xx".format(A)][...][plastic]
                sig_xy = data["/sync-A/element/{0:d}/sig_xy".format(A)][...][plastic]
                sig_yy = data["/sync-A/element/{0:d}/sig_yy".format(A)][...][plastic]

                system_ret['plastic']['sig_xx'][ifile] = np.mean(sig_xx)
                system_ret['plastic']['sig_xy'][ifile] = np.mean(sig_xy)
                system_ret['plastic']['sig_yy'][ifile] = np.mean(sig_yy)

                system_stored_plastic[ifile] = 1

            elif "/sync-A/plastic/{0:d}/sig_xx".format(A) in data:

                sig_xx = data["/sync-A/plastic/{0:d}/sig_xx".format(A)][...]
                sig_xy = data["/sync-A/plastic/{0:d}/sig_xy".format(A)][...]
                sig_yy = data["/sync-A/plastic/{0:d}/sig_yy".format(A)][...]

                system_ret['plastic']['sig_xx'][ifile] = np.mean(sig_xx)
                system_ret['plastic']['sig_xy'][ifile] = np.mean(sig_xy)
                system_ret['plastic']['sig_yy'][ifile] = np.mean(sig_yy)

                system_stored_plastic[ifile] = 1

        if "/sync-t/stored" in data:

            final_stored_global[ifile] = 1

            T0 = data["/sync-t/stored"][0]
            T = data["/sync-t/stored"][-1]

            final_ret['global']['sig_xx'][ifile] = data["/sync-t/global/sig_xx"][T]
            final_ret['global']['sig_xy'][ifile] = data["/sync-t/global/sig_xy"][T]
            final_ret['global']['sig_yy'][ifile] = data["/sync-t/global/sig_yy"][T]
            final_ret['global']['iiter'][ifile] = data["/sync-t/global/iiter"][T]

            if '/sync-t/plastic/{0:d}/epsp'.format(T) in data:

                epsp = data['/sync-t/plastic/{0:d}/epsp'.format(T)][...]
                epsp0 = data['/sync-t/plastic/{0:d}/epsp'.format(T0)][...]
                t = data["/sync-t/global/iiter"][T]

                final_ret['plastic']['epsp'][ifile] = np.mean(epsp - epsp0)
                final_ret['plastic']['epspdot_full'][ifile] = np.mean(epsp - epsp0) / t

                final_stored_epsp[ifile] = 1

            if "/sync-t/element/{0:d}/sig_xx".format(T) in data:

                sig_xx = data["/sync-t/element/{0:d}/sig_xx".format(T)][...][plastic]
                sig_xy = data["/sync-t/element/{0:d}/sig_xy".format(T)][...][plastic]
                sig_yy = data["/sync-t/element/{0:d}/sig_yy".format(T)][...][plastic]

                final_ret['plastic']['sig_xx'][ifile] = np.mean(sig_xx)
                final_ret['plastic']['sig_xy'][ifile] = np.mean(sig_xy)
                final_ret['plastic']['sig_yy'][ifile] = np.mean(sig_yy)

                final_stored_plastic[ifile] = 1

            elif "/sync-t/plastic/{0:d}/sig_xx".format(T) in data:

                sig_xx = data["/sync-t/plastic/{0:d}/sig_xx".format(T)][...]
                sig_xy = data["/sync-t/plastic/{0:d}/sig_xy".format(T)][...]
                sig_yy = data["/sync-t/plastic/{0:d}/sig_yy".format(T)][...]

                final_ret['plastic']['sig_xx'][ifile] = np.mean(sig_xx)
                final_ret['plastic']['sig_xy'][ifile] = np.mean(sig_xy)
                final_ret['plastic']['sig_yy'][ifile] = np.mean(sig_yy)

                final_stored_plastic[ifile] = 1

# ------------
# extract data
# ------------

idx = np.argwhere(final_stored_global).ravel()

final_ret['global']['sig_xx'] = final_ret['global']['sig_xx'][idx]
final_ret['global']['sig_xy'] = final_ret['global']['sig_xy'][idx]
final_ret['global']['sig_yy'] = final_ret['global']['sig_yy'][idx]
final_ret['global']['iiter'] = final_ret['global']['iiter'][idx]

idx = np.argwhere(final_stored_plastic).ravel()

final_ret['plastic']['sig_xx'] = final_ret['plastic']['sig_xx'][idx]
final_ret['plastic']['sig_xy'] = final_ret['plastic']['sig_xy'][idx]
final_ret['plastic']['sig_yy'] = final_ret['plastic']['sig_yy'][idx]

idx = np.argwhere(final_stored_epsp).ravel()

final_ret['plastic']['epsp'] = final_ret['plastic']['epsp'][idx]
final_ret['plastic']['epspdot_full'] = final_ret['plastic']['epspdot_full'][idx]

idx = np.argwhere(system_stored_global).ravel()

system_ret['global']['sig_xx'] = system_ret['global']['sig_xx'][idx]
system_ret['global']['sig_xy'] = system_ret['global']['sig_xy'][idx]
system_ret['global']['sig_yy'] = system_ret['global']['sig_yy'][idx]
system_ret['global']['iiter'] = system_ret['global']['iiter'][idx]

idx = np.argwhere(system_stored_plastic).ravel()

system_ret['plastic']['sig_xx'] = system_ret['plastic']['sig_xx'][idx]
system_ret['plastic']['sig_xy'] = system_ret['plastic']['sig_xy'][idx]
system_ret['plastic']['sig_yy'] = system_ret['plastic']['sig_yy'][idx]

idx = np.argwhere(system_stored_epsp).ravel()

system_ret['plastic']['epsp'] = system_ret['plastic']['epsp'][idx]
system_ret['plastic']['epspdot_full'] = system_ret['plastic']['epspdot_full'][idx]
system_ret['plastic']['epspdot'] = system_ret['plastic']['epspdot'][idx]

# -----
# store
# -----

with h5py.File(output, 'w') as data:

    Sig = np.zeros((len(system_ret['global']['sig_xx']), 2, 2))
    Sig[:, 0, 0] = system_ret['global']['sig_xx']
    Sig[:, 1, 1] = system_ret['global']['sig_yy']
    Sig[:, 0, 1] = system_ret['global']['sig_xy']
    Sig[:, 1, 0] = system_ret['global']['sig_xy']

    data['/A=N/global/iiter'] = system_ret['global']['iiter'] * dt / t0
    data['/A=N/global/sig_xx'] = system_ret['global']['sig_xx'] / sig0
    data['/A=N/global/sig_xy'] = system_ret['global']['sig_xy'] / sig0
    data['/A=N/global/sig_yy'] = system_ret['global']['sig_yy'] / sig0
    data['/A=N/global/sig_eq'] = gmat.Sigd(Sig) / sig0
    data['/A=N/global/sig_m'] = gmat.Hydrostatic(Sig) / sig0

    Sig = np.zeros((len(system_ret['plastic']['sig_xx']), 2, 2))
    Sig[:, 0, 0] = system_ret['plastic']['sig_xx']
    Sig[:, 1, 1] = system_ret['plastic']['sig_yy']
    Sig[:, 0, 1] = system_ret['plastic']['sig_xy']
    Sig[:, 1, 0] = system_ret['plastic']['sig_xy']

    data['/A=N/plastic/epsp'] = system_ret['plastic']['epsp'] / eps0
    data['/A=N/plastic/epspdot_full'] = system_ret['plastic']['epspdot_full'] / eps0 / (dt / t0)
    data['/A=N/plastic/epspdot'] = system_ret['plastic']['epspdot'] / eps0 / (dt / t0)
    data['/A=N/plastic/sig_xx'] = system_ret['plastic']['sig_xx'] / sig0
    data['/A=N/plastic/sig_xy'] = system_ret['plastic']['sig_xy'] / sig0
    data['/A=N/plastic/sig_yy'] = system_ret['plastic']['sig_yy'] / sig0
    data['/A=N/plastic/sig_eq'] = gmat.Sigd(Sig) / sig0
    data['/A=N/plastic/sig_m'] = gmat.Hydrostatic(Sig) / sig0

    Sig = np.zeros((len(final_ret['global']['sig_xx']), 2, 2))
    Sig[:, 0, 0] = final_ret['global']['sig_xx']
    Sig[:, 1, 1] = final_ret['global']['sig_yy']
    Sig[:, 0, 1] = final_ret['global']['sig_xy']
    Sig[:, 1, 0] = final_ret['global']['sig_xy']

    data['/final/global/iiter'] = final_ret['global']['iiter'] * dt / t0
    data['/final/global/sig_xx'] = final_ret['global']['sig_xx'] / sig0
    data['/final/global/sig_xy'] = final_ret['global']['sig_xy'] / sig0
    data['/final/global/sig_yy'] = final_ret['global']['sig_yy'] / sig0
    data['/final/global/sig_eq'] = gmat.Sigd(Sig) / sig0
    data['/final/global/sig_m'] = gmat.Hydrostatic(Sig) / sig0

    Sig = np.zeros((len(final_ret['plastic']['sig_xx']), 2, 2))
    Sig[:, 0, 0] = final_ret['plastic']['sig_xx']
    Sig[:, 1, 1] = final_ret['plastic']['sig_yy']
    Sig[:, 0, 1] = final_ret['plastic']['sig_xy']
    Sig[:, 1, 0] = final_ret['plastic']['sig_xy']

    data['/final/plastic/epsp'] = final_ret['plastic']['epsp'] / eps0
    data['/final/plastic/epspdot_full'] = final_ret['plastic']['epspdot_full'] / eps0 / (dt / t0)
    data['/final/plastic/sig_xx'] = final_ret['plastic']['sig_xx'] / sig0
    data['/final/plastic/sig_xy'] = final_ret['plastic']['sig_xy'] / sig0
    data['/final/plastic/sig_yy'] = final_ret['plastic']['sig_yy'] / sig0
    data['/final/plastic/sig_eq'] = gmat.Sigd(Sig) / sig0
    data['/final/plastic/sig_m'] = gmat.Hydrostatic(Sig) / sig0
