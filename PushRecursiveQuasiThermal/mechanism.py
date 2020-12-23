import os
import subprocess
import h5py
import matplotlib.pyplot as plt
import GooseMPL as gplt
import numpy as np
import GMatElastoPlasticQPot.Cartesian2d as GMat

plt.style.use(['goose', 'goose-latex'])

with h5py.File('EnsembleInfo.hdf5', 'r') as data:
    sig0 = float(data['/normalisation/sig0'][...])
    dt = float(data['/normalisation/dt'][...])
    l0 = float(data['/normalisation/l0'][...])
    cs = float(data['/normalisation/cs'][...])
    N = float(data['/normalisation/N'][...])

with h5py.File('CrackEvolutionInfo.hdf5', 'r') as data:
    sigc = data['/detail/last'][0]

files = sorted(list(filter(None, subprocess.check_output(
  "find . -iname 'id*push=*.hdf5'", shell=True).decode('utf-8').split('\n'))))

for ifile, file in enumerate(files):

    with h5py.File(file, 'r') as data:

        if ifile == 0:
            xx = data['/overview/global/sig'].attrs['xx'][...]
            xy = data['/overview/global/sig'].attrs['xy'][...]
            yy = data['/overview/global/sig'].attrs['yy'][...]

        fig, axes = gplt.subplots(ncols=2, scale_y=2)

        if 'event' not in data:
            continue
        if 'global' not in data['event']:
            continue
        if 'A' not in data['event']['global']:
            continue

        r = data['/trigger/r'][...]
        t = data['/trigger/iiter'][...] * dt * cs / (l0 * N)
        axes[0].plot(r, t, c='r', marker='s', ls='none', markersize=5, rasterized=True)

        A = data['/event/global/A'][...]
        t = data['/event/global/iiter'][...] * dt * cs / (l0 * N)
        r = data['/event/r'][...]
        s = data['/event/step'][...]

        p = np.argwhere(s >= 0).ravel()
        n = np.argwhere(s < 0).ravel()

        tlim = np.max(t)

        axes[0].set_xlim([0, N])
        axes[0].set_ylim([0, tlim])
        axes[0].set_xlabel(r'$r$')
        axes[0].set_ylabel(r'$t$')

        axes[1].set_xlim([0, 0.5])
        axes[1].set_ylim([0, tlim])
        axes[1].set_xlabel(r'$\sigma$')
        axes[1].set_ylabel(r'$t$')

        axes[0].plot(r[n], t[n], c='b', marker='.', ls='none', markersize=1, rasterized=True)
        axes[0].plot(r[p], t[p], c='k', marker='.', ls='none', markersize=1, rasterized=True)

        t = data['/overview/global/iiter'][...] * dt * cs / (l0 * N)

        sig = data['/overview/weak/sig'][...]
        Sig = np.zeros([sig.shape[1], 2, 2])
        Sig[:, 0, 0] = sig[xx, :]
        Sig[:, 0, 1] = sig[xy, :]
        Sig[:, 1, 0] = sig[xy, :]
        Sig[:, 1, 1] = sig[yy, :]
        sig_weak = GMat.Sigd(Sig)

        sig = data['/overview/global/sig'][...]
        Sig = np.zeros([sig.shape[1], 2, 2])
        Sig[:, 0, 0] = sig[xx, :]
        Sig[:, 0, 1] = sig[xy, :]
        Sig[:, 1, 0] = sig[xy, :]
        Sig[:, 1, 1] = sig[yy, :]
        sig_glob = GMat.Sigd(Sig)

        axes[1].plot(sig_weak / sig0, t, c='r', lw=1)
        axes[1].plot(sig_glob / sig0, t, c='k')

        axes[1].plot([sigc, sigc], [0, tlim], c='b', ls='--')
        axes[1].plot([0.8 * sigc, 0.8 * sigc], [0, tlim], c='g', ls='--')
        axes[1].plot([0.6 * sigc, 0.6 * sigc], [0, tlim], c='c', ls='--')


        axes[1].plot(axes[1].get_xlim(), [np.max(t), np.max(t)], c='g', ls='-', lw=1)
        axes[0].plot(axes[0].get_xlim(), [np.max(t), np.max(t)], c='g', ls='-', lw=1)

        print(file)
        fig.savefig('{0:s}'.format(os.path.basename(file).replace('.hdf5', '.pdf')))
        plt.close(fig)

