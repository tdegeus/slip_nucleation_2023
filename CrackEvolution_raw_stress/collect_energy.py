r'''
    ???

Usage:
    collect_forces.py [options] <files.yaml>

Arguments:
    <files.yaml>    Files from which to collect data.

Options:
    -o, --output=<N>    Output file. [default: output.hdf5]
    -k, --key=N         Path in the YAML-file, separated by "/". [default: /]
    -i, --info=<N>      Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
    -f, --force         Overwrite existing output-file.
    -h, --help          Print help.
'''

import os
import sys
import docopt
import click
import h5py
import numpy as np
import enstat.mean
import GooseFEM as gf
import shelephant
import tqdm
import GMatElastoPlasticQPot.Cartesian2d as gmat
from FrictionQPotFEM.UniformSingleLayer2d import HybridSystem


def LoadSystem(filename, uuid):

    with h5py.File(filename, 'r') as data:

        assert uuid == data["/uuid"].asstr()[...]

        system = HybridSystem(
            data['coor'][...],
            data['conn'][...],
            data['dofs'][...],
            data['dofsP'][...],
            data['/elastic/elem'][...],
            data['/cusp/elem'][...])

        system.setMassMatrix(data['/rho'][...])
        system.setDampingMatrix(data['/damping/alpha'][...])
        system.setElastic(data['/elastic/K'][...], data['/elastic/G'][...])
        system.setPlastic(data['/cusp/K'][...], data['/cusp/G'][...], data['/cusp/epsy'][...])
        system.setDt(data['/run/dt'][...])

        return system


def main():

    args = docopt.docopt(__doc__)

    source = args['<files.yaml>']
    key = list(filter(None, args['--key'].split('/')))
    files = shelephant.YamlGetItem(source, key)
    assert len(files) > 0
    info = args['--info']
    output = args['--output']
    source_dir = os.path.dirname(info)

    shelephant.CheckAllIsFile(files + [info])
    shelephant.OverWrite(output, args['--force'])

    # Get constants

    with h5py.File(files[0], 'r') as data:
        plastic = data['/meta/plastic'][...]
        N = len(plastic)

    # Ensemble average

    with h5py.File(output, 'w') as out:

        for ifile, file in enumerate(tqdm.tqdm(files)):

            with h5py.File(file, 'r') as data:

                idnum = data["/meta/id"][...]
                uuid = data["/meta/uuid"].asstr()[...]
                idname = "id={0:03d}.hdf5".format(idnum)
                system = LoadSystem(os.path.join(source_dir, idname), uuid)
                plastic = system.plastic()
                N = plastic.size
                stored = data["/sync-A/stored"][...]

                if ifile == 0:

                    m_A = np.arange(N + 1)
                    m_E_all = [enstat.mean.Scalar() for A in m_A]
                    m_E_elastic = [enstat.mean.Scalar() for A in m_A]
                    m_E_plastic = [enstat.mean.Scalar() for A in m_A]
                    m_E_unmoved = [enstat.mean.Scalar() for A in m_A]
                    m_E_moved = [enstat.mean.Scalar() for A in m_A]
                    m_K = [enstat.mean.Scalar() for A in m_A]

                system.setU(data["/sync-A/{0:d}/u".format(np.min(stored))][...])
                idx0 = system.plastic_CurrentIndex()[:, 0]

                if ifile == 0:

                    dV = system.quad().dV()
                    elastic = system.elastic()
                    plastic = system.plastic()
                    vector = system.vector()
                    M = system.mass().Todiagonal()

                for i, A in enumerate(tqdm.tqdm(m_A)):

                    if A not in stored:
                        continue

                    system.setU(data["/sync-A/{0:d}/u".format(A)][...])
                    system.setV(data["/sync-A/{0:d}/v".format(A)][...])
                    idx = system.plastic_CurrentIndex()[:, 0]
                    E = system.Energy()
                    unmoved = plastic[idx == idx0]
                    moved = plastic[idx != idx0]

                    m_E_all[i].add_sample(np.average(E, weights=dV))
                    m_E_elastic[i].add_sample(np.average(E[elastic, :], weights=dV[elastic, :]))
                    m_E_plastic[i].add_sample(np.average(E[plastic, :], weights=dV[plastic, :]))
                    if unmoved.size > 0:
                        m_E_unmoved[i].add_sample(np.average(E[unmoved, :], weights=dV[unmoved, :]))
                    if moved.size > 0:
                        m_E_moved[i].add_sample(np.average(E[moved, :], weights=dV[moved, :]))

                    V = vector.AsDofs(system.v())
                    m_K[i].add_sample(0.5 * np.sum(M * V ** 2))

        out['/stored'] = m_A

        for i, A in enumerate(m_A):

            out['/{0:d}/E_all'.format(A)] = m_E_all[i].mean()
            out['/{0:d}/E_elastic'.format(A)] = m_E_elastic[i].mean()
            out['/{0:d}/E_plastic'.format(A)] = m_E_plastic[i].mean()
            out['/{0:d}/E_unmoved'.format(A)] = m_E_unmoved[i].mean()
            out['/{0:d}/E_moved'.format(A)] = m_E_moved[i].mean()
            out['/{0:d}/K'.format(A)] = m_K[i].mean()


if __name__ == "__main__":

    main()
