r'''
    Collect different energy contributions.

Usage:
    collect_energy.py [options] <files.yaml>

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
import h5py
import numpy as np
import enstat.mean
import GooseFEM as gf
import shelephant
import tqdm
from FrictionQPotFEM.UniformSingleLayer2d import HybridSystem
from setuptools_scm import get_version


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

                if ifile == 0:
                    dV = system.quad().dV()
                    elastic = system.elastic()
                    plastic = system.plastic()
                    vector = system.vector()
                    M = system.mass().Todiagonal()

                    if "/meta/versions/CrackEvolution_raw_stress" not in data:
                        out["/meta/versions/CrackEvolution_raw_stress"] = data["/git/run"][...]
                    else:
                        out["/meta/versions/CrackEvolution_raw_stress"] = data["/meta/versions/CrackEvolution_raw_stress"][...]

                # ensemble average different "A"

                if ifile == 0:
                    A_A = np.arange(N + 1)
                    A_val = {
                        'K' : [enstat.mean.Scalar() for A in A_A],
                        'E_all' : [enstat.mean.Scalar() for A in A_A],
                        'E_elastic' : [enstat.mean.Scalar() for A in A_A],
                        'E_plastic' : [enstat.mean.Scalar() for A in A_A],
                        'E_unmoved' : [enstat.mean.Scalar() for A in A_A],
                        'E_moved' : [enstat.mean.Scalar() for A in A_A],
                    }

                stored = data["/sync-A/stored"][...]
                system.setU(data["/sync-A/{0:d}/u".format(np.min(stored))][...])
                A0 = np.min(stored)
                E0 = system.Energy()
                e0 = np.sum(E0 * dV)
                idx0 = system.plastic_CurrentIndex()[:, 0]

                for i, A in enumerate(tqdm.tqdm(A_A)):

                    if A not in stored:
                        continue

                    system.setU(data["/sync-A/{0:d}/u".format(A)][...])
                    system.setV(data["/sync-A/{0:d}/v".format(A)][...])

                    V = vector.AsDofs(system.v())
                    k = 0.5 * np.sum(M * V ** 2)

                    E = system.Energy()
                    e = np.sum(E * dV)

                    assert (e + k <= 1.001 * e0) or (A <= A0)

                    idx = system.plastic_CurrentIndex()[:, 0]
                    unmoved = plastic[idx == idx0]
                    moved = plastic[idx != idx0]

                    A_val['K'][i].add_sample(k)
                    A_val['E_all'][i].add_sample(e)
                    A_val['E_elastic'][i].add_sample(np.sum(E[elastic, :] * dV[elastic, :]))
                    A_val['E_plastic'][i].add_sample(np.sum(E[plastic, :] * dV[plastic, :]))
                    if unmoved.size > 0:
                        A_val['E_unmoved'][i].add_sample(np.sum(E[unmoved, :] * dV[unmoved, :]))
                    if moved.size > 0:
                        A_val['E_moved'][i].add_sample(np.sum(E[moved, :] * dV[moved, :]))

                # ensemble average for different "t"

                stored = data["/sync-t/stored"][...]

                if ifile == 0:
                    t_t = [i for i in stored]
                    t_val = {
                        'K' : [enstat.mean.Scalar() for t in t_t],
                        'E_all' : [enstat.mean.Scalar() for t in t_t],
                        'E_elastic' : [enstat.mean.Scalar() for t in t_t],
                        'E_plastic' : [enstat.mean.Scalar() for t in t_t],
                    }
                elif np.max(stored) > np.max(t_t):
                    col = np.argmax(stored > np.max(t_t))
                    extra = [i for i in stored[col:]]
                    t_t += extra
                    for key in t_val:
                        t_val[key] += [enstat.mean.Scalar() for t in extra]

                for i, t in enumerate(tqdm.tqdm(t_t)):

                    if t not in stored:
                        continue

                    system.setU(data["/sync-t/{0:d}/u".format(t)][...])
                    system.setV(data["/sync-t/{0:d}/v".format(t)][...])

                    V = vector.AsDofs(system.v())
                    k = 0.5 * np.sum(M * V ** 2)

                    E = system.Energy()
                    e = np.sum(E * dV)

                    if e + k > 1.01 * e0:
                        break

                    t_val['K'][i].add_sample(k)
                    t_val['E_all'][i].add_sample(e)
                    t_val['E_elastic'][i].add_sample(np.sum(E[elastic, :] * dV[elastic, :]))
                    t_val['E_plastic'][i].add_sample(np.sum(E[plastic, :] * dV[plastic, :]))
        # store

        out['/A/A'] = A_A

        for key in A_val:
            mean = np.array([i.mean() for i in A_val[key]])
            norm = np.array([i.norm() for i in A_val[key]])
            first = np.array([i.first() for i in A_val[key]])
            second = np.array([i.second() for i in A_val[key]])
            keep = norm >= 0.9 * np.max(norm)
            out['/A/{0:s}'.format(key)] = mean[keep]
            out['/A/{0:s}'.format(key)].attrs['norm'] = norm[keep]
            out['/A/{0:s}'.format(key)].attrs['first'] = first[keep]
            out['/A/{0:s}'.format(key)].attrs['second'] = second[keep]

        out['/t/t'] = t_t

        for key in t_val:
            mean = np.array([i.mean() for i in t_val[key]])
            norm = np.array([i.norm() for i in t_val[key]])
            first = np.array([i.first() for i in t_val[key]])
            second = np.array([i.second() for i in t_val[key]])
            keep = norm >= 0.9 * np.max(norm)
            out['/t/{0:s}'.format(key)] = mean[keep]
            out['/t/{0:s}'.format(key)].attrs['norm'] = norm[keep]
            out['/t/{0:s}'.format(key)].attrs['first'] = first[keep]
            out['/t/{0:s}'.format(key)].attrs['second'] = second[keep]

        try:
            version = get_version(root='..', relative_to=__file__)
        except:
            version = None

        if version:
            out["/meta/versions/collect_energy.py"] = version

if __name__ == "__main__":

    main()
