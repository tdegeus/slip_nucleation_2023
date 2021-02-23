r'''
    Get stress and yield index across the weak layer.
    Common practice::

        shelephant_dump *.hdf5
        python collect_weak-layer shelephant_dump.yaml layer

Usage:
    collect_weak-layer.py [options] <files.yaml> <output-root>

Arguments:
    <files.yaml>    Files from which to collect data.
    <output-root>   Directory to store selected data (file-structure preserved).

Options:
    -k, --key=N     Path in the YAML-file, separated by "/". [default: /]
    -i, --info=<N>  Path to EnsembleInfo (same directory as simulations). [default: EnsembleInfo.hdf5]
    -f, --force     Overwrite existing output-file.
    -h, --help      Print help.
'''

import os
import sys
import docopt
import click
import h5py
import numpy as np
import GooseHDF5 as g5
import shelephant
import tqdm
from FrictionQPotFEM.UniformSingleLayer2d import HybridSystem
import GooseFEM as gf
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


if __name__ == '__main__':

    args = docopt.docopt(__doc__)

    source = args['<files.yaml>']
    key = list(filter(None, args['--key'].split('/')))
    sources = shelephant.YamlGetItem(source, key)
    root = args['<output-root>']
    destinations = shelephant.PrefixPaths(root, sources)
    info = args['--info']
    source_dir = os.path.dirname(info)

    shelephant.CheckAllIsFile(sources)
    shelephant.OverWrite(destinations)
    shelephant.MakeDirs(shelephant.DirNames(destinations))

    for ifile, (source, destination) in enumerate(zip(tqdm.tqdm(sources), destinations)):

        with h5py.File(destination, 'w') as out:

            with h5py.File(source, 'r') as data:

                idnum = data["/meta/id"][...]
                uuid = data["/meta/uuid"].asstr()[...]
                idname = "id={0:03d}.hdf5".format(idnum)

                system = LoadSystem(os.path.join(source_dir, idname), uuid)

                if ifile == 0:
                    plastic = system.plastic()
                    N = plastic.size
                    quad = system.quad()
                    dV = system.dV()
                    dV_plastic = quad.AsTensor(2, dV[plastic, :])
                    dV = quad.AsTensor(2, dV)

                A = data["/sync-A/stored"][...]
                out["/sync-A/stored"] = A
                out["/sync-A/global/iiter"] = data["/sync-A/global/iiter"][...]
                sig_xx = np.zeros((N + 1), dtype=np.float64)
                sig_xy = np.zeros((N + 1), dtype=np.float64)
                sig_yy = np.zeros((N + 1), dtype=np.float64)

                for a in A:

                    system.setU(data["/sync-A/{0:d}/u".format(a)][...])

                    Sig = np.average(system.Sig(), weights=dV, axis=(0, 1))
                    sig_xx[a] = Sig[0, 0]
                    sig_xy[a] = Sig[0, 1]
                    sig_yy[a] = Sig[1, 1]

                    Sig = np.average(system.plastic_Sig(), weights=dV_plastic, axis=1)
                    out["/sync-A/plastic/{0:d}/sig_xx".format(a)] = Sig[:, 0, 0]
                    out["/sync-A/plastic/{0:d}/sig_xy".format(a)] = Sig[:, 0, 1]
                    out["/sync-A/plastic/{0:d}/sig_yy".format(a)] = Sig[:, 1, 1]
                    out["/sync-A/plastic/{0:d}/idx".format(a)] = system.plastic_CurrentIndex()[:, 0]

                out["/sync-A/global/sig_xx"] = sig_xx
                out["/sync-A/global/sig_xy"] = sig_xy
                out["/sync-A/global/sig_yy"] = sig_yy

                T = np.sort(data["/sync-t/stored"][...])
                out["/sync-t/stored"] = T
                out["/sync-t/global/iiter"] = data["/sync-t/global/iiter"][...]
                sig_xx = np.empty((T.size), dtype=np.float64)
                sig_xy = np.empty((T.size), dtype=np.float64)
                sig_yy = np.empty((T.size), dtype=np.float64)

                for i, t in enumerate(T):

                    system.setU(data["/sync-t/{0:d}/u".format(t)][...])

                    Sig = np.average(system.Sig(), weights=dV, axis=(0, 1))
                    sig_xx[i] = Sig[0, 0]
                    sig_xy[i] = Sig[0, 1]
                    sig_yy[i] = Sig[1, 1]

                    Sig = np.average(system.plastic_Sig(), weights=dV_plastic, axis=1)
                    out["/sync-t/plastic/{0:d}/sig_xx".format(t)] = Sig[:, 0, 0]
                    out["/sync-t/plastic/{0:d}/sig_xy".format(t)] = Sig[:, 0, 1]
                    out["/sync-t/plastic/{0:d}/sig_yy".format(t)] = Sig[:, 1, 1]
                    out["/sync-t/plastic/{0:d}/idx".format(t)] = system.plastic_CurrentIndex()[:, 0]


                out["/sync-t/global/sig_xx"] = sig_xx
                out["/sync-t/global/sig_xy"] = sig_xy
                out["/sync-t/global/sig_yy"] = sig_yy

                g5.copydatasets(data, out, list(g5.getdatasets(data, "/meta")))
                if "/meta/versions/CrackEvolution_raw_stress" not in data:
                    out["/meta/versions/CrackEvolution_raw_stress"] = data["/git/run"][...]
                out["/meta/versions/collect_weak-layer.py"] = get_version(root='..', relative_to=__file__)

