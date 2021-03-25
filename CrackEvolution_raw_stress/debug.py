r'''
    Collect stress distribution.

Usage:
    collect_stress.py [options] <files.yaml>

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
from FrictionQPotFEM.UniformSingleLayer2d import HybridSystem
from setuptools_scm import get_version


# https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions


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

    # Define mapping (same for all input)

    for file in files:

        with h5py.File(file, 'r') as data:

            idnum = data["/meta/id"][...]
            uuid = data["/meta/uuid"].asstr()[...]
            idname = "id={0:03d}.hdf5".format(idnum)

            system = LoadSystem(os.path.join(source_dir, idname), uuid)
            plastic = system.plastic()
            N = plastic.size
            assert np.all(np.equal(plastic, data['/meta/plastic'][...]))

            M = system.mass().Todiagonal()
            coor = system.coor()
            conn = system.conn()
            vector = system.vector()
            quad = system.quad()
            dV = quad.AsTensor(2, system.dV())
            is_p = vector.dofs_is_p()

            mesh = gf.Mesh.Quad4.FineLayer(coor, conn)
            mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)
            regular = mapping.getRegularMesh()
            assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

        break

    fine = mapping.getRegularMesh()
    elmat = fine.elementgrid()
    mid = int((N - N % 2) / 2)

    # Ensemble average

    with h5py.File(output, 'w') as out:

        # out['/intersect/nely'] = fine.nely()
        # out['/intersect/h'] = fine.h()
        # out['/mesh/h'] = mesh.h()
        # out['/mesh/nelx'] = mesh.nelx()
        # out['/mesh/nely'] = mesh.nely()
        # out['/mesh/N'] = N

        for ifile, file in enumerate(tqdm.tqdm(files)):

            with h5py.File(file, 'r') as data:

                idnum = data["/meta/id"][...]
                uuid = data["/meta/uuid"].asstr()[...]
                idname = "id={0:03d}.hdf5".format(idnum)
                system = LoadSystem(os.path.join(source_dir, idname), uuid)
                iiter = data["/sync-A/global/iiter"][...]

                if ifile == 0:
                    if "/meta/versions/CrackEvolution_raw_stress" not in data:
                        out["/meta/versions/CrackEvolution_raw_stress"] = data["/git/run"][...]
                    else:
                        out["/meta/versions/CrackEvolution_raw_stress"] = data["/meta/versions/CrackEvolution_raw_stress"][...]

                # ensemble average different "A"

                # if ifile == 0:
                #     m_A = np.linspace(0, N - N % 100, 15).astype(np.int64)
                #     m_sig_xx = [enstat.mean.StaticNd() for A in m_A]
                #     m_sig_xy = [enstat.mean.StaticNd() for A in m_A]
                #     m_sig_yy = [enstat.mean.StaticNd() for A in m_A]
                #     m_t = [enstat.mean.Scalar() for A in m_A]

                stored = data["/sync-A/stored"][...]
                system.setU(data["/sync-A/{0:d}/u".format(np.min(stored))][...])
                idx0 = system.plastic_CurrentIndex()[:, 0]

                for i, A in enumerate(tqdm.tqdm(m_A)):

                    if A not in stored:
                        continue

                    system.setU(data["/sync-A/{0:d}/u".format(A)][...])
                    Sig = np.average(system.Sig(), weights=dV, axis=1)
                    idx = system.plastic_CurrentIndex()[:, 0]
                    renum = renumber(np.argwhere(idx0 != idx).ravel(), N)
                    # get = elmat[:, renum].ravel()
                    # select = elmat[:, mid].ravel()

                    sig_xx = mapping.mapToRegular(Sig[:, 0, 0]).reshape(fine.nely(), fine.nelx())
                    sig_xy = mapping.mapToRegular(Sig[:, 0, 1]).reshape(fine.nely(), fine.nelx())
                    sig_yy = mapping.mapToRegular(Sig[:, 1, 1]).reshape(fine.nely(), fine.nelx())

                    assert sig_xy.shape == elmat.shape

                    out['sig_xx'] = sig_xx
                    out['sig_xy'] = sig_xy
                    out['sig_yy'] = sig_yy

                    return 0

#                     m_sig_xx[i].add_sample(sig_xx)
#                     m_sig_xy[i].add_sample(sig_xy)
#                     m_sig_yy[i].add_sample(sig_yy)

#                     m_t[i].add_sample(iiter[A])

#         # store

#         out['/stored'] = m_A

#         for i, A in enumerate(m_A):

#             out['/{0:d}/sig_xx'.format(A)] = m_sig_xx[i].mean()
#             out['/{0:d}/sig_xy'.format(A)] = m_sig_xx[i].mean()
#             out['/{0:d}/sig_yy'.format(A)] = m_sig_yy[i].mean()
#             out['/{0:d}/iiter'.format(A)] = m_t[i].mean()

#         try:
#             version = get_version(root='..', relative_to=__file__)
#         except:
#             version = None

#         if version:
#             out["/meta/versions/collect_stress.py"] = version

if __name__ == "__main__":

    main()
