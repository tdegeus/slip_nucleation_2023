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
from FrictionQPotFEM.UniformSingleLayer2d import HybridSystem

# https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions


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
    shelephant.OverWrite(output)

    # Get constants

    with h5py.File(files[0], 'r') as data:
        plastic = data['/meta/plastic'][...]
        N = len(plastic)
        mid = (N - N % 2) / 2

    with h5py.File(info, 'r') as data:
        sig0 = data['/normalisation/sig0'][...]
        h = data['/normalisation/l0'][...]


    # Define mapping (same for all input)

    for file in files:

        with h5py.File(file, 'r') as data:

            idnum = data["/meta/id"][...]
            uuid = data["/meta/uuid"].asstr()[...]
            idname = "id={0:03d}.hdf5".format(idnum)

            system = LoadSystem(os.path.join(source_dir, idname), uuid)

            coor = system.coor()
            conn = system.conn()
            vector = system.vector()
            is_p = vector.dofs_is_p()

            mesh = gf.Mesh.Quad4.FineLayer(coor, conn)
            mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)
            regular = mapping.getRegularMesh()
            assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

            nodal_quad = gf.Element.Quad4.Quadrature(
                vector.AsElement(coor),
                gf.Element.Quad4.Nodal.xi(),
                gf.Element.Quad4.Nodal.w())
            dV = nodal_quad.dV()

            # get nodal volume: per dimension, with periodicity applied
            dV_node = np.zeros(vector.ShapeNodevec())
            for j in range(conn.shape[1]):
                dV_node[conn[:, j], 0] += dV[:, j]
                dV_node[conn[:, j], 1] += dV[:, j]
            dV_node = vector.AsNode(vector.AssembleDofs(dV_node))

        break

    fine = mapping.getRegularMesh()
    elmat = fine.elementgrid()
    coarse = gf.Mesh.Quad4.Regular(int(fine.nelx() / 6), int(fine.nely() / 3), fine.h())
    refine = gf.Mesh.Quad4.Map.RefineRegular(coarse, 6, 3)
    assert np.all(np.equal(fine.conn(), refine.getFineMesh().conn()))

    # Ensemble average

    with h5py.File(output, 'w') as out:

        out['/nelx'] = coarse.nelx()
        out['/nely'] = coarse.nely()
        out['/hx'] = coarse.nelx() * 6 * coarse.h()
        out['/hy'] = coarse.nely() * 3 * coarse.h()

        for ifile, file in enumerate(tqdm.tqdm(files)):

            with h5py.File(file, 'r') as data:

                idnum = data["/meta/id"][...]
                uuid = data["/meta/uuid"].asstr()[...]
                idname = "id={0:03d}.hdf5".format(idnum)
                system = LoadSystem(os.path.join(source_dir, idname), uuid)
                stored = data["/sync-A/stored"][...]

                if ifile == 0:

                    m_A = [A for A in stored]
                    m_fmaterial = [enstat.mean.Static() for A in stored]
                    m_fdamp = [enstat.mean.Static() for A in stored]
                    m_fres = [enstat.mean.Static() for A in stored]
                    m_v = [enstat.mean.Static() for A in stored]

                for i, A in enumerate(tqdm.tqdm(stored)):

                    if A not in m_A:
                        continue

                    system.setU(data["/sync-A/{0:d}/u".format(A)][...])
                    system.setV(data["/sync-A/{0:d}/v".format(A)][...])

                    if i == 0:
                        idx0 = system.plastic_CurrentIndex()[:, 0]

                    idx = system.plastic_CurrentIndex()[:, 0]

                    # nodal forces (apply reaction for to "fmaterial")
                    fmaterial = system.fmaterial()
                    fmaterial = np.where(is_p, 0, fmaterial)
                    fdamp = system.fdamp()
                    fres = -(fmaterial + fdamp)

                    # nodal force density
                    fmaterial /= dV_node
                    fdamp /= dV_node
                    fres /= dV_node

                    # convert to element-vector, extrapolate on regular mesh, and take element average
                    fmaterial = np.mean(mapping.mapToRegular(vector.AsElement(fmaterial)), axis=1)
                    fdamp = np.mean(mapping.mapToRegular(vector.AsElement(fdamp)), axis=1)
                    fres = np.mean(mapping.mapToRegular(vector.AsElement(fres)), axis=1)
                    v = np.mean(mapping.mapToRegular(vector.AsElement(system.v())), axis=1)

                    # element numbers such that the crack is aligned
                    renum = renumber(np.argwhere(idx0 != idx).ravel(), N)
                    get = elmat[:, renum].ravel()

                    fmaterial = fmaterial[get]
                    fdamp = fdamp[get]
                    fres = fres[get]
                    v = v[get]

                    def coarsen(refine, vec):
                        x = np.mean(refine.mapToCoarse(vec[:, 0]), axis=1)
                        y = np.mean(refine.mapToCoarse(vec[:, 1]), axis=1)
                        ret = np.empty((x.size, 2), dtype=np.float64)
                        ret[:, 0] = x
                        ret[:, 1] = y
                        return ret

                    # coarsen and take element average
                    fmaterial = np.linalg.norm(coarsen(refine, fmaterial), axis=1).reshape(coarse.nely(), -1)
                    fdamp = np.linalg.norm(coarsen(refine, fdamp), axis=1).reshape(coarse.nely(), -1)
                    fres = np.linalg.norm(coarsen(refine, fres), axis=1).reshape(coarse.nely(), -1)
                    v = np.linalg.norm(coarsen(refine, v), axis=1).reshape(coarse.nely(), -1)

                    m_fmaterial[i].add_sample(fmaterial)
                    m_fdamp[i].add_sample(fdamp)
                    m_fres[i].add_sample(fres)
                    m_v[i].add_sample(v)

        out['/stored'] = m_A

        for i, A in enumerate(m_A):

            out['/{0:d}/fmaterial'.format(A)] = m_fmaterial[i].mean()
            out['/{0:d}/fdamp'.format(A)] = m_fdamp[i].mean()
            out['/{0:d}/fres'.format(A)] = m_fres[i].mean()
            out['/{0:d}/v'.format(A)] = m_v[i].mean()


if __name__ == "__main__":

    main()
