r'''
    Extract average force distribution. This involves:
    -   remapping to a regular mesh
    -   coarsening
        (reduces output size and speeds-up computation as the average on all A can be done at once).

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
from setuptools_scm import get_version

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
            dV = system.dV()
            is_p = vector.dofs_is_p()

            mesh = gf.Mesh.Quad4.FineLayer(coor, conn)
            mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)
            regular = mapping.getRegularMesh()
            assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

            # midpoint integration
            mid_quad = gf.Element.Quad4.Quadrature(
                vector.AsElement(coor),
                gf.Element.Quad4.MidPoint.xi(),
                gf.Element.Quad4.MidPoint.w())

            # nodal quadrature
            nodal_quad = gf.Element.Quad4.Quadrature(
                vector.AsElement(coor),
                gf.Element.Quad4.Nodal.xi(),
                gf.Element.Quad4.Nodal.w())
            dVs = nodal_quad.dV()

            # get nodal volume: per dimension, with periodicity applied
            dV_node = np.zeros(vector.ShapeNodevec())
            for j in range(conn.shape[1]):
                dV_node[conn[:, j], 0] += dVs[:, j]
                dV_node[conn[:, j], 1] += dVs[:, j]
            dV_node = vector.AsNode(vector.AssembleDofs(dV_node))

        break

    fine = mapping.getRegularMesh()
    elmat = fine.elementgrid()
    coarse = gf.Mesh.Quad4.Regular(int(fine.nelx() / 6), int(fine.nely() / 3), fine.h())
    refine = gf.Mesh.Quad4.Map.RefineRegular(coarse, 6, 3)
    assert np.all(np.equal(fine.conn(), refine.getFineMesh().conn()))

    # Ensemble average

    with h5py.File(output, 'w') as out:

        out['/coarse/nelx'] = coarse.nelx()
        out['/coarse/nely'] = coarse.nely()
        out['/coarse/Lx'] = coarse.nelx() * 6 * coarse.h()
        out['/coarse/Ly'] = coarse.nely() * 3 * coarse.h()
        out['/mesh/h'] = mesh.h()
        out['/mesh/nelx'] = mesh.nelx()
        out['/mesh/nely'] = mesh.nely()
        out['/mesh/N'] = N

        for ifile, file in enumerate(tqdm.tqdm(files)):

            with h5py.File(file, 'r') as data:

                idnum = data["/meta/id"][...]
                uuid = data["/meta/uuid"].asstr()[...]
                idname = "id={0:03d}.hdf5".format(idnum)
                system = LoadSystem(os.path.join(source_dir, idname), uuid)
                stored = data["/sync-A/stored"][...]
                iiter = data["/sync-A/global/iiter"][...]

                if ifile == 0:

                    m_A = np.linspace(300, 1400, 12).astype(np.int64) + 58
                    m_t = [enstat.mean.Scalar() for A in m_A]
                    m_fmaterial = [enstat.mean.StaticNd() for A in m_A]
                    m_fdamp = [enstat.mean.StaticNd() for A in m_A]
                    m_fres = [enstat.mean.StaticNd() for A in m_A]
                    m_Ekin = [enstat.mean.StaticNd() for A in m_A]
                    m_Epot = [enstat.mean.StaticNd() for A in m_A]
                    m_v = [enstat.mean.StaticNd() for A in m_A]
                    m_S = [enstat.mean.StaticNd() for A in m_A]

                system.setU(data["/sync-A/{0:d}/u".format(np.min(stored))][...])
                idx0 = system.plastic_CurrentIndex()[:, 0]

                for i, A in enumerate(tqdm.tqdm(m_A)):

                    if A not in stored:
                        continue

                    system.setU(data["/sync-A/{0:d}/u".format(A)][...])
                    system.setV(data["/sync-A/{0:d}/v".format(A)][...])
                    idx = system.plastic_CurrentIndex()[:, 0]

                    # nodal forces (apply reaction for to "fmaterial")
                    fmaterial = system.fmaterial()
                    fmaterial = np.where(is_p, 0, fmaterial)
                    fdamp = system.fdamp()
                    fres = -(fmaterial + fdamp)
                    V = vector.AsDofs(system.v())
                    K = M * V ** 2
                    Ekin = vector.AsNode(K)

                    # nodal force density
                    fmaterial /= dV_node
                    fdamp /= dV_node
                    fres /= dV_node
                    Ekin /= dV_node

                    # potential energy
                    Epot = np.average(system.Energy(), weights=dV, axis=1)

                    # convert to element-vector, interpolate to the element's midpoint, extrapolate on regular mesh
                    fmaterial = mapping.mapToRegular(mid_quad.Interp_N_vector(vector.AsElement(fmaterial)).reshape(-1, 2))
                    fdamp = mapping.mapToRegular(mid_quad.Interp_N_vector(vector.AsElement(fdamp)).reshape(-1, 2))
                    fres = mapping.mapToRegular(mid_quad.Interp_N_vector(vector.AsElement(fres)).reshape(-1, 2))
                    Ekin = mapping.mapToRegular(mid_quad.Interp_N_vector(vector.AsElement(Ekin)).reshape(-1, 2))
                    Epot = mapping.mapToRegular(Epot)
                    v = mapping.mapToRegular(mid_quad.Interp_N_vector(vector.AsElement(system.v())).reshape(-1, 2))

                    # element numbers such that the crack is aligned
                    renum = renumber(np.argwhere(idx0 != idx).ravel(), N)
                    get = elmat[:, renum].ravel()

                    # align crack
                    fmaterial = fmaterial[get]
                    fdamp = fdamp[get]
                    fres = fres[get]
                    Ekin = Ekin[get]
                    Epot = Epot[get]
                    v = v[get]
                    S = (idx.astype(np.int64) - idx0.astype(np.int64))[renum]

                    # coarsen by taking element average, take vector norm, reshape to grid
                    fmaterial = np.linalg.norm(refine.meanToCoarse(fmaterial), axis=1).reshape(coarse.nely(), -1)
                    fdamp = np.linalg.norm(refine.meanToCoarse(fdamp), axis=1).reshape(coarse.nely(), -1)
                    fres = np.linalg.norm(refine.meanToCoarse(fres), axis=1).reshape(coarse.nely(), -1)
                    Ekin = np.linalg.norm(refine.meanToCoarse(Ekin), axis=1).reshape(coarse.nely(), -1)
                    Epot = refine.meanToCoarse(Epot).reshape(coarse.nely(), -1)
                    v = np.linalg.norm(refine.meanToCoarse(v), axis=1).reshape(coarse.nely(), -1)
                    S = np.mean(S.reshape(-1, 6), axis=1)

                    m_fmaterial[i].add_sample(fmaterial)
                    m_fdamp[i].add_sample(fdamp)
                    m_fres[i].add_sample(fres)
                    m_Ekin[i].add_sample(Ekin)
                    m_Epot[i].add_sample(Epot)
                    m_v[i].add_sample(v)
                    m_S[i].add_sample(S)
                    m_t[i].add_sample(iiter[A])

        out['/stored'] = m_A

        for i, A in enumerate(m_A):

            out['/{0:d}/fmaterial'.format(A)] = m_fmaterial[i].mean()
            out['/{0:d}/fdamp'.format(A)] = m_fdamp[i].mean()
            out['/{0:d}/fres'.format(A)] = m_fres[i].mean()
            out['/{0:d}/Ekin'.format(A)] = m_Ekin[i].mean()
            out['/{0:d}/Epot'.format(A)] = m_Epot[i].mean()
            out['/{0:d}/v'.format(A)] = m_v[i].mean()
            out['/{0:d}/S'.format(A)] = m_S[i].mean()
            out['/{0:d}/iiter'.format(A)] = m_t[i].mean()

        if "/meta/versions/CrackEvolution_raw_stress" not in data and "/git/run" in data:
            out["/meta/versions/CrackEvolution_raw_stress"] = data["/git/run"][...]
        out["/meta/versions/collect_forces.py"] = get_version(root='..', relative_to=__file__)

if __name__ == "__main__":

    main()
