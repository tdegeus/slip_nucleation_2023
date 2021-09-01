r"""
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
"""
import os

import docopt
import enstat.mean
import GooseFEM as gf
import h5py
import numpy as np
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

    with h5py.File(filename, "r") as data:

        assert uuid == data["/uuid"].asstr()[...]

        system = HybridSystem(
            data["coor"][...],
            data["conn"][...],
            data["dofs"][...],
            data["dofsP"][...],
            data["/elastic/elem"][...],
            data["/cusp/elem"][...],
        )

        system.setMassMatrix(data["/rho"][...])
        system.setDampingMatrix(data["/damping/alpha"][...])
        system.setElastic(data["/elastic/K"][...], data["/elastic/G"][...])
        system.setPlastic(
            data["/cusp/K"][...], data["/cusp/G"][...], data["/cusp/epsy"][...]
        )
        system.setDt(data["/run/dt"][...])

        return system


def main():

    args = docopt.docopt(__doc__)

    source = args["<files.yaml>"]
    key = list(filter(None, args["--key"].split("/")))
    files = shelephant.yaml.read_item(source, key)
    assert len(files) > 0
    info = args["--info"]
    output = args["--output"]
    source_dir = os.path.dirname(info)

    shelephant.path.check_allisfile(files + [info])
    shelephant.path.overwrite(output, args["--force"])

    # Read normalisation

    with h5py.File(info, "r") as data:
        sig0 = data["/normalisation/sig0"][...]

    # Define mapping (same for all input)

    for file in files:

        with h5py.File(file, "r") as data:

            idnum = data["/meta/id"][...]
            uuid = data["/meta/uuid"].asstr()[...]
            idname = f"id={idnum:03d}.hdf5"

            system = LoadSystem(os.path.join(source_dir, idname), uuid)
            plastic = system.plastic()
            N = plastic.size
            mid = int((N - N % 2) / 2)
            assert np.all(np.equal(plastic, data["/meta/plastic"][...]))

            coor = system.coor()
            conn = system.conn()
            quad = system.quad()
            dV = quad.AsTensor(2, system.dV())

            mesh = gf.Mesh.Quad4.FineLayer(coor, conn)
            mapping = gf.Mesh.Quad4.Map.FineLayer2Regular(mesh)
            assert np.all(np.equal(plastic, mesh.elementsMiddleLayer()))

        break

    fine = mapping.getRegularMesh()
    elmat = fine.elementgrid()
    coarse = gf.Mesh.Quad4.Regular(int(fine.nelx() / 6), int(fine.nely() / 3), fine.h())
    refine = gf.Mesh.Quad4.Map.RefineRegular(coarse, 6, 3)
    assert np.all(np.equal(fine.conn(), refine.getFineMesh().conn()))

    # Ensemble average

    with h5py.File(output, "w") as out:

        out["/coarse/nelx"] = coarse.nelx()
        out["/coarse/nely"] = coarse.nely()
        out["/coarse/Lx"] = coarse.nelx() * 6 * coarse.h()
        out["/coarse/Ly"] = coarse.nely() * 3 * coarse.h()
        out["/center/nely"] = fine.nely()
        out["/center/h"] = fine.h()
        out["/mesh/h"] = mesh.h()
        out["/mesh/nelx"] = mesh.nelx()
        out["/mesh/nely"] = mesh.nely()
        out["/mesh/N"] = N

        for ifile, file in enumerate(tqdm.tqdm(files)):

            with h5py.File(file, "r") as data:

                idnum = data["/meta/id"][...]
                uuid = data["/meta/uuid"].asstr()[...]
                idname = f"id={idnum:03d}.hdf5"
                system = LoadSystem(os.path.join(source_dir, idname), uuid)
                iiter = data["/sync-A/global/iiter"][...]

                ver = "/meta/versions/CrackEvolution_raw_stress"
                if ifile == 0:
                    if ver not in data:
                        out[ver] = data["/git/run"][...]
                    else:
                        out[ver] = data[ver][...]

                # ensemble average different "A"

                if ifile == 0:
                    m_A = np.linspace(0, N - N % 100, 15).astype(np.int64)
                    m_i_sig_xx = [enstat.mean.StaticNd() for A in m_A]
                    m_i_sig_xy = [enstat.mean.StaticNd() for A in m_A]
                    m_i_sig_yy = [enstat.mean.StaticNd() for A in m_A]
                    m_c_sig_xx = [enstat.mean.StaticNd() for A in m_A]
                    m_c_sig_xy = [enstat.mean.StaticNd() for A in m_A]
                    m_c_sig_yy = [enstat.mean.StaticNd() for A in m_A]
                    m_c_S = [enstat.mean.StaticNd() for A in m_A]
                    m_t = [enstat.mean.Scalar() for A in m_A]

                stored = data["/sync-A/stored"][...]
                system.setU(data[f"/sync-A/{np.min(stored):d}/u"][...])
                idx0 = system.plastic_CurrentIndex()[:, 0]

                for i, A in enumerate(tqdm.tqdm(m_A)):

                    if A not in stored:
                        continue

                    system.setU(data[f"/sync-A/{A:d}/u"][...])
                    Sig = np.average(system.Sig(), weights=dV, axis=1)
                    idx = system.plastic_CurrentIndex()[:, 0]
                    renum = renumber(np.argwhere(idx0 != idx).ravel(), N)
                    get = elmat[:, renum].ravel()
                    select = elmat[:, mid].ravel()

                    sig_xx = mapping.mapToRegular(Sig[:, 0, 0])[get] / sig0
                    sig_xy = mapping.mapToRegular(Sig[:, 0, 1])[get] / sig0
                    sig_yy = mapping.mapToRegular(Sig[:, 1, 1])[get] / sig0

                    m_c_sig_xx[i].add_sample(
                        refine.meanToCoarse(sig_xx).reshape(coarse.nely(), -1)
                    )
                    m_c_sig_xy[i].add_sample(
                        refine.meanToCoarse(sig_xy).reshape(coarse.nely(), -1)
                    )
                    m_c_sig_yy[i].add_sample(
                        refine.meanToCoarse(sig_yy).reshape(coarse.nely(), -1)
                    )

                    m_i_sig_xx[i].add_sample(sig_xx[select])
                    m_i_sig_xy[i].add_sample(sig_xy[select])
                    m_i_sig_yy[i].add_sample(sig_yy[select])

                    S = (idx.astype(np.int64) - idx0.astype(np.int64))[renum]
                    S = np.mean(S.reshape(-1, 6), axis=1)
                    m_c_S[i].add_sample(S)
                    m_t[i].add_sample(iiter[A])

        # store

        out["/stored"] = m_A

        for i, A in enumerate(m_A):

            out[f"/{A:d}/center/sig_xx"] = m_i_sig_xx[i].mean()
            out[f"/{A:d}/center/sig_xy"] = m_i_sig_xy[i].mean()
            out[f"/{A:d}/center/sig_yy"] = m_i_sig_yy[i].mean()
            out[f"/{A:d}/coarse/sig_xx"] = m_c_sig_xx[i].mean()
            out[f"/{A:d}/coarse/sig_xy"] = m_c_sig_xy[i].mean()
            out[f"/{A:d}/coarse/sig_yy"] = m_c_sig_yy[i].mean()
            out[f"/{A:d}/coarse/S"] = m_c_S[i].mean()
            out[f"/{A:d}/iiter"] = m_t[i].mean()

        try:
            version = get_version(root="..", relative_to=__file__)
        except:  # noqa: E722
            version = None

        if version:
            out["/meta/versions/collect_stress.py"] = version


if __name__ == "__main__":

    main()
