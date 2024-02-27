import os

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np

from mycode_front import Flow
from mycode_front import QuasiStatic


def test_fext():
    mesh = GooseFEM.Mesh.Quad4.Regular(5, 5)
    coor = mesh.coor

    top = mesh.nodesTopEdge
    bottom = mesh.nodesBottomEdge
    left = mesh.nodesLeftOpenEdge
    right = mesh.nodesRightOpenEdge
    h = coor[top[1], 0] - coor[top[0], 0]

    dofs = mesh.dofs
    dofs[right, :] = dofs[left, :]
    dofs = GooseFEM.Mesh.renumber(dofs)

    iip = np.concatenate((dofs[bottom, :].ravel(), dofs[top, :].ravel()))

    plastic = mesh.elementgrid[2, :]
    elastic = np.setdiff1d(np.arange(mesh.nelem), plastic)

    system = QuasiStatic.model.System(
        coor=coor,
        conn=mesh.conn,
        dofs=dofs,
        iip=iip,
        elastic_elem=elastic,
        elastic_K=np.ones((elastic.size, 4)),
        elastic_G=np.ones((elastic.size, 4)),
        plastic_elem=plastic,
        plastic_K=np.ones((plastic.size, 4)),
        plastic_G=np.ones((plastic.size, 4)),
        plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(100 * np.ones((plastic.size, 1))),
        dt=0.1,
        rho=1,
        alpha=0,
        eta=0.1,
    )

    u = np.zeros_like(mesh.coor)

    for i in range(u.shape[0]):
        u[i, 0] += 0.1 * mesh.coor[i, 1]

    system.u = u

    fext = system.fext[top, 0]
    fext[0] += fext[-1]
    fext = np.mean(fext[:-1]) / h

    dV = GooseFEM.AsTensor(2, system.dV, 2)
    Sig = np.average(system.Sig(), weights=dV, axis=(0, 1))

    assert np.isclose(fext, Sig[0, 1])
    assert np.isclose(2 * fext, GMat.Sigd(Sig))


def test_small(tmp_path):
    """
    *   Run a small simulation, read output.
    *   Branch of velocity-jump experiments and run one, and read output.
    """
    N = 9
    files = Flow.cli_generate(["-N", N, "--dev", tmp_path, "--gammadot", "2e-9"])

    with h5py.File(files[-1], "a") as file:
        file["/param/cusp/epsy/nchunk"][...] = 200

    with h5py.File(files[-1], "a") as file:
        file["/Flow/boundcheck"][...] = 193

    Flow.cli_run(["--dev", files[-1]])
    Flow.cli_ensembleinfo(["-o", os.path.join(tmp_path, "einfo.h5"), files[-1]])
    Flow.cli_paraview(["-o", os.path.join(tmp_path, "tmp"), files[-1]])

    # branch = Flow.cli_branch_velocityjump(["--dev", "-o", tmp_path, "-i", "250000", files[-1]])
    # Flow.cli_run(["--dev", branch[-1]])
