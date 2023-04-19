import os
import pathlib
import shutil
import sys
import unittest
from functools import partialmethod

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = pathlib.Path(__file__).parent.joinpath("..").resolve()
if (root / "mycode_front" / "_version.py").exists():
    sys.path.insert(0, str(root))

from mycode_front import Flow  # noqa: E402
from mycode_front import QuasiStatic  # noqa: E402

dirname = pathlib.Path(__file__).parent / "output" / "Flow"


def setUpModule():
    """
    Create working directory.
    """
    os.makedirs(dirname, exist_ok=True)


def tearDownModule():
    """
    Remove working directory.
    """
    shutil.rmtree(dirname)


class TestPhysics(unittest.TestCase):
    """
    Tests
    """

    def test_fext(self):
        mesh = GooseFEM.Mesh.Quad4.Regular(5, 5)
        coor = mesh.coor()

        top = mesh.nodesTopEdge()
        bottom = mesh.nodesBottomEdge()
        left = mesh.nodesLeftOpenEdge()
        right = mesh.nodesRightOpenEdge()
        h = coor[top[1], 0] - coor[top[0], 0]

        dofs = mesh.dofs()
        dofs[right, :] = dofs[left, :]
        dofs = GooseFEM.Mesh.renumber(dofs)

        iip = np.concatenate((dofs[bottom, :].ravel(), dofs[top, :].ravel()))

        plastic = mesh.elementgrid()[2, :]
        elastic = np.setdiff1d(np.arange(mesh.nelem), plastic)

        system = QuasiStatic.model.System(
            coor=coor,
            conn=mesh.conn(),
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

        u = np.zeros_like(mesh.coor())

        for i in range(u.shape[0]):
            u[i, 0] += 0.1 * mesh.coor()[i, 1]

        system.u = u

        fext = system.fext[top, 0]
        fext[0] += fext[-1]
        fext = np.mean(fext[:-1]) / h

        dV = GooseFEM.AsTensor(2, system.dV, 2)
        Sig = np.average(system.Sig(), weights=dV, axis=(0, 1))

        self.assertAlmostEqual(fext, Sig[0, 1])
        self.assertAlmostEqual(2 * fext, GMat.Sigd(Sig))


class TestFlow(unittest.TestCase):
    """
    Tests
    """

    def test_small(self):
        """
        *   Run a small simulation, read output.
        *   Branch of velocity-jump experiments and run one, and read output.
        """

        N = 9
        files = Flow.cli_generate(["-N", N, "--dev", dirname, "--gammadot", "2e-9"])

        with h5py.File(files[-1], "a") as file:
            file["/param/cusp/epsy/nchunk"][...] = 200

        with h5py.File(files[-1], "a") as file:
            file["/Flow/boundcheck"][...] = 193

        Flow.cli_run(["--develop", files[-1]])
        Flow.cli_ensembleinfo(["-o", os.path.join(dirname, "einfo.h5"), files[-1]])
        Flow.cli_paraview(["-o", os.path.join(dirname, "tmp"), files[-1]])

        # branch = Flow.cli_branch_velocityjump(["--dev", "-o", dirname, "-i", "250000", files[-1]])
        # Flow.cli_run(["--dev", branch[-1]])


if __name__ == "__main__":
    unittest.main()
