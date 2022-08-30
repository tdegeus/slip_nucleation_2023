import os
import shutil
import sys
import unittest

import enstat
import GooseFEM
import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_front import QuasiStatic  # noqa: E402
from mycode_front import MeasureDynamics  # noqa: E402
from mycode_front import Trigger  # noqa: E402
from mycode_front import tools  # noqa: E402

dirname = "mytest"
idname = "id=0.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")


class MyTests(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        """
        Generate one realisation.
        """

        for file in [filename, infoname]:
            if os.path.isfile(file):
                os.remove(file)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        N = 9
        QuasiStatic.generate(filename, N=N, test_mode=True, dev=True)
        QuasiStatic.cli_run(["--develop", filename])
        QuasiStatic.cli_ensembleinfo([filename, "--output", infoname, "--dev"])

    @classmethod
    def tearDownClass(self):
        """
        Remove all generated data.
        """

        shutil.rmtree(dirname)

    def test_elements_at_height(self):
        """
        Identify elements at a certain height above the weak layer.
        """

        N = 3**3
        mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N)

        conn = mesh.conn()
        coor = mesh.coor()
        e = mesh.elementsMiddleLayer()

        assert np.all(np.equal(MeasureDynamics.elements_at_height(coor, conn, 0), e))
        assert np.all(np.equal(MeasureDynamics.elements_at_height(coor, conn, 1), e + N))
        assert np.all(np.equal(MeasureDynamics.elements_at_height(coor, conn, 2), e + N * 2))

    def test_partial_storage(self):
        """
        Store the displacement filed only for a selection of elements.
        """

        with h5py.File(infoname, "r") as file:
            A = file[f"/full/{idname}/A"][...]
            N = file["/normalisation/N"][...]

        inc = np.argwhere(A == N).ravel()[-1]

        with h5py.File(filename, "r") as file:
            system = QuasiStatic.System(file)
            u = file[f"/disp/{inc:d}"][...]

        plastic = system.plastic_elem
        system.u = u
        Sig = system.Sig() / system.sig0
        Sig_p = Sig[plastic, ...]

        vector = system.vector
        partial = tools.PartialDisplacement(
            conn=system.conn,
            dofs=system.dofs,
            element_list=plastic,
        )
        dofstore = partial.dof_is_stored()
        doflist = partial.dof_list()

        ustore = vector.AsDofs(u)[dofstore]
        udof = np.zeros(vector.shape_dofval())
        udof[doflist] = ustore
        system.u = vector.AsNode(udof)

        self.assertFalse(np.allclose(Sig, system.Sig() / system.sig0))
        self.assertTrue(np.allclose(Sig_p, system.Sig()[plastic, ...] / system.sig0))

    def test_AlignedAverage(self):

        N = 10
        nip = 4
        elem = np.arange(N)
        nitem = 10
        V = np.random.random((N, nip, 2, 2))
        D = np.random.random((nitem, N + 1, N, nip, 2, 2))
        M = np.random.random((nitem, N + 1, N)) < 0.5

        av = MeasureDynamics.AlignedAverage(shape=[N + 1, N, 2, 2], elements=elem, dV=V)
        check_00 = [enstat.static(shape=[N]) for i in range(N + 1)]
        check_01 = [enstat.static(shape=[N]) for i in range(N + 1)]
        check_10 = [enstat.static(shape=[N]) for i in range(N + 1)]
        check_11 = [enstat.static(shape=[N]) for i in range(N + 1)]

        for i in range(nitem):
            for a in range(N + 1):
                av.add_subsample(i, D[i, a, ...], roll=0, broken=~M[i, a, ...])
                d = np.average(D[i, a, ...], weights=V, axis=1)
                check_00[i].add_sample(d[..., 0, 0], mask=M[i, a, ...])
                check_01[i].add_sample(d[..., 0, 1], mask=M[i, a, ...])
                check_10[i].add_sample(d[..., 1, 0], mask=M[i, a, ...])
                check_11[i].add_sample(d[..., 1, 1], mask=M[i, a, ...])

        res = np.empty([N + 1, N, 2, 2])
        for a in range(N + 1):
            res[a, :, 0, 0] = check_00[a].mean()
            res[a, :, 0, 1] = check_01[a].mean()
            res[a, :, 1, 0] = check_10[a].mean()
            res[a, :, 1, 1] = check_11[a].mean()

        self.assertTrue(np.allclose(av.mean(), res, equal_nan=True))

    def test_rerun(self):

        with h5py.File(infoname, "r") as file:
            A = file[f"/full/{idname}/A"][...]
            N = file["/normalisation/N"][...]

        inc = np.argwhere(A == N).ravel()[-1]

        outname = os.path.join(dirname, f"id=0_reruninc={inc:d}.h5")
        average = os.path.join(dirname, "MeasureDynamics_Average.h5")
        MeasureDynamics.cli_run(
            ["--dev", "-f", "--height", 2, "--inc", inc, "-o", outname, filename]
        )
        MeasureDynamics.cli_average_systemspanning(["--dev", "-f", "-o", average, outname])

    def test_trigger_run(self):

        commands = Trigger.cli_job_deltasigma(
            ["--dev", "-f", infoname, "-d", 0.12, "-p", 1, "-o", dirname, "--nmax", 10]
        )
        triggername = Trigger.cli_run(["--dev"] + commands[-1].split(" ")[1:])
        triggerpack = os.path.join(dirname, "TriggerEnsemblePack.h5")
        triggerinfo = os.path.join(dirname, "TriggerEnsembleInfo.h5")
        Trigger.cli_ensemblepack(["-f", "-o", triggerpack, triggername])
        Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo, triggerpack])

        with h5py.File(triggerinfo, "r") as file:
            A = file["A"][0]

        outname = os.path.join(dirname, "rerun_trigger.h5")
        MeasureDynamics.cli_run(["--dev", "-f", "--inc", 1, "-o", outname, triggername])

        with h5py.File(outname, "r") as file:
            a = file["/dynamics/A"][-1]
            iiter = file["/dynamics/stored"][-1]
            ustore = file[f"/dynamics/u/{iiter:d}"][...]
            doflist = file["/dynamics/doflist"][...]

        with h5py.File(triggername, "r") as file:
            system = QuasiStatic.System(file)
            u = file["/disp/1"][...]
            system.u = u
            Sig = system.plastic.Sig / system.sig0
            Eps = system.plastic.Eps / system.eps0
            Epsp = system.plastic.epsp / system.eps0

        udof = np.zeros(system.vector.shape_dofval())
        udof[doflist] = ustore
        system.u = system.vector.AsNode(udof)

        self.assertTrue(np.allclose(Sig, system.plastic.Sig / system.sig0))
        self.assertTrue(np.allclose(Eps, system.plastic.Eps / system.eps0))
        self.assertTrue(np.allclose(Epsp, system.plastic.epsp / system.eps0))

        self.assertEqual(A, a)


if __name__ == "__main__":

    unittest.main()
