import os
import shutil
import sys
import unittest

import GooseFEM
import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_front as my  # noqa: E402

dirname = "mytest"
idname = "id=0.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")


class MyTests(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):

        for file in [filename, infoname]:
            if os.path.isfile(file):
                os.remove(file)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        N = 9
        my.System.generate(filename, N=N, test_mode=True, dev=True)
        my.System.cli_run(["--develop", filename])
        my.System.cli_ensembleinfo([filename, "--output", infoname, "--dev"])

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_elements_at_height(self):

        N = 3**3
        mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N)

        conn = mesh.conn()
        coor = mesh.coor()
        e = mesh.elementsMiddleLayer()

        assert np.all(np.equal(my.MeasureDynamics.elements_at_height(coor, conn, 0), e))
        assert np.all(np.equal(my.MeasureDynamics.elements_at_height(coor, conn, 1), e + N))
        assert np.all(np.equal(my.MeasureDynamics.elements_at_height(coor, conn, 2), e + N * 2))

    def test_partial_storage(self):

        with h5py.File(infoname, "r") as file:
            A = file[f"/full/{idname}/A"][...]
            N = file["/normalisation/N"][...]

        inc = np.argwhere(A == N).ravel()[-1]

        with h5py.File(filename, "r") as file:
            system = my.System.init(file)
            u = file[f"/disp/{inc:d}"][...]

        plastic = system.plastic()
        system.setU(u)
        Sig = system.Sig()
        Sig_p = Sig[plastic, ...]

        vector = system.vector()
        partial = my.tools.PartialDisplacement(
            conn=system.conn(),
            dofs=system.dofs(),
            element_list=plastic,
        )
        dofstore = partial.dof_is_stored()
        doflist = partial.dof_list()

        ustore = vector.AsDofs(u)[dofstore]
        udof = np.zeros(vector.shape_dofval())
        udof[doflist] = ustore
        system.setU(vector.AsNode(udof))

        self.assertFalse(np.allclose(Sig, system.Sig()))
        self.assertTrue(np.allclose(Sig_p, system.Sig()[plastic, ...]))

    def test_rerun(self):

        with h5py.File(infoname, "r") as file:
            A = file[f"/full/{idname}/A"][...]
            N = file["/normalisation/N"][...]

        inc = np.argwhere(A == N).ravel()[-1]

        outname = os.path.join(dirname, f"id=0_reruninc={inc:d}.h5")
        info = os.path.join(dirname, "MeasureDynamicsEnsembleInfo.h5")
        my.MeasureDynamics.cli_run(["--dev", "-f", "--inc", inc, "-o", outname, filename])
        my.MeasureDynamics.cli_ensembleinfo(
            ["--dev", "-f", "--source", os.path.abspath(dirname), "-o", info, outname]
        )

    def test_trigger_run(self):

        d = dirname
        commands = my.Trigger.cli_job_deltasigma(
            ["--dev", "-f", infoname, "-d", 0.12, "-p", 1, "-o", d, "--nmax", 10]
        )
        triggername = my.Trigger.cli_run(["--dev"] + commands[-1].split(" ")[1:])
        triggerpack = os.path.join(dirname, "TriggerEnsemblePack.h5")
        triggerinfo = os.path.join(dirname, "TriggerEnsembleInfo.h5")
        my.Trigger.cli_ensemblepack(["-f", "-o", triggerpack, triggername])
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo, triggerpack])

        with h5py.File(triggerinfo, "r") as file:
            A = file["A"][0]

        outname = os.path.join(dirname, "rerun_trigger.h5")
        my.MeasureDynamics.cli_run(["--dev", "-f", "--inc", 1, "-o", outname, triggername])

        with h5py.File(outname, "r") as file:
            a = file["/A"][-1]
            iiter = file["/stored"][-1]
            ustore = file[f"/u/{iiter:d}"][...]
            doflist = file["/doflist"][...]

        with h5py.File(triggername, "r") as file:
            system = my.System.init(file)
            u = file["/disp/1"][...]
            system.setU(u)
            Sig = system.plastic_Sig()
            Eps = system.plastic_Eps()
            Epsp = system.plastic_Epsp()

        vector = system.vector()
        udof = np.zeros(vector.shape_dofval())
        udof[doflist] = ustore
        system.setU(np.zeros_like(u))
        system.setU(vector.AsNode(udof))

        self.assertTrue(np.allclose(Sig, system.plastic_Sig()))
        self.assertTrue(np.allclose(Eps, system.plastic_Eps()))
        self.assertTrue(np.allclose(Epsp, system.plastic_Epsp()))

        self.assertEqual(A, a)


if __name__ == "__main__":

    unittest.main()
