import os
import pathlib
import shutil
import sys
import unittest

import GMatElastoPlasticQPot.Cartesian2d as GMat
import h5py
import numpy as np
import path
import shelephant

root = pathlib.Path(__file__).parent.joinpath("..").resolve()
if (root / "mycode_front" / "_version.py").exists():
    sys.path.insert(0, root)

from mycode_front import Dynamics  # noqa: E402
from mycode_front import EventMap  # noqa: E402
from mycode_front import QuasiStatic  # noqa: E402
from mycode_front import storage  # noqa: E402

basedir = pathlib.Path(__file__).parent / "output"
qsdir = basedir / "QuasiStatic"
idname = "id=0.h5"
simpath = qsdir / idname
infopath = qsdir / "EnsembleInfo.h5"


def setUpModule():
    """
    Generate and run on simulation.
    """

    os.makedirs(qsdir, exist_ok=True)

    for file in [simpath, infopath]:
        if os.path.isfile(file):
            os.remove(file)

    QuasiStatic.generate(simpath, N=9, dev=True)

    with h5py.File(simpath, "a") as file:
        file["/param/cusp/epsy/nchunk"][...] = 200

    QuasiStatic.cli_run(["--develop", simpath])
    QuasiStatic.cli_ensembleinfo([simpath, "--output", infopath, "--dev"])


def tearDownModule():
    """
    Remove working directory.
    """

    shutil.rmtree(basedir)


class test_QuasiStatic(unittest.TestCase):
    """
    Test the QuasiStatic module, and Dynamics/EventMap output deriving from it.
    """

    def test_historic(self):
        """
        Generate + run + check historic output
        """

        historic = shelephant.yaml.read(pathlib.Path(__file__).parent / "data_System_small.yaml")

        with h5py.File(infopath, "r") as file:
            epsd = file[f"/full/{idname}/eps"][...]
            sigd = file[f"/full/{idname}/sig"][...]

        ref_eps = historic["epsd"][3:]
        ref_sig = historic["sigd"][3:]

        self.assertTrue(np.allclose(epsd[1:][: len(ref_eps)], ref_eps))
        self.assertTrue(np.allclose(sigd[1:][: len(ref_sig)], ref_sig))

    def test_generate(self):
        """
        Generate a simulation (no real test)
        """

        mygendir = qsdir / "mygen"

        if mygendir.is_dir():
            shutil.rmtree(mygendir)

        mygendir.mkdir()

        QuasiStatic.cli_generate(["--dev", mygendir])

    def test_status(self):
        """
        Check that file was completed.
        """
        ret = QuasiStatic.cli_status(
            ["-k", f"/meta/{QuasiStatic.entry_points['cli_run']}", simpath]
        )
        self.assertEqual(ret, {"completed": [str(simpath)], "new": [], "partial": []})

    def test_interface_state(self):
        """
        Check interface state (no real test)
        """

        with h5py.File(infopath, "r") as file:
            steps = file[f"/full/{idname}/step"][...]

        QuasiStatic.interface_state({simpath: steps[-2:]})

    def test_branch_fixed_stress(self):
        """
        Branch fixed stress.
        """

        with h5py.File(infopath, "r") as file:
            A = file[f"/full/{idname}/A"][...]
            N = file["/normalisation/N"][...]
            sig_bot = file["/averages/sig_bottom"][...]
            sig_top = file["/averages/sig_top"][...]

        step_c = np.argwhere(A == N).ravel()[-2]
        stress = (sig_bot + sig_top) / 2
        branchpath = qsdir / "branch_fixed_stress.h5"

        with h5py.File(simpath) as src, h5py.File(branchpath, "w") as dest:
            QuasiStatic.branch_fixed_stress(
                src, dest, "foo", step_c=step_c, stress=stress, normalised=True, dev=True
            )
            system = QuasiStatic.System(dest)
            out = QuasiStatic.basic_output(system, dest, "foo")
            self.assertAlmostEqual(out["sig"][0], stress)

    def test_rerun_dynamics(self):
        """
        Rerun a quasistatic step with Dynamics, check the macroscopic stress.
        """

        outdir = qsdir / "Dynamics"
        commands = QuasiStatic.cli_rerun_dynamics_job_systemspanning(["-f", "-o", outdir, infopath])
        _, _, outpath, _, step, inpath = commands[0].split(" ")
        step = int(step)
        inpath = pathlib.Path(inpath).name

        with h5py.File(infopath) as file:
            sig0 = file["/normalisation/sig0"][...]
            check = file[f"/full/{inpath}/sig"][step]

        with path.Path(outdir):

            Dynamics.cli_run(commands[0].split(" ")[1:] + ["--dev"])

            with h5py.File(outpath) as file:
                sig = GMat.Sigd(storage.symtens2_read(file, "/Dynamics/Sigbar")[-1, ...])
                self.assertAlmostEqual(sig / sig0, check)

            Dynamics.cli_average_systemspanning([outpath, "-o", outdir / "average.h5", "--dev"])

    def test_rerun_eventmap(self):
        """
        Rerun a quasistatic step with EventMap, check the event size
        """

        outdir = qsdir / "EventMap"
        commands = QuasiStatic.cli_rerun_dynamics_job_systemspanning(["-f", "-o", outdir, infopath])
        _, _, outpath, _, step, inpath = commands[0].split(" ")
        step = int(step)
        inpath = pathlib.Path(inpath).name

        with h5py.File(infopath) as file:
            check = file[f"/full/{inpath}/S"][step]

        with path.Path(outdir):

            EventMap.cli_run(commands[0].split(" ")[1:] + ["--dev"])

            with h5py.File(outpath) as file:
                S = file["S"][...]
                self.assertEqual(np.sum(S), check)

    def test_state_after_systemspanning(self):
        """
        Read the state of systemspanning events (no test!)
        """

        QuasiStatic.cli_state_after_systemspanning(["-f", "-o", qsdir / "state", infopath])


if __name__ == "__main__":

    unittest.main(verbosity=2)
