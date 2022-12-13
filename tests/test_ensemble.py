import os
import pathlib
import re
import shutil
import sys
import unittest
from functools import partialmethod

import enstat
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import GooseHDF5 as g5
import h5py
import numpy as np
import path
import shelephant
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = pathlib.Path(__file__).parent.joinpath("..").resolve()
if (root / "mycode_front" / "_version.py").exists():
    sys.path.insert(0, str(root))

from mycode_front import Dynamics  # noqa: E402
from mycode_front import EventMap  # noqa: E402
from mycode_front import QuasiStatic  # noqa: E402
from mycode_front import storage  # noqa: E402
from mycode_front import Trigger  # noqa: E402
from mycode_front import tools  # noqa: E402

basedir = pathlib.Path(__file__).parent / "output"
qsdir = basedir / "QuasiStatic"
triggerdir = basedir / "Trigger"
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

    def test_meta_move(self):
        """
        Move meta-data.
        """
        old = f"/meta/{QuasiStatic.entry_points['cli_run']}"
        new = f"{old}_new"
        QuasiStatic.cli_move_meta(["--dev", old, new, simpath])
        with h5py.File(simpath) as file:
            self.assertEqual(file[old].attrs["version"], file[new].attrs["version"])

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

    def test_rerun_dynamics_run_highfrequency(self):
        """
        Rerun a quasistatic step with Dynamics, check the macroscopic stress.
        """

        outdir = qsdir / "Dynamics"
        commands = QuasiStatic.cli_rerun_dynamics_job_systemspanning(["-f", "-o", outdir, infopath])
        _, _, outpath, _, step, inpath = commands[0].split(" ")
        step = int(step)
        # inpath = pathlib.Path(inpath).name

        with h5py.File(outdir / inpath) as file:
            system = QuasiStatic.System(file)
            system.restore_quasistatic_step(file["QuasiStatic"], step - 1)
            check = np.average(system.Sig(), weights=system.dV(rank=2), axis=(0, 1))[0, 1]

        with path.Path(outdir):

            Dynamics.cli_run_highfrequency(commands[0].split(" ")[1:] + ["--dev"])

            with h5py.File(outpath) as file:
                sig = file["/DynamicsHighFrequency/fext"][0]
                self.assertAlmostEqual(sig, check)

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


class test_Trigger(unittest.TestCase):
    """
    Test triggers deriving from ensemble.
    """

    @classmethod
    def setUpClass(self):
        """
        Branch and run.
        """

        n = 5
        self.opts = ["--dev", "-f", infopath, "-d", 0.12, "-p", 2, "-o", triggerdir, "--nmax", n]
        self.commands = Trigger.cli_job_deltasigma(self.opts)
        Trigger.cli_job_deltasigma(self.opts + ["--element", "2"])

        # Check that copying worked

        elem = [Trigger.interpret_filename(i)["element"] for i in self.commands]
        elem = np.unique(elem)
        o = [i.split(" ")[-1] for i in self.commands]
        uni = [re.sub(r"(.*)(element)(=[0-9]*)(.*)", r"\1\2=" + str(elem[0]) + r"\4", i) for i in o]
        uni = [str(i) for i in np.unique(uni)]

        for e in elem[1:]:
            for src in uni:
                dst = re.sub(r"(.*)(element)(=[0-9]*)(.*)", r"\1\2=" + str(e) + r"\4", src)
                res = g5.compare(src, dst)
                assert res["!="] == ["/Trigger/try_element"]
                assert res["->"] == []
                assert res["<-"] == []
                with h5py.File(src) as file:
                    assert elem[0] == file["/Trigger/try_element"][1]
                with h5py.File(dst) as file:
                    assert e == file["/Trigger/try_element"][1]
                    assert -1 == file["/Trigger/element"][1]

        # Run

        self.files = []
        for command in self.commands:
            self.files.append(Trigger.cli_run(command.split(" ")[1:]))

        # Collect

        self.pack = triggerdir / "ensemblepack.h5"
        self.info = triggerdir / "ensembleinfo.h5"

        Trigger.cli_ensemblepack(["-f", "-o", self.pack, *self.files])
        Trigger.cli_ensembleinfo(["--dev", "-f", "-o", self.info, self.pack])

    def test_ensemblepack(self):
        """
        Pack the ensemble. Try regenerating one realisation, and read the output.
        """

        t1 = self.pack
        t2 = triggerdir / "ensemblepack_t.h5"
        p1 = triggerdir / "ensemblepack_a.h5"
        p2 = triggerdir / "ensemblepack_b.h5"

        Trigger.cli_ensemblepack(["-f", "-o", p1, *self.files[:2]])
        Trigger.cli_ensemblepack(["-f", "-o", p2, *self.files[2:]])
        Trigger.cli_ensemblepack_merge(["-f", "-o", t2, p1, p2])

        with h5py.File(t1) as file, h5py.File(t2) as dest:
            res = g5.compare(file, dest)

        for key in res:
            if key != "==":
                self.assertEqual(res[key], [])

    def test_filter(self):
        """
        Pack the ensemble, check filtering new generation.
        """

        extra = Trigger.cli_job_deltasigma(self.opts + ["--filter", self.pack])
        self.assertFalse(np.any(np.in1d(extra, self.commands)))

    def test_ensembleinfo(self):
        """
        Pack the ensemble. Try regenerating one realisation, and read the output.
        """

        clonename = triggerdir / "myclone.h5"
        with h5py.File(self.info) as file:
            Trigger.restore_from_ensembleinfo(
                ensembleinfo=file, index=0, destpath=clonename, sourcedir=qsdir, dev=True
            )

        Trigger.cli_run(["--dev", "--rerun", clonename])

        with h5py.File(self.files[0]) as source, h5py.File(clonename) as dest:

            paths = list(g5.getdatapaths(source))

            for key in [
                "/meta/Trigger_JobDeltaSigma",
                "/meta/Trigger_Run",
                "/meta/branch_fixed_stress",
            ]:
                if key in paths:
                    paths.remove(key)

            self.assertTrue(g5.allequal(source, dest, paths))

    def test_rerun_dynamics_ensemble(self):
        """
        Rerun one trigger with Dynamics, check the macroscopic stress.
        """

        outdir = triggerdir / "Dynamics"
        commands = Trigger.cli_job_rerun_dynamics(
            [
                "--dev",
                "-f",
                "--test",
                "-n",
                "2",
                "--height",
                2,
                "-s",
                qsdir,
                "-o",
                outdir,
                self.info,
            ]
        )

        _, _, outpath, _, step, _, _, inpath = commands[0].split(" ")
        step = int(step)

        with h5py.File(inpath) as file:
            u = file["/Trigger/u/0"][...]

        with h5py.File(triggerdir / "deltasigma=0,00000_id=0_stepc=11_element=0.h5") as file:
            trigger = QuasiStatic.System(file)

            trigger.restore_quasistatic_step(file["Trigger"], 0)
            trigger_i_n = np.copy(trigger.plastic.i)
            if not np.allclose(u, trigger.u):
                print(u)
                print(trigger.u)
            self.assertTrue(np.allclose(u, trigger.u))

            trigger.restore_quasistatic_step(file["Trigger"], 1)

        with path.Path(outdir):

            Dynamics.cli_run(commands[0].split(" ")[1:] + ["--dev"])

            with h5py.File(outpath) as file:
                root = file["Dynamics"]
                dynamics = QuasiStatic.System(file)

                udof = np.zeros(dynamics.vector.shape_dofval())
                udof[root["doflist"][...]] = root["u"]["0"][...]
                dynamics.u = dynamics.vector.AsNode(udof)
                i_n = np.copy(dynamics.plastic.i)

                udof = np.zeros(dynamics.vector.shape_dofval())
                udof[root["doflist"][...]] = root["u"][str(file["/Dynamics/inc"].size - 1)][...]
                dynamics.u = dynamics.vector.AsNode(udof)

        self.assertTrue(np.all(np.equal(trigger_i_n, i_n)))
        self.assertTrue(np.all(np.equal(trigger.plastic.i, dynamics.plastic.i)))

    def test_rerun_dynamics(self):
        """
        Rerun one trigger with Dynamics, check the macroscopic stress.
        """

        outdir = triggerdir / "Dynamics2"
        commands = Trigger.cli_job_deltasigma(
            ["--dev", "-f", infopath, "-d", 0.12, "-p", 2, "-o", outdir, "--nmax", 10]
        )
        triggername = Trigger.cli_run(["--dev"] + commands[-1].split(" ")[1:])
        triggerpack = outdir / "tmp_Pack.h5"
        triggerinfo = outdir / "tmp_Info.h5"
        Trigger.cli_ensemblepack(["-f", "-o", triggerpack, triggername])
        Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo, triggerpack])

        with h5py.File(triggerinfo, "r") as file:
            A = file["A"][0]

        outname = outdir / "rerun_trigger.h5"
        Dynamics.cli_run(["--dev", "-f", "--step", 1, "-o", outname, triggername])

        with h5py.File(outname, "r") as file:
            a = file["/Dynamics/A"][-1]
            iiter = file["/Dynamics/inc"].size - 1
            ustore = file[f"/Dynamics/u/{iiter:d}"][...]
            doflist = file["/Dynamics/doflist"][...]

        with h5py.File(triggername, "r") as file:
            system = QuasiStatic.System(file)
            u = file["/Trigger/u/1"][...]
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

    def test_rerun_eventmap(self):
        """
        Rerun one trigger with EventMap, check the macroscopic stress.
        """

        outdir = triggerdir / "EventMap"
        commands = Trigger.cli_job_rerun_eventmap(
            ["--dev", "-f", "--amin", "2", "-n", "2", "-s", qsdir, "-o", outdir, self.info]
        )
        outpath = commands[0].split(" ")[2]

        EventMap.cli_run(["--dev", "-f"] + commands[0].split(" ")[1:])

        with h5py.File(triggerdir / "deltasigma=0,00000_id=0_stepc=11_element=5.h5") as file:
            system = QuasiStatic.System(file)
            system.restore_quasistatic_step(file["Trigger"], 0)
            i_n = np.copy(system.plastic.i).astype(np.int64)
            system.restore_quasistatic_step(file["Trigger"], 1)
            check = np.sum((system.plastic.i.astype(np.int64) - i_n)[:, 0])

        with h5py.File(outpath) as file:
            S = file["S"][...]

        self.assertEqual(np.sum(S), check)


class test_Dynamics(unittest.TestCase):
    def test_elements_at_height(self):
        """
        Identify elements at a certain height above the weak layer.
        """

        N = 3**3
        mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N)

        conn = mesh.conn()
        coor = mesh.coor()
        e = mesh.elementsMiddleLayer()

        assert np.all(np.equal(Dynamics.elements_at_height(coor, conn, 0), e))
        assert np.all(np.equal(Dynamics.elements_at_height(coor, conn, 1), e + N))
        assert np.all(np.equal(Dynamics.elements_at_height(coor, conn, 2), e + N * 2))

    def test_partial_storage(self):
        """
        Store the displacement filed only for a selection of elements.
        """

        with h5py.File(infopath, "r") as file:
            A = file[f"/full/{idname}/A"][...]
            N = file["/normalisation/N"][...]

        step = np.argwhere(A == N).ravel()[-1]

        with h5py.File(simpath, "r") as file:
            system = QuasiStatic.System(file)
            u = file[f"/QuasiStatic/u/{step:d}"][...]

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

        av = Dynamics.AlignedAverage(shape=[N + 1, N, 2, 2], elements=elem, dV=V)
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


if __name__ == "__main__":

    unittest.main(verbosity=2)
