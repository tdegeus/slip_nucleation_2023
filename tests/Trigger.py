import os
import re
import shutil
import sys
import unittest

import GooseHDF5 as g5
import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_front import QuasiStatic  # noqa: E402
from mycode_front import Trigger  # noqa: E402
from mycode_front import MeasureDynamics  # noqa: E402
from mycode_front import EventMap  # noqa: E402

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
        QuasiStatic.generate(filename, N=N, test_mode=True, classic=False, dev=True)
        QuasiStatic.cli_run(["--dev", filename])
        QuasiStatic.cli_ensembleinfo(["--dev", filename, "--output", infoname])

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_deltasigma(self):

        n = 5
        opts = ["--dev", "-f", infoname, "-d", 0.12, "-p", 2, "-o", dirname, "--nmax", n]
        commands = Trigger.cli_job_deltasigma(opts)
        Trigger.cli_job_deltasigma(opts + ["--element", "2"])

        # check that copying worked

        elem = [int(i.split(" ")[-1].split("element=")[1].split("_")[0]) for i in commands]
        elem = np.unique(elem)
        o = [i.split(" ")[-1] for i in commands]
        uni = [re.sub(r"(.*)(element)(=[0-9]*)(.*)", r"\1\2=" + str(elem[0]) + r"\4", i) for i in o]
        uni = [str(i) for i in np.unique(uni)]

        for e in elem[1:]:
            for src in uni:
                dst = re.sub(r"(.*)(element)(=[0-9]*)(.*)", r"\1\2=" + str(e) + r"\4", src)
                res = g5.compare(src, dst)
                self.assertEqual(res["!="], ["/trigger/try_element"])
                self.assertEqual(res["->"], [])
                self.assertEqual(res["<-"], [])
                with h5py.File(src, "r") as file:
                    self.assertEqual(elem[0], file["/trigger/try_element"][1])
                with h5py.File(dst, "r") as file:
                    self.assertEqual(e, file["/trigger/try_element"][1])
                    self.assertEqual(-1, file["/trigger/element"][1])

        # run ensemble

        output = []
        for command in commands:
            output.append(Trigger.cli_run(command.split(" ")[1:]))

        triggerpack = os.path.join(dirname, "TriggerPack.h5")
        triggerpack_a = os.path.join(dirname, "TriggerPack_a.h5")
        triggerpack_b = os.path.join(dirname, "TriggerPack_b.h5")
        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        triggerinfo_b = os.path.join(dirname, "TriggerInfo_b.h5")
        Trigger.cli_ensemblepack(["-f", "-o", triggerpack_a] + output[:2])
        Trigger.cli_ensemblepack(["-f", "-o", triggerpack_b] + output[2:])
        Trigger.cli_ensemblepack_merge(["-f", "-o", triggerpack, triggerpack_a, triggerpack_b])
        Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo, triggerpack])
        Trigger.cli_ensembleinfo_merge(["--dev", "-o", triggerinfo_b, triggerinfo, triggerinfo])

        # check partial re-rendering

        extra = Trigger.cli_job_deltasigma(opts + ["--filter", triggerpack])
        self.assertFalse(np.any(np.in1d(extra, commands)))

        # check if restoring works

        clonename = os.path.join(dirname, "myclone.h5")
        with h5py.File(triggerinfo, "r") as file:
            Trigger.restore_from_ensembleinfo(file, 0, clonename, dev=True)

        Trigger.cli_run(["--dev", "--rerun", clonename])

        with h5py.File(output[0], "r") as source, h5py.File(clonename, "r") as dest:
            self.assertTrue(np.allclose(source["/disp/1"][...], dest["/disp/1"][...]))
            self.assertTrue(np.isclose(np.diff(source["/t"][...]), np.diff(dest["/t"][...])))

        # create ensemble to rerun dynamics and try to rerun one

        outdir = os.path.join(dirname, "dynamics")
        cmd = Trigger.cli_job_rerun_dynamics(
            [
                "--dev",
                "-f",
                "--test",
                "-n",
                "2",
                "--height",
                2,
                "-s",
                dirname,
                "-o",
                outdir,
                triggerinfo,
            ]
        )
        MeasureDynamics.cli_run(["--dev", "-f"] + cmd[0].split(" ")[1:])

        # create ensemble to rerun EventMap and try to rerun one

        outdir = os.path.join(dirname, "eventmap")
        cmd = Trigger.cli_job_rerun_eventmap(
            ["--dev", "-f", "--amin", "2", "-n", "2", "-s", dirname, "-o", outdir, triggerinfo]
        )
        EventMap.cli_run(["--dev", "-f"] + cmd[0].split(" ")[1:])


if __name__ == "__main__":

    unittest.main()
