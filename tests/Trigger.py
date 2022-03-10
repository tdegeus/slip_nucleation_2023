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
        my.System.generate(filename, N=N, test_mode=True, classic=False, dev=True)
        my.System.cli_run(["--dev", filename])
        my.System.cli_ensembleinfo(["--dev", filename, "--output", infoname])

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_strain(self):

        n = 3
        opts = ["--dev", "-f", infoname, "-n", 4, "-p", 2, "-o", dirname, "--nmax", n, "-r"]
        commands = my.Trigger.cli_job_strain(opts)

        output = []
        for command in commands:
            output.append(my.Trigger.cli_run(command.split(" ")[1:]))

        triggerpack = os.path.join(dirname, "TriggerPack.h5")
        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        my.Trigger.cli_ensemblepack(["-f", "-o", triggerpack] + output[:2])
        my.Trigger.cli_ensemblepack(["-a", "-o", triggerpack] + output[2:])
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo, triggerpack])

        extra = my.Trigger.cli_job_strain(opts + ["--filter", triggerpack])
        self.assertFalse(np.any(np.in1d(extra, commands)))

    def test_deltasigma(self):

        n = 5
        opts = ["--dev", "-f", infoname, "-d", 0.12, "-p", 2, "-o", dirname, "--nmax", n]
        commands = my.Trigger.cli_job_deltasigma(opts)

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
                self.assertEqual(res["!="], ["/trigger/element"])
                self.assertEqual(res["->"], [])
                self.assertEqual(res["<-"], [])
                with h5py.File(src, "r") as file:
                    self.assertEqual(elem[0], file["/trigger/element"][0])
                with h5py.File(dst, "r") as file:
                    self.assertEqual(e, file["/trigger/element"][0])

        # run ensemble

        output = []
        for command in commands:
            output.append(my.Trigger.cli_run(command.split(" ")[1:]))

        triggerpack = os.path.join(dirname, "TriggerPack.h5")
        triggerpack_a = os.path.join(dirname, "TriggerPack_a.h5")
        triggerpack_b = os.path.join(dirname, "TriggerPack_b.h5")
        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        my.Trigger.cli_ensemblepack(["-f", "-o", triggerpack_a] + output[:2])
        my.Trigger.cli_ensemblepack(["-f", "-o", triggerpack_b] + output[2:])
        my.Trigger.cli_ensemblepack_merge(["-f", "-o", triggerpack, triggerpack_a, triggerpack_b])
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo, triggerpack])

        # check partial re-rendering

        extra = my.Trigger.cli_job_deltasigma(opts + ["--filter", triggerpack])
        self.assertFalse(np.any(np.in1d(extra, commands)))

        # check if restoring works

        clonename = os.path.join(dirname, "myclone.h5")
        with h5py.File(triggerinfo, "r") as file:
            my.Trigger.restore_from_ensembleinfo(file, 0, clonename, dev=True)

        my.Trigger.cli_run(["--dev", clonename])

        with h5py.File(output[0], "r") as source, h5py.File(clonename, "r") as dest:
            self.assertTrue(np.allclose(source["/disp/1"][...], dest["/disp/1"][...]))
            self.assertTrue(np.isclose(np.diff(source["/t"][...]), np.diff(dest["/t"][...])))

        # create ensemble to rerun dynamics and try to rerun one

        outdir = os.path.join(dirname, "dynamics")
        cmd = my.Trigger.cli_job_rerun_dynamics(
            ["--dev", "-f", "-n", "2", "-s", dirname, "-o", outdir, triggerinfo]
        )
        cmd = cmd[0].split(" ")[1:]
        cmd[-1] = os.path.join(outdir, cmd[-1])
        cmd[-2] = os.path.join(outdir, cmd[-2])
        my.MeasureDynamics.cli_run(["--dev", "-f"] + cmd)

        # create ensemble to rerun EventMap and try to rerun one

        outdir = os.path.join(dirname, "eventmap")
        cmd = my.Trigger.cli_job_rerun_eventmap(
            ["--dev", "-f", "--amin", "2", "-n", "2", "-s", dirname, "-o", outdir, triggerinfo]
        )
        cmd = cmd[0].split(" ")[1:]
        cmd[-1] = os.path.join(outdir, cmd[-1])
        cmd[-2] = os.path.join(outdir, cmd[-2])
        my.EventMap.cli_run(["--dev", "-f"] + cmd)


if __name__ == "__main__":

    unittest.main()
