import os
import shutil
import sys
import unittest

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


def generate():

    for file in [filename, infoname]:
        if os.path.isfile(file):
            os.remove(file)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    N = 9
    my.System.generate(filename, N=N, test_mode=True, classic=False, dev=True)
    my.System.cli_run(["--dev", filename])
    my.System.cli_ensembleinfo(["--dev", filename, "--output", infoname])


class MyTests(unittest.TestCase):
    """ """

    def test_strain(self):

        n = 6
        ret = my.Trigger.cli_job_strain(
            ["--dev", "-f", infoname, "-p", 4, "--pushes-per-config", 1, "-o", dirname, "--nmax", n]
        )

        output = []
        for i in range(n):
            c = ret["command"][i].split(" ")[1:]
            c[-1] = os.path.join(dirname, c[-1])
            output.append(my.Trigger.cli_run(["--dev"] + c))

        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo] + output)

    def test_deltasigma(self):

        n = 5
        ret = my.Trigger.cli_job_deltasigma(
            ["--dev", "-f", infoname, "-p", 4, "--pushes-per-config", 1, "-o", dirname, "--nmax", n]
        )

        output = []
        for i in range(n):
            c = ret["command"][i].split(" ")[1:]
            c[-1] = os.path.join(dirname, c[-1])
            output.append(my.Trigger.cli_run(["--dev"] + c))

        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo] + output)

        # check if restoring works

        clonename = os.path.join(dirname, "myclone.h5")
        with h5py.File(triggerinfo, "r") as file:
            my.Trigger.restore_from_ensembleinfo(file, 0, clonename, dev=True)

        my.Trigger.cli_run(["--dev", clonename])

        with h5py.File(output[0], "r") as source, h5py.File(clonename, "r") as dest:
            self.assertTrue(np.allclose(source["/disp/1"][...], dest["/disp/1"][...]))
            self.assertTrue(np.isclose(np.diff(source["/t"][...]), np.diff(dest["/t"][...])))

        # check that dynamics can be rerun on restored file

        tempname = os.path.join(dirname, "myrerun.h5")
        my.MeasureDynamics.cli_run(["--dev", "-f", "-i", 1, "-o", tempname, clonename])


if __name__ == "__main__":

    generate()
    unittest.main()
    shutil.rmtree(dirname)
