import os
import shutil
import sys
import unittest

import h5py
import numpy as np
import shelephant

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

    def test_generate(self):
        """
        Generate a simulation (no real test)
        """

        mygendir = os.path.join(dirname, "mygen")

        if not os.path.isdir(mygendir):
            os.makedirs(mygendir)

        my.System.cli_generate(["--dev", mygendir])

    def test_status(self):
        """
        Check that file was completed.
        """
        ret = my.System.cli_status(["-k", f"/meta/{my.System.entry_points['cli_run']}", filename])
        self.assertEqual(ret, {"completed": [filename], "new": [], "partial": []})

    def test_small(self):
        """
        Generate + run + check historic output
        """

        historic = shelephant.yaml.read(
            os.path.join(os.path.dirname(__file__), "data_System_small.yaml")
        )

        with h5py.File(infoname, "r") as file:
            epsd = file[f"/full/{idname}/epsd"][...]
            sigd = file[f"/full/{idname}/sigd"][...]
            incs = file[f"/full/{idname}/inc"][...]

        self.assertTrue(np.allclose(epsd[1:], historic["epsd"][3:]))
        self.assertTrue(np.allclose(sigd[1:], historic["sigd"][3:]))

        # function call without without check
        my.System.interface_state({filename: incs[-2:]})

    def test_rerun(self):

        with h5py.File(infoname, "r") as file:
            S = file[f"/full/{idname}/S"][...]

        # Rerun increment

        name = os.path.join(dirname, "rerun.h5")
        i = np.argmax(S)
        my.System.cli_rerun_event(["--dev", "-f", "-i", i, "-o", name, filename])

        with h5py.File(name, "r") as file:
            s = file["S"][...]
            self.assertEqual(S[i], np.sum(s))

        # function call without without check

        my.System.cli_rerun_event_collect(
            ["--dev", "-f", "-o", os.path.join(dirname, "eventcollect.h5"), name]
        )
        my.System.cli_rerun_event_job_systemspanning(
            ["-f", "-o", os.path.join(dirname, "eventmap"), infoname]
        )

        my.System.cli_rerun_dynamics_job_systemspanning(
            ["-f", "-o", os.path.join(dirname, "rerundynamics"), infoname]
        )


if __name__ == "__main__":

    unittest.main()
