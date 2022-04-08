import os
import shutil
import sys
import unittest

import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_front import QuasiStatic  # noqa: E402
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
        QuasiStatic.generate(filename, N=N, test_mode=True, dev=True)
        QuasiStatic.cli_run(["--develop", filename])
        QuasiStatic.cli_ensembleinfo([filename, "--output", infoname, "--dev"])

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_rerun(self):

        with h5py.File(infoname, "r") as file:
            S = file[f"/full/{idname}/S"][...]

        # Rerun increment

        name = os.path.join(dirname, "rerun.h5")
        i = np.argmax(S)
        EventMap.cli_run(["--dev", "-f", "-i", i, "-o", name, filename])

        with h5py.File(name, "r") as file:
            s = file["S"][...]
            self.assertEqual(S[i], np.sum(s))

        # function call without without check

        EventMap.cli_basic_output(
            ["--dev", "-f", "-o", os.path.join(dirname, "eventcollect.h5"), name]
        )


if __name__ == "__main__":

    unittest.main()
