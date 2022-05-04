import os
import shutil
import sys
import unittest

import h5py

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_front import Flow  # noqa: E402

dirname = "mytest"


class MyTests(unittest.TestCase):
    """
    Tests
    """

    @classmethod
    def setUpClass(self):

        if os.path.isdir(dirname):
            shutil.rmtree(dirname)

        os.makedirs(dirname)

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_small(self):
        """
        *   Run a small simulation, read output.
        *   Branch of velocity-jump experiments and run one, and read output.
        """

        N = 9
        files = Flow.cli_generate(["-N", N, "--dev", dirname])

        with h5py.File(files[-1], "a") as file:
            file["/boundcheck"][...] = 193

        Flow.cli_run(["--develop", files[-1]])
        Flow.cli_ensembleinfo(["-o", os.path.join(dirname, "einfo.h5"), files[-1]])
        Flow.cli_paraview(["-o", os.path.join(dirname, "tmp"), files[-1]])

        branch = Flow.cli_branch_velocityjump(["--dev", "-o", dirname, "-i", "250000", files[-1]])
        Flow.cli_run(["--dev", branch[-1]])


if __name__ == "__main__":

    unittest.main()
