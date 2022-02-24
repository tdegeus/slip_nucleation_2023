import os
import shutil
import sys
import unittest

import h5py

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_front as my  # noqa: E402

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

    def test_branch(self):

        N = 9
        filename = "id=0.h5"
        my.System.generate(filename, N=N, test_mode=True, dev=True)
        my.System.cli_run(["--develop", filename])
        files = my.Flow.cli_generate(["--dev", filename])

        with h5py.File(files[-1], "a") as file:
            file["/boundcheck"][...] = 170

        my.Flow.cli_run(["--develop", files[-1]])

    def test_small(self):
        """
        *   Run a small simulation, read output.
        *   Branch of velocity-jump experiments and run one, and read output.
        """

        N = 9
        files = my.Flow.cli_generate(["-n", 1, "-N", N, "--dev", "-o", dirname])

        with h5py.File(files[-1], "a") as file:
            file["/boundcheck"][...] = 170

        my.Flow.cli_run(["--develop", files[-1]])
        # my.Flow.cli_ensembleinfo(["-o", os.path.join(dirname, "einfo.h5"), files[-1]])

        # branch = my.Flow.cli_branch_velocityjump(["--develop", "-o", dirname, files[-1]])
        # my.Flow.cli_run(["--develop", branch[-1]])
        # my.Flow.cli_ensembleinfo_velocityjump(["-o", os.path.join(dirname, "vinfo.h5"), branch[-1]])


if __name__ == "__main__":

    unittest.main()
