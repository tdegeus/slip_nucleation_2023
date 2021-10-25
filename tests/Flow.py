import os
import shutil
import sys
import unittest

import h5py

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    """
    Tests
    """

    def test_small(self):
        """
        *   Run a small simulation, read output.
        *   Branch of velocity-jump experiments and run one, and read output.
        """

        dirname = "mytest"

        if os.path.isdir(dirname):
            shutil.rmtree(dirname)

        os.makedirs(dirname)

        N = 9
        files = my.Flow.cli_generate(["-N", N, "--dev", dirname])

        with h5py.File(files[-1], "a") as file:
            file["/boundcheck"][...] = 170

        my.Flow.cli_run(["--develop", files[-1]])
        my.Flow.cli_ensembleinfo(["-o", os.path.join(dirname, "einfo.h5"), files[-1]])

        branch = my.Flow.cli_branch_velocityjump(["--develop", "-o", dirname, files[-1]])
        my.Flow.cli_run(["--develop", branch[-1]])
        my.Flow.cli_ensembleinfo_velocityjump(["-o", os.path.join(dirname, "vinfo.h5"), branch[-1]])

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()