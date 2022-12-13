import os
import pathlib
import shutil
import sys
import unittest
from functools import partialmethod

import h5py
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = pathlib.Path(__file__).parent.joinpath("..").resolve()
if (root / "mycode_front" / "_version.py").exists():
    sys.path.insert(0, str(root))

from mycode_front import Flow  # noqa: E402

dirname = pathlib.Path(__file__).parent / "output" / "Flow"


def setUpModule():
    """
    Create working directory.
    """
    os.makedirs(dirname, exist_ok=True)


def tearDownModule():
    """
    Remove working directory.
    """
    shutil.rmtree(dirname)


class MyTests(unittest.TestCase):
    """
    Tests
    """

    def test_small(self):
        """
        *   Run a small simulation, read output.
        *   Branch of velocity-jump experiments and run one, and read output.
        """

        N = 9
        files = Flow.cli_generate(["-N", N, "--dev", dirname])

        with h5py.File(files[-1], "a") as file:
            file["/param/cusp/epsy/nchunk"][...] = 200

        with h5py.File(files[-1], "a") as file:
            file["/Flow/boundcheck"][...] = 193

        Flow.cli_run(["--develop", files[-1]])
        Flow.cli_ensembleinfo(["-o", os.path.join(dirname, "einfo.h5"), files[-1]])
        Flow.cli_paraview(["-o", os.path.join(dirname, "tmp"), files[-1]])

        branch = Flow.cli_branch_velocityjump(["--dev", "-o", dirname, "-i", "250000", files[-1]])
        Flow.cli_run(["--dev", branch[-1]])


if __name__ == "__main__":

    unittest.main()
