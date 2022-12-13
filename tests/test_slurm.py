import os
import sys
import unittest
from functools import partialmethod

from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_front import slurm  # noqa: E402


class MyTests(unittest.TestCase):
    def test_exec_cmd(self):

        cmd = 'echo "hello world"'
        script = slurm.script_exec(cmd)
        self.assertEqual(script.split("\n")[-2], cmd)


if __name__ == "__main__":

    unittest.main()
