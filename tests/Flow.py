import os
import shutil
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    """
    Tests
    """

    def test_small(self):

        dirname = "mytest"

        if os.path.isdir(dirname):
            shutil.rmtree(dirname)

        os.makedirs(dirname)

        N = 9
        files = my.Flow.cli_generate(["-N", N, "--dev", dirname])
        my.Flow.run(files[-1], dev=True, maxinc=int(1e7))

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
