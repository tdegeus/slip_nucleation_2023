import os
import sys
import unittest
import shutil

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(root))
import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):

    def test_small(self):

        dirname = "test"
        filename = os.path.join(dirname, "mysim.h5")

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        my.System.generate(filename, N=9)
        my.System.run(filename, True)

        # shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
