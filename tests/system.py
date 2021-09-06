import os
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(root))
import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_small(self):

        my.system.generate("mysim.h5", N=9)


if __name__ == "__main__":

    unittest.main()
