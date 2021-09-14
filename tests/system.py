import os
import shutil
import sys
import unittest

import h5py
import numpy as np
import yaml

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(root))
import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_small(self):

        with open(os.path.join(os.path.dirname(__file__), "system_small.yaml")) as file:
            historic = yaml.load(file.read(), Loader=yaml.FullLoader)

        dirname = "mytest"
        filename = os.path.join(dirname, "mysim.h5")

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        N = 9
        my.System.generate(filename, N=N, test_mode=True)
        my.System.run(filename, dev=True)

        with h5py.File(filename, "r") as file:
            system = my.System.init(file)
            data = my.System.basic_output(system, file)

        self.assertTrue(np.allclose(data["Eps"], historic["Eps"]))
        self.assertTrue(np.allclose(data["Sig"], historic["Sig"]))

        iss = np.argwhere(data["A"] == N).ravel()
        sign = np.mean(data["Sig"][iss - 1])
        sigc = np.mean(data["Sig"][iss])

        incc = iss[-2]
        sigtarget = 0.5 * (sigc + sign)

        pushname = os.path.join(dirname, "mypush.h5")

        my.PinAndTrigger.cli_main(
            [
                "--file",
                filename,
                "--output",
                pushname,
                "--stress",
                sigtarget * data["sig0"],
                "--incc",
                incc,
                "--element",
                0,
                "--size",
                4,
            ]
        )

        # shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
