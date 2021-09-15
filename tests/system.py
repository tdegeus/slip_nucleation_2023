import os
import sys
import unittest

import h5py
import numpy as np
import shelephant

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(root))
import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_small(self):

        historic = shelephant.yaml.read(os.path.join(os.path.dirname(__file__), "system_small.yaml"))

        dirname = "mytest"
        filename = os.path.join(dirname, "id=0.h5")

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

        sigtarget = 0.5 * (sigc + sign)
        pushincs = [iss[-10], iss[-4]]
        fmt = os.path.join(dirname, "stress=mid_A=4_id=0_incc={0:d}_element=0.h5")
        pushnames = [fmt.format(i) for i in pushincs]
        collectname = os.path.join(dirname, "mypushes.h5")

        for pushname, incc in zip(pushnames, pushincs):

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

        if os.path.isfile(collectname):
            os.remove(collectname)

        my.PinAndTrigger.cli_collect(
            [
                "--output",
                collectname,
                "--min-A",
                1,
            ] + pushnames
        )

        # shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
