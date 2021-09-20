import os
import sys
import unittest

import h5py
import numpy as np
import shelephant

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(root))
import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_small(self):

        # Basic run / Get output

        historic = shelephant.yaml.read(
            os.path.join(os.path.dirname(__file__), "system_small.yaml")
        )

        dirname = "mytest"
        idname = "id=0.h5"
        filename = os.path.join(dirname, idname)
        infoname = os.path.join(dirname, "EnsembleInfo.h5")

        if os.path.isfile(infoname):
            os.remove(infoname)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        N = 9
        my.System.generate(filename, N=N, test_mode=True)
        my.System.run(filename, dev=True)
        my.System.cli_ensembleinfo([filename, "--output", infoname])

        with h5py.File(infoname, "r") as file:
            Eps = file[f"/full/{idname}/epsd"][...]
            Sig = file[f"/full/{idname}/sigd"][...]
            A = file[f"/full/{idname}/A"][...]
            sig0 = file["/normalisation/sig0"][...]

        self.assertTrue(np.allclose(Eps, historic["Eps"]))
        self.assertTrue(np.allclose(Sig, historic["Sig"]))

        # PinAndTrigger : full run + collection (try running only, not real test)

        iss = np.argwhere(A == N).ravel()
        sign = np.mean(Sig[iss - 1])
        sigc = np.mean(Sig[iss])

        sigtarget = 0.5 * (sigc + sign)
        pushincs = [iss[-30], iss[-20], iss[-10], iss[-4]]
        fmt = os.path.join(dirname, "stress=mid_A=4_id=0_incc={0:d}_element=0.h5")
        pushnames = [fmt.format(i) for i in pushincs]

        for pushname, incc in zip(pushnames, pushincs):

            my.PinAndTrigger.cli_main(
                [
                    "--file",
                    filename,
                    "--output",
                    pushname,
                    "--stress",
                    sigtarget * sig0,
                    "--incc",
                    incc,
                    "--element",
                    0,
                    "--size",
                    4,
                ]
            )

        collectname1 = os.path.join(dirname, "mypushes_1.h5")
        collectname2 = os.path.join(dirname, "mypushes_2.h5")
        collectname = os.path.join(dirname, "mypushes.h5")

        for file in [collectname1, collectname2, collectname]:
            if os.path.isfile(file):
                os.remove(file)

        my.PinAndTrigger.cli_collect(
            [
                "--output",
                collectname1,
                "--min-A",
                1,
            ]
            + pushnames[:2]
        )

        my.PinAndTrigger.cli_collect(
            [
                "--output",
                collectname2,
                "--min-A",
                1,
            ]
            + pushnames[2:]
        )

        my.PinAndTrigger.cli_collect_combine(
            [
                "--output",
                collectname,
                collectname1,
                collectname2,
            ]
        )

        # PinAndTrigger : job-creation (try running only, not real test)

        my.PinAndTrigger.cli_job(["-A", 4, infoname, "-o", dirname, "-n", int(1e9)])

        pwd = os.getcwd()
        os.chdir(dirname)
        with open("PinAndTrigger_1-of-1.slurm") as file:
            cmd = (
                file.read()
                .split("\n")[-2]
                .split("stdbuf -o0 -e0 PinAndTrigger ")[1]
                .split(" ")
            )
            my.PinAndTrigger.cli_main(cmd)
        os.chdir(pwd)

        # shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
