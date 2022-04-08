import os
import shutil
import sys
import unittest

import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_front import QuasiStatic  # noqa: E402
from mycode_front import PinAndTrigger  # noqa: E402

dirname = "mytest"
idname = "id=0.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")


class MyTests(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):

        for file in [filename, infoname]:
            if os.path.isfile(file):
                os.remove(file)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        N = 9
        QuasiStatic.generate(filename, N=N, test_mode=True, dev=True)
        QuasiStatic.cli_run(["--develop", filename])
        QuasiStatic.cli_ensembleinfo([filename, "--output", infoname, "--dev"])

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_small(self):

        # Basic run / Get output

        with h5py.File(infoname, "r") as file:
            sigd = file[f"/full/{idname}/sigd"][...]
            A = file[f"/full/{idname}/A"][...]
            sig0 = file["/normalisation/sig0"][...]
            N = int(file["/normalisation/N"][...])

        # PinAndTrigger : full run + collection (try running only, not really test)

        iss = np.argwhere(A == N).ravel()
        sign = np.mean(sigd[iss - 1])
        sigc = np.mean(sigd[iss])

        sigtarget = 0.5 * (sigc + sign)
        pushincs = [iss[-30], iss[-20], iss[-15], iss[-10], iss[-4]]
        fmt = os.path.join(dirname, "stress=mid_A=4_id=0_incc={0:d}_element=0.h5")
        pushnames = [fmt.format(i) for i in pushincs]

        for pushname, incc in zip(pushnames, pushincs):

            PinAndTrigger.cli_run(
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

        PinAndTrigger.cli_collect(
            [
                "--output",
                collectname1,
                "--min-a",
                1,
            ]
            + pushnames[:2]
        )

        PinAndTrigger.cli_collect(
            [
                "--output",
                collectname2,
                "--min-a",
                1,
            ]
            + pushnames[2:]
        )

        PinAndTrigger.cli_collect_combine(
            [
                "--output",
                collectname,
                collectname1,
                collectname2,
            ]
        )

        # PinAndTrigger : interpret data (try running only, not really test)

        interpret = os.path.join(dirname, "myinterpret.h5")
        spatial = os.path.join(dirname, "myspatial.h5")

        PinAndTrigger.cli_output_scalar(["-f", "-i", infoname, "-o", interpret, collectname])

        PinAndTrigger.cli_output_spatial(["-f", "-i", infoname, "-o", spatial, collectname])

        # PinAndTrigger : extract dynamics (try running only, not really test)

        paths = PinAndTrigger.cli_getdynamics_sync_A_job(
            ["-c", collectname, "-i", infoname, "--group", 2, dirname]
        )

        for path in paths:
            d = os.path.dirname(path)
            f = os.path.basename(path)
            pwd = os.getcwd()
            os.chdir(d)
            PinAndTrigger.cli_getdynamics_sync_A([f])
            os.chdir(pwd)

        PinAndTrigger.cli_getdynamics_sync_A_combine(
            ["-f", "-o", os.path.join(dirname, "mydynamics.h5")]
            + [path.replace(".yaml", ".h5") for path in paths]
        )

        PinAndTrigger.cli_getdynamics_sync_A_average(
            [
                "-f",
                "-o",
                os.path.join(dirname, "myaverage.h5"),
                "-s",
                os.path.join(dirname, "myaverage.yaml"),
            ]
            + paths
        )

    def PinAndTrigger_cli_job(self):
        """
        Tries running only, not really a test
        """

        PinAndTrigger.cli_job(["-a", 4, infoname, "-o", dirname, "-n", int(1e9)])

        pwd = os.getcwd()
        os.chdir(dirname)
        with open("PinAndTrigger_1-of-1.slurm") as file:
            cmd = file.read().split("\n")[-3].split("stdbuf -o0 -e0 PinAndTrigger ")[1].split(" ")
            PinAndTrigger.cli_run(cmd)
        os.chdir(pwd)


if __name__ == "__main__":

    unittest.main()
