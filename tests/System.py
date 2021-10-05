import os
import shutil
import sys
import unittest

import h5py
import numpy as np
import shelephant

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(root))
import mycode_front as my  # noqa: E402


def getfilebase(path):
    """
    Remove directory and extension.
    """
    return os.path.splitext(os.path.basename(path))[0]


class MyTests(unittest.TestCase):
    def test_generate(self):

        dirname = "mytest"

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        my.System.cli_generate([dirname])

        shutil.rmtree(dirname)

    def test_small(self):

        # Basic run / Get output

        historic = shelephant.yaml.read(
            os.path.join(os.path.dirname(__file__), "data_System_small.yaml")
        )

        dirname = "mytest"
        idname = "id=0.h5"
        filename = os.path.join(dirname, idname)
        infoname = os.path.join(dirname, "EnsembleInfo.h5")

        for file in [filename, infoname]:
            if os.path.isfile(file):
                os.remove(file)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        N = 9
        my.System.generate(filename, N=N, test_mode=True)
        my.System.cli_run(["--force", filename])
        my.System.cli_ensembleinfo([filename, "--output", infoname])

        with h5py.File(infoname, "r") as file:
            epsd = file[f"/full/{idname}/epsd"][...]
            sigd = file[f"/full/{idname}/sigd"][...]
            A = file[f"/full/{idname}/A"][...]
            sig0 = file["/normalisation/sig0"][...]

        self.assertTrue(np.allclose(epsd, historic["epsd"]))
        self.assertTrue(np.allclose(sigd, historic["sigd"]))

        # PinAndTrigger : full run + collection (try running only, not really test)

        iss = np.argwhere(A == N).ravel()
        sign = np.mean(sigd[iss - 1])
        sigc = np.mean(sigd[iss])

        sigtarget = 0.5 * (sigc + sign)
        pushincs = [iss[-30], iss[-20], iss[-15], iss[-10], iss[-4]]
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
                "--min-a",
                1,
            ]
            + pushnames[:2]
        )

        my.PinAndTrigger.cli_collect(
            [
                "--output",
                collectname2,
                "--min-a",
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

        # PinAndTrigger : extract dynamics (try running only, not really test)

        paths = my.PinAndTrigger.cli_getdynamics_sync_A_job(
            ["-c", collectname, "-i", infoname, "--group", 2, dirname]
        )

        for path in paths:
            dirname = os.path.dirname(path)
            filename = os.path.basename(path)
            pwd = os.getcwd()
            os.chdir(dirname)
            my.PinAndTrigger.cli_getdynamics_sync_A([filename])
            os.chdir(pwd)

        my.PinAndTrigger.cli_getdynamics_sync_A_combine(
            ["-f", "-o", os.path.join(dirname, "mydynamics.h5")]
            + [path.replace(".yaml", ".h5") for path in paths]
        )

        my.PinAndTrigger.cli_getdynamics_sync_A_average(
            [
                "-f",
                "-o",
                os.path.join(dirname, "myaverage.h5"),
                "-s",
                os.path.join(dirname, "myaverage.yaml"),
            ]
            + paths
        )

        shutil.rmtree(dirname)

    def PinAndTrigger_cli_job(self):
        """
        Tries running only, not really a test
        """
        return

        dirname = "mytest"
        infoname = os.path.join(dirname, "EnsembleInfo.h5")

        my.PinAndTrigger.cli_job(["-a", 4, infoname, "-o", dirname, "-n", int(1e9)])

        pwd = os.getcwd()
        os.chdir(dirname)
        with open("PinAndTrigger_1-of-1.slurm") as file:
            cmd = (
                file.read()
                .split("\n")[-3]
                .split("stdbuf -o0 -e0 PinAndTrigger ")[1]
                .split(" ")
            )
            my.PinAndTrigger.cli_main(cmd)
        os.chdir(pwd)

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
