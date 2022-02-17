import os
import shutil
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    """ """

    def test_small(self):
        """ """

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
        my.System.generate(filename, N=N, test_mode=True, classic=False, dev=True)
        my.System.cli_run(["--dev", filename])
        my.System.cli_ensembleinfo(["--dev", filename, "--output", infoname])

        n = 6
        ret = my.Trigger.cli_job_strain(
            ["--dev", "-f", infoname, "-p", 4, "--pushes-per-config", 1, "-o", dirname, "--nmax", n]
        )
        output = []
        for i in range(n):
            c = ret["command"][i].split(" ")[1:]
            c[-1] = os.path.join(dirname, c[-1])
            output.append(my.Trigger.cli_run(["--dev"] + c))

        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo] + output)

        n = 5
        ret = my.Trigger.cli_job_deltasigma(
            ["--dev", "-f", infoname, "-p", 4, "--pushes-per-config", 1, "-o", dirname, "--nmax", n]
        )
        for i in range(n):
            c = ret["command"][i].split(" ")[1:]
            c[-1] = os.path.join(dirname, c[-1])
            output.append(my.Trigger.cli_run(["--dev"] + c))

        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo] + output)

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
