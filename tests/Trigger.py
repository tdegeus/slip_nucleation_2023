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
        my.System.generate(filename, N=N, test_mode=True, classic=False)
        my.System.cli_run(["--dev", filename])
        my.System.cli_ensembleinfo(["--dev", filename, "--output", infoname])

        ret, args = my.Trigger.cli_job_strain(
            ["--dev", "-f", infoname, "-p", 4, "--pushes-per-config", 1, "-o", dirname]
        )
        for key in ret:
            ret[key] = ret[key][:4]
        my.Trigger._write(ret, "mytrigger", **args)
        commands = ret["command"]
        commands = [c.split(" ") for c in commands]
        for i in range(len(commands)):
            commands[i][-1] = os.path.join(dirname, commands[i][-1])
        output = []
        output.append(my.Trigger.cli_run(["--dev"] + commands[0][1:]))
        output.append(my.Trigger.cli_run(["--dev"] + commands[1][1:]))
        output.append(my.Trigger.cli_run(["--dev"] + commands[2][1:]))
        output.append(my.Trigger.cli_run(["--dev"] + commands[3][1:]))

        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo] + output)

        ret, args = my.Trigger.cli_job_deltasigma(
            ["--dev", "-f", infoname, "-p", 4, "--pushes-per-config", 1, "-o", dirname]
        )
        for key in ret:
            ret[key] = ret[key][:4]
        my.Trigger._write(ret, "mytrigger", **args)
        commands = ret["command"]
        commands = [c.split(" ") for c in commands]
        for i in range(len(commands)):
            commands[i][-1] = os.path.join(dirname, commands[i][-1])
        output = []
        output.append(my.Trigger.cli_run(["--dev"] + commands[0][1:]))
        output.append(my.Trigger.cli_run(["--dev"] + commands[1][1:]))
        output.append(my.Trigger.cli_run(["--dev"] + commands[2][1:]))
        output.append(my.Trigger.cli_run(["--dev"] + commands[3][1:]))

        triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
        my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo] + output)

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
