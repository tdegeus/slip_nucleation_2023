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

        def run(classic):

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
            my.System.generate(filename, N=N, test_mode=True, classic=classic)
            my.System.cli_run(["--dev", filename])
            my.System.cli_ensembleinfo(["--dev", filename, "--output", infoname])

            ret = my.Trigger.cli_job_strain(
                ["-f", infoname, "-p", 4, "--pushes-per-config", 1, "-o", dirname]
            )
            commands = ret["commands"]
            output = []
            output.append(my.Trigger.cli_run(["--dev"] + commands[0].split(" ")[1:]))
            output.append(my.Trigger.cli_run(["--dev"] + commands[1].split(" ")[1:]))
            output.append(my.Trigger.cli_run(["--dev"] + commands[2].split(" ")[1:]))
            output.append(my.Trigger.cli_run(["--dev"] + commands[3].split(" ")[1:]))
            print("\n".join(commands[:4]))

            triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
            my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo] + output)

            ret = my.Trigger.cli_job_deltasigma(
                ["-f", infoname, "-p", 4, "--pushes-per-config", 1, "-o", dirname]
            )
            commands = ret["commands"]
            output = []
            output.append(my.Trigger.cli_run(["--dev"] + commands[0].split(" ")[1:]))
            output.append(my.Trigger.cli_run(["--dev"] + commands[1].split(" ")[1:]))
            output.append(my.Trigger.cli_run(["--dev"] + commands[2].split(" ")[1:]))
            output.append(my.Trigger.cli_run(["--dev"] + commands[3].split(" ")[1:]))
            print("\n".join(commands[:4]))

            triggerinfo = os.path.join(dirname, "TriggerInfo.h5")
            my.Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo] + output)

            shutil.rmtree(dirname)

        run(classic=False)


if __name__ == "__main__":

    unittest.main()
