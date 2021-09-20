import os
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(root))
import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_exec_cmd(self):

        cmd = 'echo "hello world"'
        script = my.slurm.script_exec(cmd)
        self.assertEqual(script.split("\n")[-1], "stdbuf -o0 -e0 " + cmd)

        _ = my.slurm.script_exec(cmd, conda=dict(condabase="my"))
        _ = my.slurm.script_exec(cmd, conda=("my", "/root"))
        _ = my.slurm.script_exec(cmd, conda=None)
        _ = my.slurm.script_exec(cmd, conda=False)


if __name__ == "__main__":

    unittest.main()
