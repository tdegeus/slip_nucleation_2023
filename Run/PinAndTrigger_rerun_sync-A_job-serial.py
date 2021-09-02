"""
Generate configuration file for ``PinAndTrigger_rerun_sync-A.py``.
All output is written to the current working directory.
All inputs should be absolute or relative to the current working directory.
"""
import argparse
import itertools
import os
import re
import textwrap

import GooseHDF5 as g5
import GooseSLURM
import h5py
import numpy as np
import shelephant

if __name__ == "__main__":

    basename = os.path.splitext(os.path.basename(__file__))[0]
    reldir = os.path.relpath(os.path.dirname(__file__), os.getcwd())

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFormatter, description=__doc__)

    parser.add_argument(
        "-c",
        "--collect",
        type=str,
        default="PinAndTrigger_collect.h5",
        help='Existing data, generated with "PinAndTrigger.py" (read-only)',
    )

    parser.add_argument(
        "-b",
        "--basename",
        type=str,
        default="PinAndTrigger_Rerun",
        help="Base-name for job-scripts",
    )

    parser.add_argument(
        "-i",
        "--info",
        type=str,
        default="EnsembleInfo.h5",
        help="EnsembleInfo to read normalisation (read-only)",
    )

    parser.add_argument(
        "-n",
        "--group",
        type=int,
        default=50,
        help="Number of runs to group in a single job.",
    )

    parser.add_argument(
        "--pyscript",
        type=str,
        default=os.path.normpath(os.path.join(reldir, "PinAndTrigger_rerun_sync-A.py")),
        help="Python-script to run simulations",
    )

    parser.add_argument(
        "--conda",
        type=str,
        default="code_velocity",
        help="Base-name of the Conda environment (appended with '_E5v4' and '_s6g1')",
    )

    args = parser.parse_args()
    assert os.path.isfile(os.path.realpath(args.collect))
    assert os.path.isfile(os.path.realpath(args.info))

    config = dict(
        collected=args.collect,
        info=args.info,
    )

    with h5py.File(args.collect, "r") as data:

        # list with realisations
        paths = list(g5.getpaths(data, root="data", max_depth=5))
        paths = np.array([path.split("data/")[1].split("/...")[0] for path in paths])

        # lists with stress/element/A of each realisation
        stress = [re.split(r"(stress\=[0-9A-z]*)", path)[1] for path in paths]
        element = [re.split(r"(element\=[0-9]*)", path)[1] for path in paths]
        a_target = [int(re.split(r"(A\=)([0-9]*)", path)[2]) for path in paths]
        a_real = [int(data[g5.join("/data", path, "A")][...]) for path in paths]
        stress = np.array(stress)
        element = np.array(element)
        a_target = np.array(a_target)
        a_real = np.array(a_real)

        # lists with possible stress/element/A identifiers (unique)
        Stress = np.unique(stress)
        Element = np.unique(element)
        A_target = np.unique(a_target)

        files = []
        for a, s in itertools.product(A_target, Stress):
            subset = paths[(a_real > a - 10) * (a_real < a + 10) * (stress == s)]
            files += list(subset)

    # write simulation files

    ngroup = int(np.ceil(len(files) / args.group))
    fmt = str(int(np.ceil(np.log10(ngroup))))

    for group in range(ngroup):

        ii = group * args.group
        jj = (group + 1) * args.group
        jobname = ("{0:s}-{1:0" + fmt + "d}").format(basename, group)

        config = dict(
            collected=args.collect,
            info=args.info,
            output=jobname + ".h5",
            paths=[str(i) for i in files[ii:jj]],
        )

        shelephant.yaml.dump(jobname + ".yaml", config, force=True)

        command = textwrap.dedent(
            f"""
        # print jobid
        echo "SLURM_JOBID = ${{SLURM_JOBID}}"
        echo ""

        # for safety set the number of cores
        export OMP_NUM_THREADS=1

        # load conda environment
        source ~/miniconda3/etc/profile.d/conda.sh

        if [[ "${{SYS_TYPE}}" == *E5v4* ]]; then
            conda activate {args.conda}_E5v4
        elif [[ "${{SYS_TYPE}}" == *s6g1* ]]; then
            conda activate {args.conda}_s6g1
        elif [[ "${{SYS_TYPE}}" == *S6g1* ]]; then
            conda activate {args.conda}_s6g1
        else
            echo "Unknown SYS_TYPE ${{SYS_TYPE}}"
            exit 1
        fi

        stdbuf -o0 -e0 python {args.pyscript} {jobname}.yaml
        """
        )

        sbatch = {
            "job-name": jobname,
            "out": jobname + ".out",
            "nodes": 1,
            "ntasks": 1,
            "cpus-per-task": 1,
            "time": "24h",
            "account": "pcsl",
            "partition": "serial",
        }

        open(jobname + ".slurm", "w").write(
            GooseSLURM.scripts.plain(command=command, **sbatch)
        )
