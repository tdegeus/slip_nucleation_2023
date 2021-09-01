r"""
Run for ``A = 1200`` by pushing two different elements
(and accordingly pinning different parts of the system).
Note that this way two mostly independent ensembles are created.

Afterwards, ``PinAndTrigger_job-serial-compact.py`` can be used to run for ``A < 1200``.
That function skips all events that are know to be too small, and therefore less time is waisted
on computing small events.
"""
import argparse
import itertools
import os

import GooseHDF5 as g5
import GooseSLURM
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("info", type=str, help="EnsembleInfo (read-only)")
parser.add_argument("-n", "--group", type=int, default=100)
parser.add_argument("-e", "--executable", type=str, default="python PinAndTrigger.py")
parser.add_argument(
    "-c", "--collection", type=str, help="Result of PinAndTrigger_collect.py"
)
args = parser.parse_args()
assert os.path.isfile(os.path.realpath(args.info))

executable = args.executable

paths = []

if args.collection:
    with h5py.File(args.collection, "r") as data:
        paths = list(g5.getpaths(data, max_depth=6))
        paths = [path.replace("/...", "") for path in paths]


with h5py.File(args.info, "r") as data:

    files = data["/files"].asstr()[...]
    N = data["/normalisation/N"][...]
    sig0 = data["/normalisation/sig0"][...]
    sigc = data["/averages/sigd_bottom"][...] * sig0
    sign = data["/averages/sigd_top"][...] * sig0

    stress_names = [
        "stress=0d6",
        "stress=1d6",
        "stress=2d6",
        "stress=3d6",
        "stress=4d6",
        "stress=5d6",
        "stress=6d6",
    ]

    stresses = [
        0.0 * (sign - sigc) / 6.0 + sigc,
        1.0 * (sign - sigc) / 6.0 + sigc,
        2.0 * (sign - sigc) / 6.0 + sigc,
        3.0 * (sign - sigc) / 6.0 + sigc,
        4.0 * (sign - sigc) / 6.0 + sigc,
        5.0 * (sign - sigc) / 6.0 + sigc,
        6.0 * (sign - sigc) / 6.0 + sigc,
    ]

    commands = []

    for file, (stress, stress_name) in itertools.product(
        files, zip(stresses, stress_names)
    ):

        simid = file.replace(".hdf5", "")

        a = data["full"][file]["A"][...]
        sig = data["full"][file]["sigd"][...] * sig0
        i = data["full"][file]["steadystate"][...]
        a[:i] = 0
        sig[:i] = 0.0
        ss = np.argwhere(a == N).ravel()
        trigger = []

        for i, j in zip(ss[:-1], ss[1:]):
            if stress >= sig[i] and stress <= sig[j - 1]:
                trigger += [i]

        for element, A, incc in itertools.product([0, int(N / 2)], [1200], trigger):

            root = (
                f"/data/{stress_name}/A={A:d}/{simid}/incc={incc:d}/element={element:d}"
            )
            if root in paths:
                continue

            output = (
                f"{stress_name}_A={A:d}_{simid}_incc={incc:d}_element={element:d}.hdf5"
            )
            cmd = f"{executable} -f {file} -o {output} -s {stress:.8e} -i {incc:d} -e {element:d} -a {A:d}"
            commands += [cmd]


slurm = """
# print jobid
echo "SLURM_JOBID = ${{SLURM_JOBID}}"
echo ""

# for safety set the number of cores
export OMP_NUM_THREADS=1

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

if [[ "${{SYS_TYPE}}" == *E5v4* ]]; then
    conda activate code_velocity_E5v4
elif [[ "${{SYS_TYPE}}" == *s6g1* ]]; then
    conda activate code_velocity_s6g1
elif [[ "${{SYS_TYPE}}" == *S6g1* ]]; then
    conda activate code_velocity_s6g1
else
    echo "Unknown SYS_TYPE ${{SYS_TYPE}}"
    exit 1
fi

{0:s}
"""

commands = ["stdbuf -o0 -e0 " + cmd for cmd in commands]

ngroup = int(np.ceil(len(commands) / args.group))
fmt = str(int(np.ceil(np.log10(ngroup))))

for group in range(ngroup):

    ii = group * args.group
    jj = (group + 1) * args.group
    c = commands[ii:jj]
    command = "\n".join(c)
    command = slurm.format(command)

    jobname = ("{0:s}-{1:0" + fmt + "d}").format(
        args.executable.replace(" ", "_"), group
    )

    sbatch = {
        "job-name": "velocity_" + jobname,
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
