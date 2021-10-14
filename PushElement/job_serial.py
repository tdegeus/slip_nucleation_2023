import os
import subprocess

import GooseSLURM as gs
import h5py

# ----

files = sorted(
    list(
        filter(
            None,
            subprocess.check_output("find . -iname 'id*.hdf5'", shell=True)
            .decode("utf-8")
            .split("\n"),
        )
    )
)


def getCompleted(file):
    with h5py.File(file, "r") as data:
        if "completed" in data:
            return int(data["completed"][...])
    return 0


files = [os.path.relpath(file) for file in files if getCompleted(file) != 200]

# ----

slurm = """
# for safety set the number of cores
export OMP_NUM_THREADS=1

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

if [[ "${{SYS_TYPE}}" == *E5v4* ]]; then
    conda activate code_velocity_E5v4
elif [[ "${{SYS_TYPE}}" == *s6g1* ]]; then
    conda activate code_velocity_s6g1
fi

{0:s}
"""

for file in files:

    basename = os.path.splitext(file)[0]

    command = f'PushElement --input="{file:s}" --output="{basename:s}"'
    command = slurm.format(command)

    sbatch = {
        "job-name": basename,
        "out": basename + ".out",
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": 1,
        "time": "12h",
        "account": "pcsl",
        "partition": "serial",
    }

    open(basename + ".slurm", "w").write(gs.scripts.plain(command=slurm.format(command), **sbatch))
