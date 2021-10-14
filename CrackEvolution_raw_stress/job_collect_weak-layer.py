import GooseSLURM as gs

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

{command:s}
"""

fbase = "job_collect_weak-layer"
info = "../../../data/nx=3^6x2/EnsembleInfo.hdf5"
cmd = f"python ../collect_weak-layer.py --force -i {info} shelephant_dump.yaml weak"

sbatch = {
    "job-name": fbase,
    "out": fbase + ".out",
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 1,
    "time": "10h",
    "account": "pcsl",
    "partition": "serial",
    "mem": "8G",
}

open(fbase + ".slurm", "w").write(gs.scripts.plain(command=slurm.format(command=cmd), **sbatch))
