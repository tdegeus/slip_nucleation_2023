import GooseSLURM as gs

slurm = """
# for safety set the number of cores
export OMP_NUM_THREADS=1

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

conda activate code_collect_s6g1

{command:s}
"""

fbase = "job_collect_forces"
cmd = [
    r"python ../collect_forces.py --force -i ../../../data/nx=3\^6x2/EnsembleInfo.hdf5 -o out_stress=0d6.hdf5 list_stress=0d6.yaml",
    r"python ../collect_forces.py --force -i ../../../data/nx=3\^6x2/EnsembleInfo.hdf5 -o out_stress=1d6.hdf5 list_stress=1d6.yaml",
    r"python ../collect_forces.py --force -i ../../../data/nx=3\^6x2/EnsembleInfo.hdf5 -o out_stress=2d6.hdf5 list_stress=2d6.yaml",
    r"python ../collect_forces.py --force -i ../../../data/nx=3\^6x2/EnsembleInfo.hdf5 -o out_stress=3d6.hdf5 list_stress=3d6.yaml",
    r"python ../collect_forces.py --force -i ../../../data/nx=3\^6x2/EnsembleInfo.hdf5 -o out_stress=4d6.hdf5 list_stress=4d6.yaml",
    r"python ../collect_forces.py --force -i ../../../data/nx=3\^6x2/EnsembleInfo.hdf5 -o out_stress=5d6.hdf5 list_stress=5d6.yaml",
    r"python ../collect_forces.py --force -i ../../../data/nx=3\^6x2/EnsembleInfo.hdf5 -o out_stress=6d6.hdf5 list_stress=6d6.yaml",
]

for i, c in enumerate(cmd):

    f = fbase + "_" + str(i)

    sbatch = {
        "job-name": f,
        "out": f + ".out",
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": 1,
        "time": "10h",
        "account": "pcsl",
        "partition": "serial",
        "mem": "8G",
    }

    open(f + ".slurm", "w").write(
        gs.scripts.plain(command=slurm.format(command=c), **sbatch)
    )
