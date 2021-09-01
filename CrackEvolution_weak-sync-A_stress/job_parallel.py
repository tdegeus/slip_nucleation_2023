import h5py
import os
import numpy as np
import GooseSLURM as gs

dbase = "../../../data/nx=3^6x2"
nx = "nx=3^6x2"
N = (3 ** 6) * 2

with h5py.File(os.path.join(dbase, "EnsembleInfo.hdf5"), "r") as data:

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
    0.0 * (sign - sigc) / 6.9 + sigc,
    1.0 * (sign - sigc) / 6.9 + sigc,
    2.0 * (sign - sigc) / 6.9 + sigc,
    3.0 * (sign - sigc) / 6.9 + sigc,
    4.0 * (sign - sigc) / 6.9 + sigc,
    5.0 * (sign - sigc) / 6.9 + sigc,
    6.0 * (sign - sigc) / 6.9 + sigc,
]

# --------------------------------------------------------------------------------------------------


def get_runs(name, stress):

    commands = []

    with h5py.File(
        os.path.join(dbase, "AvalancheAfterPush_%s.hdf5" % name), "r"
    ) as data:

        p_files = data["files"][...]
        p_file = data["file"][...]
        p_elem = data["element"][...]
        p_A = data["A"][...]
        p_sig = data["sigd0"][...]
        p_sigc = data["sig_c"][...]
        p_incc = data["inc_c"][...]

    idx = np.where(p_A == N)[0]

    p_file = p_file[idx]
    p_elem = p_elem[idx]
    p_A = p_A[idx]
    p_sig = p_sig[idx]
    p_sigc = p_sigc[idx]
    p_incc = p_incc[idx]

    idx = np.argsort(np.abs(p_sigc - sigc))

    full_dir = os.path.join(dbase, "CrackEvolution_" + name)

    for i in idx:

        fname = "{id:s}_elem={element:04d}_incc={incc:03d}.hdf5".format(
            element=p_elem[i],
            incc=p_incc[i],
            id=p_files[p_file[i]].replace(".hdf5", ""),
        )

        if os.path.isfile(os.path.join(full_dir, fname)):
            print("Skipping full")
            continue

        commands += [
            {
                "file": os.path.join("..", dbase, p_files[p_file[i]]),
                "element": p_elem[i],
                "incc": p_incc[i],
                "output": fname,
                "stress": stress,
            }
        ]

    lines = [
        "./CrackEvolution_stress --file {file:s} --element {element:d} --incc {incc:d} --stress {stress:.8e} --output {output:s}".format(
            **c
        )
        for c in commands
    ]

    return lines


# --------------------------------------------------------------------------------------------------

slurm = """
# for safety set the number of cores
export OMP_NUM_THREADS=1

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate code_velocity

# compile
cmake ../..
make

parallel --max-procs=${SLURM_CPUS_PER_TASK} :::: commands.txt
"""

# --------------------------------------------------------------------------------------------------

for name, stress in zip(stress_names, stresses):

    commands = get_runs(name, stress)
    commands = commands[: 2 * 28]

    dirname = "CrackEvolution_weak-sync-A_" + name

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    open(os.path.join(dirname, "commands.txt"), "w").write("\n".join(commands))

    sbatch = {
        "job-name": f"velocity_{name:s}",
        "out": "job.out",
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": 28,
        "time": "1d",
        "account": "pcsl",
    }

    open(os.path.join(dirname, "job.slurm"), "w").write(
        gs.scripts.plain(command=slurm, **sbatch)
    )
