import h5py
import os
import numpy as np
import GooseSLURM as gs

dbase = "../../../data/nx=3^6x2"
nx = "nx=3^6x2"
N = (3 ** 6) * 2

# --------------------------------------------------------------------------------------------------


def get_runs():

    commands = []

    with h5py.File(
        os.path.join(dbase, "AvalancheAfterPush_strain=00d10.hdf5"), "r"
    ) as data:

        p_files = data["files"].asstr()[...]
        p_file = data["file"][...]
        p_elem = data["element"][...]
        p_A = data["A"][...]
        p_S = data["S"][...]
        p_sig = data["sigd0"][...]
        p_sigc = data["sig_c"][...]
        p_incc = data["inc_c"][...]

    idx = np.argwhere(p_S > 1).ravel()

    p_file = p_file[idx]
    p_elem = p_elem[idx]
    p_A = p_A[idx]
    p_S = p_S[idx]
    p_sig = p_sig[idx]
    p_sigc = p_sigc[idx]
    p_incc = p_incc[idx]

    idx = np.argsort(p_sig)

    full_dir = os.path.join(dbase, "EventEvolution_strain=0")

    for i in idx:

        fname = "{id:s}_elem={element:04d}_incc={incc:03d}.hdf5".format(
            element=p_elem[i],
            incc=p_incc[i],
            id=p_files[p_file[i]].replace(".hdf5", ""),
        )

        if os.path.isfile(os.path.join(full_dir, fname)):
            print("Skipping")
            continue

        commands += [
            {
                "file": os.path.join(dbase, p_files[p_file[i]]),
                "element": p_elem[i],
                "incc": p_incc[i],
                "output": fname,
            }
        ]

    fmt = (
        "EventEvolution_strain "
        "--file {file:s} "
        "--element {element:d} "
        "--incc {incc:d} "
        "--output {output:s}"
    )

    lines = [fmt.format(**c) for c in commands]

    return lines


# --------------------------------------------------------------------------------------------------

slurm = """
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
fi

{0:s}
"""

# --------------------------------------------------------------------------------------------------

commands = get_runs()
commands_per_group = 10
ngroup = int(np.ceil(len(commands) / commands_per_group))
fmt = str(int(np.ceil(np.log10(ngroup))))

for group in range(ngroup):

    ii = group * commands_per_group
    jj = (group + 1) * commands_per_group
    c = commands[ii:jj]
    command = "\n".join(c)
    command = slurm.format(command)

    jobname = ("EventEvolution_strain-{0:0" + fmt + "d}").format(group)

    sbatch = {
        "job-name": jobname,
        "out": jobname + ".out",
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": 1,
        "time": "12h",
        "account": "pcsl",
        "partition": "serial",
    }

    open(jobname + ".slurm", "w").write(gs.scripts.plain(command=command, **sbatch))
