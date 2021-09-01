import os
import h5py
import numpy as np
import GooseSLURM
import argparse
import shutil

dbase = os.path.relpath(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "../../data/nx=3^6x2"))
)
N = 2 * (3 ** 6)

with h5py.File(os.path.join(dbase, "Run", "EnsembleInfo.h5"), "r") as data:

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
source = []
dest = []

for name, stress in zip(stress_names, stresses):

    if not os.path.isdir(name):
        os.makedirs(name)

    with h5py.File(
        os.path.join(dbase, "AvalancheAfterPush", f"{name:s}.hdf5"), "r"
    ) as data:

        p_files = data["files"].asstr()[...]
        p_file = data["file"][...]
        p_elem = data["element"][...]
        p_A = data["A"][...]
        p_S = data["S"][...]
        p_sig = data["sigd0"][...]
        p_sigc = data["sig_c"][...]
        p_incc = data["inc_c"][...]

    i = p_A != N
    p_file = p_file[i]
    p_elem = p_elem[i]
    p_A = p_A[i]
    p_S = p_S[i]
    p_sig = p_sig[i]
    p_sigc = p_sigc[i]
    p_incc = p_incc[i]

    i = np.argsort(p_A)
    p_file = p_file[i][-100:]
    p_elem = p_elem[i][-100:]
    p_A = p_A[i][-100:]
    p_S = p_S[i][-100:]
    p_sig = p_sig[i][-100:]
    p_sigc = p_sigc[i][-100:]
    p_incc = p_incc[i][-100:]

    for i in range(p_file.size):

        fname = "{stress:s}/{id:s}_elem={element:04d}_incc={incc:03d}.hdf5".format(
            stress=name,
            element=p_elem[i],
            incc=p_incc[i],
            id=p_files[p_file[i]].replace(".hdf5", ""),
        )

        if os.path.isfile(os.path.join(dbase, "EventEvolution", fname)):
            continue

        commands += [
            "EventEvolution_stress --file {file:s} --element {element:d} --incc {incc:d} --stress {stress:.8e} --output {output:s}".format(
                file=os.path.join("Run", p_files[p_file[i]]),
                element=p_elem[i],
                incc=p_incc[i],
                output=fname,
                stress=stress,
            )
        ]

        s = os.path.join(dbase, "Run", p_files[p_file[i]])
        d = os.path.join("Run", p_files[p_file[i]])

        if s not in source:
            source += [s]
            dest += [d]

if not os.path.isdir("Run"):
    os.makedirs("Run")

for s, d in zip(source, dest):
    shutil.copy2(s, d)


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

args_group = 10
ngroup = int(np.ceil(len(commands) / args_group))
fmt = str(int(np.ceil(np.log10(ngroup))))

for group in range(ngroup):

    c = commands[group * args_group : (group + 1) * args_group]
    command = "\n".join(c)
    command = slurm.format(command)

    jobname = ("EventEvolution_stress-{0:0" + fmt + "d}").format(group)

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
