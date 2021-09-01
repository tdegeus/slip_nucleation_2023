import os
import subprocess
import h5py
import GooseSLURM as gs
import GooseHDF5 as g5

dbase = "../../../data"
nx = "nx=3^6x2"
N = (3 ** 6) * 2

with h5py.File(os.path.join(dbase, nx, "EnsembleInfo.hdf5"), "r") as data:

    sig0 = data["/normalisation/sig0"][...]
    sigc = data["/averages/sigd_bottom"][...] * sig0
    sign = data["/averages/sigd_top"][...] * sig0

Stress_names = [
    "stress=0d6",
    "stress=1d6",
    "stress=2d6",
    "stress=3d6",
    "stress=4d6",
    "stress=5d6",
    "stress=6d6",
]

Stress_values = [
    0.0 * (sign - sigc) / 6.0 + sigc,
    1.0 * (sign - sigc) / 6.0 + sigc,
    2.0 * (sign - sigc) / 6.0 + sigc,
    3.0 * (sign - sigc) / 6.0 + sigc,
    4.0 * (sign - sigc) / 6.0 + sigc,
    5.0 * (sign - sigc) / 6.0 + sigc,
    6.0 * (sign - sigc) / 6.0 + sigc,
]

commands = []

for stress_name, stress_value in zip(Stress_names, Stress_values):

    if not os.path.isdir(stress_name):
        os.mkdir(stress_name)

    fol = os.path.join(dbase, nx, "CrackEvolution_" + stress_name)

    files = sorted(
        list(
            filter(
                None,
                subprocess.check_output(
                    f"find {fol:s} -maxdepth 1 -iname 'id*.hdf5'", shell=True
                )
                .decode("utf-8")
                .split("\n"),
            )
        )
    )

    files = [f.split("/")[-1] for f in files]

    for file in files:

        dest = file.split("_")[0] + ".hdf5"
        source = os.path.join(dbase, nx, dest)

        with h5py.File(source, "r") as data:

            paths = list(g5.getdatasets(data))
            paths.remove("/damping/alpha")

            with h5py.File(dest, "w") as ret:
                g5.copydatasets(data, ret, paths)
                ret["/damping/alpha"] = data["/damping/alpha"][...] * 0

        commands += [
            {
                "file": dest,
                "element": int(file.split("elem=")[1].split("_")[0]),
                "incc": int(file.split("incc=")[1].split(".")[0]),
                "output": os.path.join(stress_name, file),
                "stress": stress_value,
            }
        ]

fmt = (
    "CrackEvolution_raw_stress "
    "--tfac 10 "
    "--file {file:s} "
    "--element {element:d} "
    "--incc {incc:d} "
    "--stress {stress:.8e} "
    "--output {output:s}"
)

lines = [fmt.format(**c) for c in commands]


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

for i, line in enumerate(lines):

    fbase = f"job_{i:03d}"

    sbatch = {
        "job-name": fbase,
        "out": fbase + ".out",
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": 1,
        "time": "6h",
        "account": "pcsl",
        "partition": "serial",
        "mem": "8G",
    }

    open(fbase + ".slurm", "w").write(
        gs.scripts.plain(command=slurm.format(command=line), **sbatch)
    )
