import os

import GooseSLURM as gs
import h5py
import numpy as np

# --------------------------------------------------------------------------------------------------

dbase = "../../data"
nx = "nx=3^6x2"
N = (3**6) * 2

# --------------------------------------------------------------------------------------------------

commands = []

data = h5py.File(os.path.join(dbase, nx, "AvalancheAfterPush_strain=00d10.hdf5"), "r")

p_files = data["files"][...]
p_file = data["file"][...]
p_elem = data["element"][...]
p_A = data["A"][...]
p_sig = data["sigd0"][...]
p_sigc = data["sig_c"][...]
p_incc = data["inc_c"][...]

data.close()

idx = np.where(p_A == N)[0]
np.random.shuffle(idx)

sbase = f"../data/{nx:s}"
outdir = f"data/{nx:s}"

if not os.path.isdir(outdir):
    os.makedirs(outdir)

for n, i in enumerate(idx):

    # file-name
    fname = "{id:s}_elem={element:04d}_incc={incc:03d}.hdf5".format(
        element=p_elem[i], incc=p_incc[i], id=p_files[p_file[i]].replace(".hdf5", "")
    )

    # skip existing
    if os.path.isfile(os.path.join(sbase, fname)):
        continue

    # stop at 50 drawn files
    if n > 50:
        break

    commands += [
        {
            "file": os.path.join("..", dbase, nx, p_files[p_file[i]]),
            "element": p_elem[i],
            "incc": p_incc[i],
            "output": os.path.join("..", outdir, fname),
        }
    ]

lines = [
    "./Run --file {file:s} --element {element:d} --incc {incc:d} --output {output:s}".format(**c)
    for c in commands
]

# --------------------------------------------------------------------------------------------------

slurm = """
# for safety set the number of cores
export OMP_NUM_THREADS=1

# compile
cmake ../..
make

{command:s}
"""

# --------------------------------------------------------------------------------------------------

for i, line in enumerate(lines):

    dirname = f"job_{i:03d}"

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    fbase = "job"

    # job-options
    sbatch = {
        "job-name": fbase,
        "out": fbase + ".out",
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": 1,
        "time": "6h",
        "account": "pcsl",
        "partition": "serial",
    }

    # write SLURM file
    open(os.path.join(dirname, fbase + ".slurm"), "w").write(
        gs.scripts.plain(command=slurm.format(command=line), **sbatch)
    )
