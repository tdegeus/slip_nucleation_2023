r'''
Run events small than pre-run events. This avoid rerunning small events.
'''

import argparse
import GooseHDF5 as g5
import GooseSLURM
import h5py
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("collection", type=str, help="Result of PinAndTrigger_collect.py for earlier ran simulations")
parser.add_argument("-A", "--size", type=int, default=600, help="Size of the events to simulate")
parser.add_argument("-n", "--group", type=int, default=50)
parser.add_argument("-e", "--executable", type=str, default="python PinAndTrigger.py")
args = parser.parse_args()
assert os.path.isfile(os.path.realpath(args.collection))

commands = []

with h5py.File(args.collection, "r") as data:

    paths = list(g5.getpaths(data, root="/data", max_depth=6))
    paths = [path.replace('/...', '') for path in paths]
    A = np.array([int(data[g5.join(path, "A")][...]) for path in paths])

    for path in np.array(paths)[A >= args.size]:

        _, _, stress_name, _, simid, incc, element = path.split("/")

        keys = dict(
            A = args.size,
            element = int(element.split("=")[1]),
            executable = args.executable,
            file = data[path]["file"].asstr()[...],
            id = simid,
            incc = int(incc.split("=")[1]),
            stress = data[path]["target_stress"][...],
            stress_name = stress_name,
        )

        root = "/data/{stress_name:s}/A={A:d}/{id:s}/incc={incc:d}/element={element:d}".format(**keys)
        if root in paths:
            continue

        keys['output'] = "{stress_name:s}_A={A:d}_{id:s}_incc={incc:d}_element={element:d}.hdf5".format(**keys)
        cmd = '{executable:s} -f {file:s} -o {output:s} -s {stress:.8e} -i {incc:d} -e {element:d} -a {A:d}'.format(**keys)
        commands += [cmd]


slurm = '''
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
'''

commands = ['stdbuf -o0 -e0 ' + cmd for cmd in commands]

ngroup = int(np.ceil(len(commands) / args.group))
fmt = str(int(np.ceil(np.log10(ngroup))))

for group in range(ngroup):

    c = commands[group * args.group: (group + 1) * args.group]
    command = '\n'.join(c)
    command = slurm.format(command)

    jobname = ('{0:s}-{1:0' + fmt + 'd}').format(args.executable.replace(' ', '_'), group)

    sbatch = {
        'job-name': 'velocity_' + jobname,
        'out': jobname + '.out',
        'nodes': 1,
        'ntasks': 1,
        'cpus-per-task': 1,
        'time': '24h',
        'account': 'pcsl',
        'partition': 'serial',
    }

    open(jobname + '.slurm', 'w').write(GooseSLURM.scripts.plain(command=command, **sbatch))


