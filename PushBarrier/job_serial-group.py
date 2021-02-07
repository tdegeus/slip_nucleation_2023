
import sys
import os
import re
import subprocess
import h5py
import GooseSLURM as gs
import numpy as np

# ----

files = sorted(list(filter(None, subprocess.check_output(
    "find . -iname 'id*.hdf5'", shell=True).decode('utf-8').split('\n'))))
files = [str(i) + '.hdf5' for i in range(223)]


# ----

slurm = '''
# for safety set the number of cores
export OMP_NUM_THREADS=1

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

if [[ "${{SYS_TYPE}}" == *E5v4* ]]; then
    conda activate code_velocity_E5v4
elif [[ "${{SYS_TYPE}}" == *s6g1* ]]; then
    conda activate code_velocity_s6g1
else
    echo "Unknown SYS_TYPE ${{SYS_TYPE}}"
fi

{0:s}
'''

files_per_group = 10
print(files)

for group in range(int(np.ceil(len(files) / files_per_group))):

    f = files[group * files_per_group: (group + 1) * files_per_group]
    c = []

    for file in f:
        basename = os.path.splitext(os.path.normpath(file))[0]
        c += ['PushBarrier "{0:s}" "{1:s}"'.format(file, basename)]

    command = '\n'.join(c)
    command = slurm.format(command)

    jobname = 'PushBarrier-{0:d}'.format(group)

    sbatch = {
        'job-name': jobname,
        'out': jobname + '.out',
        'nodes': 1,
        'ntasks': 1,
        'cpus-per-task': 1,
        'time': '12h',
        'account': 'pcsl',
        'partition': 'serial',
        'mem' : '8G',
    }

    open(jobname + '.slurm', 'w').write(gs.scripts.plain(command=command, **sbatch))
