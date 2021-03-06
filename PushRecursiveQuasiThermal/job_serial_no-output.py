
import sys
import os
import re
import subprocess
import h5py
import GooseSLURM as gs

# ----

files = sorted(list(filter(None, subprocess.check_output(
    "find . -iname 'id*.hdf5'", shell=True).decode('utf-8').split('\n'))))

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

for file in files:

    basename = os.path.splitext(os.path.normpath(file))[0]

    command = 'PushRecursiveQuasiThermal "{0:s}"'.format(file)
    command = slurm.format(command)

    sbatch = {
        'job-name': basename,
        'out': basename + '.out',
        'nodes': 1,
        'ntasks': 1,
        'cpus-per-task': 1,
        'time': '12h',
        'account': 'pcsl',
        'partition': 'serial',
        'mem': '8G',
    }

    open(basename + '.slurm', 'w').write(gs.scripts.plain(command=slurm.format(command), **sbatch))
