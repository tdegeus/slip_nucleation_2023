
import os, subprocess, h5py
import numpy as np
import GooseSLURM as gs

dbase = '../../../data/nx=3^6x2'
nx = 'nx=3^6x2'
N = (3**6) * 2

# --------------------------------------------------------------------------------------------------

def get_runs():

    commands = []

    with h5py.File(os.path.join(dbase, 'AvalancheAfterPush_strain=00d10.hdf5'), 'r') as data:

        p_files = data['files'  ][...]
        p_file  = data['file'   ][...]
        p_elem  = data['element'][...]
        p_A     = data['A'      ][...]
        p_sig   = data['sigd0'  ][...]
        p_sigc  = data['sig_c'  ][...]
        p_incc  = data['inc_c'  ][...]

    idx = np.where(p_A > 10)[0]

    p_file = p_file[idx]
    p_elem = p_elem[idx]
    p_A    = p_A   [idx]
    p_sig  = p_sig [idx]
    p_sigc = p_sigc[idx]
    p_incc = p_incc[idx]

    idx = np.argsort(p_sig)

    full_dir = os.path.join(dbase, 'EventEvolution_strain=0')

    for i in idx:

        fname = '{id:s}_elem={element:04d}_incc={incc:03d}.hdf5'.format(
            element = p_elem[i],
            incc = p_incc[i],
            id = p_files[p_file[i]].replace('.hdf5', ''))

        if os.path.isfile(os.path.join(full_dir, fname)):
            print('Skipping full')
            continue

        commands += [{
            'file'   : os.path.join('..', dbase, p_files[p_file[i]]),
            'element': p_elem[i],
            'incc'   : p_incc[i],
            'output' : fname,
        }]

    lines = ['EventEvolution_strain --file {file:s} --element {element:d} --incc {incc:d} --output {output:s}'.format(**c) for c in commands]

    return lines

# --------------------------------------------------------------------------------------------------

slurm = '''
# for safety set the number of cores
export OMP_NUM_THREADS=1

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate code_velocity

{0:s}
'''

# --------------------------------------------------------------------------------------------------

for name, stress in zip([stress_names[0]], [stresses[0]]):

    commands = get_runs(name, stress)
    commands = commands[:5]

    dirname = 'EventEvolution_' + name

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    for i, command in enumerate(commands):

        basename = 'job{0:d}'.format(i)

        sbatch = {
            'job-name': 'EventEvolution_{0:s}_{1:d}'.format(name, i),
            'out': basename + '.out',
            'nodes': 1,
            'ntasks': 1,
            'cpus-per-task': 1,
            'time': '3h',
            'account': 'pcsl',
            'partition': 'serial',
            'constraint': 'E5v4',
        }

        open(os.path.join(dirname, basename + '.slurm'), 'w').write(
            gs.scripts.plain(command=slurm.format(command), **sbatch))
