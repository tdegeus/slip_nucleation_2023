
import os, subprocess, h5py
import numpy as np
import GooseSLURM as gs

dbase = '../../../data/nx=3^6x2'
nx = 'nx=3^6x2'
N = (3**6) * 2

with h5py.File(os.path.join(dbase, 'EnsembleInfo.hdf5'), 'r') as data:

    sig0 = data['/normalisation/sig0'  ][...]
    sigc = data['/averages/sigd_bottom'][...] * sig0
    sign = data['/averages/sigd_top'   ][...] * sig0

stress_names = [
    'stress=0d6',
    'stress=1d6',
    'stress=2d6',
    'stress=3d6',
    'stress=4d6',
    'stress=5d6',
    'stress=6d6',
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

    with h5py.File(os.path.join(dbase, 'AvalancheAfterPush_%s.hdf5' % name), 'r') as data:

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

    idx = np.argsort(np.abs(p_sigc - sigc))

    full_dir = os.path.join(dbase, 'EventEvolution_' + name)

    for i in idx:

        fname = '{id:s}_elem={element:04d}_incc={incc:03d}.hdf5'.format(
            element = p_elem[i],
            incc = p_incc[i],
            id = p_files[p_file[i]].replace('.hdf5', ''))

        if os.path.isfile(os.path.join(full_dir, fname)):
            print('Skipping')
            continue

        commands += [{
            'file'   : os.path.join('..', dbase, p_files[p_file[i]]),
            'element': p_elem[i],
            'incc'   : p_incc[i],
            'output' : fname,
            'stress' : stress,
        }]

    lines = ['EventEvolution_stress --file {file:s} --element {element:d} --incc {incc:d} --stress {stress:.8e} --output {output:s}'.format(**c) for c in commands]

    return lines

# --------------------------------------------------------------------------------------------------

slurm = '''
# for safety set the number of cores
export OMP_NUM_THREADS=1

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
if [[ "${{SYS_TYPE}}" == *E5v4* ]]; then
    conda activate code_velocity_E5v4
elif [[ "${{SYS_TYPE}}" == *s6g1* ]]; then
    conda activate code_velocity_s6g1
fi

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
        }

        open(os.path.join(dirname, basename + '.slurm'), 'w').write(
            gs.scripts.plain(command=slurm.format(command), **sbatch))
