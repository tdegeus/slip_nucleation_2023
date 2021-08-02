import argparse
import GooseSLURM
import h5py
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("info", type=str, help="EnsembleInfo (read-only)")
parser.add_argument('-n', '--group', type=int, default=100)
parser.add_argument('-e', '--executable', type=str, default='python PinAndTrigger.py')
args = parser.parse_args()
assert os.path.isfile(os.path.realpath(args.info))


with h5py.File(args.info, "r") as data:

    files = data['/files'].asstr()[...]
    N = data['/normalisation/N'][...]
    sig0 = data['/normalisation/sig0'][...]
    sigc = data['/averages/sigd_bottom'][...] * sig0
    sign = data['/averages/sigd_top'][...] * sig0

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
        0.0 * (sign - sigc) / 6.0 + sigc,
        1.0 * (sign - sigc) / 6.0 + sigc,
        2.0 * (sign - sigc) / 6.0 + sigc,
        3.0 * (sign - sigc) / 6.0 + sigc,
        4.0 * (sign - sigc) / 6.0 + sigc,
        5.0 * (sign - sigc) / 6.0 + sigc,
        6.0 * (sign - sigc) / 6.0 + sigc,
    ]

    commands = []

    for file in files:

        for stress, sname in zip(stresses, stress_names):

            # todo: select allowed stress!

            A = data["full"][file]["A"][...]
            sig = data["full"][file]["sigd"][...] * sig0
            i = data["full"][file]["steadystate"][...]
            A[:i] = 0
            sig[:i] = 0.0
            ss = np.argwhere(A == N).ravel()
            trigger = []

            for i, j in zip(ss[:-1], ss[1:]):
                if stress >= sig[i] and stress <= sig[j - 1]:
                    trigger += [i]

            # todo: skip based on stress

            for element in [0]:

                for A in [200]:

                    for i in trigger:

                        keys = dict(
                            executable = args.executable,
                            file = file,
                            stress_name = sname,
                            fid = file.replace(".hdf5", ""),
                            stress = stress,
                            incc = i,
                            element = element,
                            A = A)

                        keys['output'] = "{stress_name:s}_A={A:d}_{fid:s}_incc={incc:d}_element={element:d}.hdf5".format(**keys)
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


