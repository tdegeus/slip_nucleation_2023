
import os, subprocess, h5py
import numpy      as np
import GooseSLURM as gs
import GooseHDF5 as g5

dbase = '../../../data'
nx = 'nx=3^6x2'
N = (3**6) * 2


data = h5py.File(os.path.join(dbase, nx, 'EnsembleInfo.hdf5'), 'r')

sig0 = data['/normalisation/sig0'  ][...]
sigc = data['/averages/sigd_bottom'][...] * sig0
sign = data['/averages/sigd_top'   ][...] * sig0

data.close()

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
  0. * (sign - sigc) / 6. + sigc,
  1. * (sign - sigc) / 6. + sigc,
  2. * (sign - sigc) / 6. + sigc,
  3. * (sign - sigc) / 6. + sigc,
  4. * (sign - sigc) / 6. + sigc,
  5. * (sign - sigc) / 6. + sigc,
  6. * (sign - sigc) / 6. + sigc,
]

commands = []

for stress_name, stress in zip(stress_names, stresses):

  if not os.path.isdir(stress_name):
    os.mkdir(stress_name)

  data = h5py.File(os.path.join(dbase, nx, 'AvalancheAfterPush_%s.hdf5' % stress_name), 'r')

  p_files = data['files'  ].asstr()[...]
  p_file  = data['file'   ][...]
  p_elem  = data['element'][...]
  p_A     = data['A'      ][...]
  p_sig   = data['sigd0'  ][...]
  p_sigc  = data['sig_c'  ][...]
  p_incc  = data['inc_c'  ][...]

  data.close()

  idx = np.where(p_A == N)[0]

  p_file = p_file[idx]
  p_elem = p_elem[idx]
  p_A    = p_A   [idx]
  p_sig  = p_sig [idx]
  p_sigc = p_sigc[idx]
  p_incc = p_incc[idx]

  idx = np.argsort(np.abs(p_sigc - sigc))

  for n, i in enumerate(idx):

    # file-name
    fname = '{id:s}_elem={element:04d}_incc={incc:03d}.hdf5'.format(
      element = p_elem[i],
      incc    = p_incc[i],
      id      = p_files[p_file[i]].replace('.hdf5', ''))

    # stop at 75 drawn files
    if n > 75:
      break

    source = os.path.join(dbase, nx, p_files[p_file[i]])
    dest = p_files[p_file[i]]

    with h5py.File(source, 'r') as data:

        paths = list(g5.getdatasets(data))
        paths.remove('/damping/alpha')

        with h5py.File(dest, 'w') as ret:
            g5.copydatasets(data, ret, paths)
            ret['/damping/alpha'] = data['/damping/alpha'][...] * 0


    commands += [{
      'file'   : dest,
      'element': p_elem[i],
      'incc'   : p_incc[i],
      'output' : os.path.join(stress_name, fname),
      'stress' : stress,
    }]

lines = ['CrackEvolution_raw_stress --tfac 2 --file {file:s} --element {element:d} --incc {incc:d} --stress {stress:.8e} --output {output:s}'.format(**c) for c in commands]

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

{command:s}
'''

# --------------------------------------------------------------------------------------------------

for i, line in enumerate(lines):

  fbase = 'job_{0:03d}'.format(i)

  # job-options
  sbatch = {
    'job-name'      : fbase,
    'out'           : fbase+'.out',
    'nodes'         : 1,
    'ntasks'        : 1,
    'cpus-per-task' : 1,
    'time'          : '6h',
    'account'       : 'pcsl',
    'partition'     : 'serial',
    'mem'           : '8G',
  }

  # write SLURM file
  open(fbase+'.slurm','w').write(gs.scripts.plain(command=slurm.format(command=line),**sbatch))
