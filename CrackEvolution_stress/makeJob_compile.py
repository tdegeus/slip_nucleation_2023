
import os, subprocess, h5py
import numpy      as np
import GooseSLURM as gs

# --------------------------------------------------------------------------------------------------

dbase = '../../data'
nx    = 'nx=3^6x2'
N     = (3**6) * 2

# --------------------------------------------------------------------------------------------------

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

  data = h5py.File(os.path.join(dbase, nx, 'AvalancheAfterPush_%s.hdf5' % stress_name), 'r')

  p_files = data['files'  ][...]
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

  sbase  = '../data/{nx:s}_{stress:s}'.format(nx=nx, stress=stress_name)
  outdir = 'data/{nx:s}_{stress:s}'.format(nx=nx, stress=stress_name)

  if not os.path.isdir(outdir):
    os.makedirs(outdir)

  for n, i in enumerate(idx):

    # file-name
    fname = '{id:s}_elem={element:04d}_incc={incc:03d}.hdf5'.format(
      element = p_elem[i],
      incc    = p_incc[i],
      id      = p_files[p_file[i]].replace('.hdf5', ''))

    # skip existing
    if os.path.isfile(os.path.join(sbase, fname)):
      continue

    # stop at 75 drawn files
    if n > 75:
      break

    commands += [{
      'file'   : os.path.join('..', dbase, nx, p_files[p_file[i]]),
      'element': p_elem[i],
      'incc'   : p_incc[i],
      'output' : os.path.join('..', outdir, fname),
      'stress' : stress,
    }]

lines = ['./Run --file {file:s} --element {element:d} --incc {incc:d} --stress {stress:.8e} --output {output:s}'.format(**c) for c in commands]

# --------------------------------------------------------------------------------------------------

slurm = '''
# for safety set the number of cores
export OMP_NUM_THREADS=1

# compile
cmake ../..
make

{command:s}
'''

# --------------------------------------------------------------------------------------------------

for i, line in enumerate(lines):

  dirname = 'job_{0:03d}'.format(i)

  if not os.path.isdir(dirname): os.makedirs(dirname)

  fbase = 'job'

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
  }

  # write SLURM file
  open(os.path.join(dirname, fbase+'.slurm'),'w').write(gs.scripts.plain(command=slurm.format(command=line),**sbatch))