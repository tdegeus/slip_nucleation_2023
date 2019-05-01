
import os, subprocess, h5py
import numpy      as np
import GooseSLURM as gs

# --------------------------------------------------------------------------------------------------

dbase = '../../data'
nx    = 'nx=3^6x2'
N     = (3**6) * 2

# --------------------------------------------------------------------------------------------------

commands = []

data = h5py.File(os.path.join(dbase, nx, 'AvalancheAfterPush_strain=00d10.hdf5'), 'r')

p_files = data['files'  ][...]
p_file  = data['file'   ][...]
p_elem  = data['element'][...]
p_A     = data['A'      ][...]
p_sig   = data['sigd0'  ][...]
p_sigc  = data['sig_c'  ][...]
p_incc  = data['inc_c'  ][...]

data.close()

idx = np.arange(len(p_incc))
np.random.shuffle(idx)

dirnames = []

for i in idx[:100]:

  command = {
    'file'   : os.path.join('..', dbase, nx, p_files[p_file[i]]),
    'element': p_elem[i],
    'incc'   : p_incc[i],
  }

  command['output'] = \
    '../ensemble_{nx:s}/{nx:s}_{fname:s}_elem={element:04d}_incc={incc:03d}.hdf5'.format(
    nx = nx,
    fname = p_files[p_file[i]].replace('.hdf5', ''),
    **command)

  commands += [command]

  dirnames += ['ensemble_{nx:s}'.format(nx=nx)]

lines = ['./Run --file {file:s} --element {element:d} --incc {incc:d} --output {output:s}'.format(**c) for c in commands]

dirnames = list(set(dirnames))

# --------------------------------------------------------------------------------------------------

for dirname in dirnames:

  if not os.path.isdir(dirname):
    os.makedirs(dirname)

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
