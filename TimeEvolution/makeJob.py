
import os, subprocess, h5py
import numpy      as np
import GooseSLURM as gs

# data folder
dbase = '../../data'

# ==================================================================================================
# get file/element/stress at which to push
# ==================================================================================================

nx = 'nx=3^6x2'
N  = (3**6) * 2

data = h5py.File(os.path.join(dbase, 'nx=3^6x2', 'EnsembleInfo.hdf5'), 'r')

sig0 = data['/normalisation/sigy'  ][...]
sigc = data['/averages/sigd_bottom'][...] * sig0
sign = data['/averages/sigd_top'   ][...] * sig0

data.close()

stress = sign

commands = []

for i in [6]:

  data = h5py.File(os.path.join(dbase, 'nx=3^6x2', 'AvalancheAfterPush_stress=%dd6.hdf5' % i), 'r')

  p_files = data['files'   ][...]
  p_file  = data['file'    ][...]
  p_elem  = data['element' ][...]
  p_A     = data['A'       ][...]
  p_sig   = data['sigd0'   ][...]
  p_sigc  = data['sig_c'   ][...]
  p_incc  = data['inc_c'   ][...]

  data.close()

  idx = np.where(p_A == N)[0]

  p_file = p_file[idx]
  p_elem = p_elem[idx]
  p_A    = p_A   [idx]
  p_sig  = p_sig [idx]
  p_sigc = p_sigc[idx]
  p_incc = p_incc[idx]

  idx = np.argsort(np.abs(p_sigc - sigc))
  i = idx[0]

  command = {
    'file'   : os.path.join(dbase, 'nx=3^6x2', p_files[p_file[i]]),
    'element': p_elem[i],
    'incc'   : p_incc[i],
    'stress' : stress,
  }

  command['output'] = '{nx:s}_{fname:s}_elem={element:d}_incc={incc:d}_stress=6d6.hdf5'.format(nx='nx=3^6x2', fname=p_files[p_file[i]].replace('.hdf5', ''), **command)

  commands += [command]

# convert to commands
lines = ['./Run --file {file:s} --element {element:d} --incc {incc:d} --stress {stress:.8e} --output {output:s}'.format(**c) for c in commands]
command = '\n'.join(lines)

# --------------------------------------------------------------------------------------------------

slurm = '''
# change current directory to the location of the sbatch command
cd "${{SLURM_SUBMIT_DIR}}"

# for safety set the number of cores
export OMP_NUM_THREADS=1

{command:s}
'''

# --------------------------------------------------------------------------------------------------

# job-options
sbatch = {
  'job-name'      : 'job',
  'out'           : 'job.out',
  'nodes'         : 1,
  'ntasks'        : 1,
  'cpus-per-task' : 1,
  'time'          : '6h',
  'account'       : 'pcsl',
  'partition'     : 'serial',
}

# write SLURM file
open('job.slurm','w').write(gs.scripts.plain(command=slurm.format(command=command),**sbatch))

# open('job.txt')







  # print(idx)








# # ==================================================================================================
# # define relative stress at which to measure
# # ==================================================================================================

# denominator  = 6
# stresses     = np.linspace(0, 1, denominator+1)
# stresses[-1] = .99

# # ==================================================================================================
# # get all ensembles
# # ==================================================================================================

# files = subprocess.check_output("find ../../data -iname 'EnsembleInfo.hdf5'",shell=True).decode('utf-8')
# files = list(filter(None, files.split('\n')))

# ensembles = {}

# for file in files:

#   ensemble = file.replace('/EnsembleInfo.hdf5', '')

#   ensembles[ensemble] = file

# # ==================================================================================================
# # write commands
# # ==================================================================================================

# commands = []

# for ensemble in sorted(ensembles):

#   for istress, stress in enumerate(stresses):

#     nx = 'nx='+ensemble.split('nx=')[1].split('/')[0]
#     nu = 'nu='+ensemble.split('nu=')[1].split('/')[0]

#     logname = 'debug_{nx:s}_{nu:s}_stress={0:d}d{1:d}.log'.format(istress, denominator, nx=nx, nu=nu)

#     outname = 'EnsembleYieldDistance_stress={0:d}d{1:d}.hdf5'.format(istress, denominator)

#     commands += ['./Run --stress {0:.8e} --outname "{1:s}" "{2:s}" > "{3:s}"'.format(stress,outname,ensembles[ensemble],logname)]

# # --------------------------------------------------------------------------------------------------

# slurm = '''
# # change current directory to the location of the sbatch command
# cd "${{SLURM_SUBMIT_DIR}}"

# # for safety set the number of cores
# export OMP_NUM_THREADS=1

# # signal start of job
# echo "started" > {fbase:s}.lock

# # run in parallel
# parallel --max-procs=${{SLURM_CPUS_PER_TASK}} :::: {fbase:s}.txt

# # remove 'lock'-file, to signal that the job has completed
# rm {fbase:s}.lock

# # write that the job was completed
# echo "completed" > {fbase:s}.done
# '''

# # --------------------------------------------------------------------------------------------------

# # base of the file-name of the job-file
# fbase = 'EnsembleYieldDistance_stressControl'

# # write commands to text file
# open(fbase+'.txt','w').write('\n'.join(commands))

# # job-options
# sbatch = {
#   'job-name'      : fbase,
#   'out'           : fbase+'.out',
#   'nodes'         : 1,
#   'ntasks'        : 1,
#   'cpus-per-task' : 28,
#   'time'          : '6h',
#   'account'       : 'pcsl',
# }

# # write SLURM file
# open(fbase+'.slurm','w').write(gs.scripts.plain(command=slurm.format(fbase=fbase),**sbatch))

# # ==================================================================================================
# # write copy commands
# # ==================================================================================================

# commands = []

# for ensemble in sorted(ensembles):

#   for istress, stress in enumerate(stresses):

#     outname = 'EnsembleYieldDistance_stress={0:d}d{1:d}.hdf5'.format(istress, denominator)

#     dest = os.path.join(ensemble, outname)

#     src = os.path.normpath(os.path.join('~/data/22_depinning-inertia/EnsembleYieldDistance_stressControl/build/', dest))

#     command = 'echo {src:s};\nscp fidis:{src:s} {dest:s}'.format(src=src, dest=dest)

#     commands += [command]

# # write copy-commands to text file
# open(fbase+'.scp','w').write('\n'.join(commands))
