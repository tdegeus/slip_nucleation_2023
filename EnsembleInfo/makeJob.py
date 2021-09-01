# import libraries
import os, subprocess
import GooseSLURM as gs

# ==================================================================================================
# get all simulation files, split in ensembles
# ==================================================================================================

files = subprocess.check_output("find ../../data -iname 'id*.hdf5'", shell=True).decode(
    "utf-8"
)
files = list(filter(None, files.split("\n")))

enembles = {}

for ensemble in list({file.split("id=")[0] for file in files}):

    enembles[ensemble] = sorted(file for file in files if len(file.split(ensemble)) > 1)

# ==================================================================================================
# write commands
# ==================================================================================================

commands = []

for ensemble in sorted(enembles):

    files = enembles[ensemble]

    commands += ["./Run {files:s}".format(files=" ".join(files))]

# --------------------------------------------------------------------------------------------------

slurm = """
# change current directory to the location of the sbatch command
cd "${{SLURM_SUBMIT_DIR}}"

# for safety set the number of cores
export OMP_NUM_THREADS=1

# signal start of job
echo "started" > {fbase:s}.lock

# run in parallel
parallel --max-procs=${{SLURM_CPUS_PER_TASK}} :::: {fbase:s}.txt

# remove 'lock'-file, to signal that the job has completed
rm {fbase:s}.lock

# write that the job was completed
echo "completed" > {fbase:s}.done
"""

# --------------------------------------------------------------------------------------------------

# base of the file-name of the job-file
fbase = "EnsembleInfo"

# write commands to text file
open(fbase + ".txt", "w").write("\n".join(commands))

# job-options
sbatch = {
    "job-name": fbase,
    "out": fbase + ".out",
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 28,
    "time": "6h",
    "account": "pcsl",
}

# write SLURM file
open(fbase + ".slurm", "w").write(
    gs.scripts.plain(command=slurm.format(fbase=fbase), **sbatch)
)

# ==================================================================================================
# write copy commands
# ==================================================================================================

commands = []

for ensemble in sorted(enembles):

    outname = "EnsembleInfo.hdf5"

    dest = os.path.join(ensemble, outname)

    src = os.path.normpath(
        os.path.join("~/data/22_depinning-inertia/EnsembleInfo/build/", dest)
    )

    command = "echo {src:s};\nscp fidis:{src:s} {dest:s}".format(src=src, dest=dest)

    commands += [command]

# write copy-commands to text file
open(fbase + ".from_fidis", "w").write("\n".join(commands))
