#!/bin/bash
#SBATCH --job-name build_E5v4
#SBATCH --out build_E5v4.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --constraint E5v4
#SBATCH --account pcsl
#SBATCH --partition build

# print jobid
echo "SLURM_JOBID = ${SLURM_JOBID}"
echo ""

# change current directory to the location of the sbatch command
cd "${SLURM_SUBMIT_DIR}"

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

if [[ "${SYS_TYPE}" == *E5v4* ]]; then
    conda activate code_velocity_E5v4
elif [[ "${SYS_TYPE}" == *s6g1* ]]; then
    conda activate code_velocity_s6g1
elif [[ "${SYS_TYPE}" == *S6g1* ]]; then
    conda activate code_velocity_s6g1
else
    echo "Unknown SYS_TYPE ${SYS_TYPE}"
    exit 1
fi


cd "${SLURM_TMPDIR}"

git clone git@github.com:tdegeus/FrictionQPotFEM.git
cd FrictionQPotFEM
export CMAKE_ARGS="-DUSE_SIMD=1"
python -m pip install -vv
