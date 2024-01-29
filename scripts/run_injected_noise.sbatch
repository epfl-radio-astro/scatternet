#!/bin/bash -l
#SBATCH --chdir /home/etolley/scatternet
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 40G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

module load gcc mvapich2 py-mpi4py py-tensorflow
export PYTHONPATH="$PWD"
source venv/bin/activate
module load gcc mvapich2 py-mpi4py py-tensorflow

python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"

srun python run_all_scenarios.py mbb 1









