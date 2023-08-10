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
#SBATCH --array=0-2

module load gcc mvapich2 py-mpi4py py-tensorflow
export PYTHONPATH="$PWD"
source venv/bin/activate
module load gcc mvapich2 py-mpi4py py-tensorflow

python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"

case $SLURM_ARRAY_TASK_ID in

  0)
    srun python run_all_rf.py mbb 1
    ;;

  1)
    srun python run_all_rf.py mbb
    ;;

  2)
    srun python run_all_rf.py mb 1
    ;;

  3)
    srun python run_all_rf.py minst
    ;;

  4)
    srun python run_all_rf.py galaxy
    ;;

  *)
    srun python run_all_rf.py galaxy 1
    ;;
esac








