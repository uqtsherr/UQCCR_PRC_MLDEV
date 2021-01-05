#!/bin/bash
#SBATCH -N 3
#SBATCH --job-name=tim_run_tf2_script
#SBATCH -n 3
#SBATCH -c 1
#SBATCH --mem=50000
#SBATCH -o tensor_out.txt
#SBATCH -e tensor_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load gnu7
module load cuda/11.1.1
module load anaconda
module load mvapich2
module load pmix/2.2.2
#source activate /opt/ohpc/pub/apps/tensorflow_2.2

srun -n2 --mpi=pmi2 python3.6 benchmarks/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_gpus=2 --model resnet50 --batch_size 128
