#!/bin/bash
#SBATCH --job-name=conway
#SBATCH --partition=cudatemp
#SBATCH --time=00:00:05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=conway.txt
#SBATCH --mem-per-cpu=10M
#SBATCH --gres=gpu:1

module load nvidia/cudasdk/10.1

nvcc main.cu -o main
./main