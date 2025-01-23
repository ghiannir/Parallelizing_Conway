#!/bin/bash
#SBATCH --job-name=conway
#SBATCH --partition=cuda
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:2


echo "Generating matrix"

module load intel/python/3/2017.3.052

echo "Loading CUDA and compiling"

module load nvidia/cudasdk/10.1

nvcc -O3 -Xcompiler="-fopenmp" /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/main_v2.cu -o main_cuda
nvcc -O3 -Xcompiler="-fopenmp" /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/main_v2_multi_kernel.cu -o main_cuda_multi

for i in {100..1000..100}
do

    for j in {100..500..100}
    do
        export N="$i"
        export ITER="$j"
        python /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/generator.py
        
        ./main_cuda $SLURM_JOB_ID

        ./main_cuda_multi $SLURM_JOB_ID
    done

done





exit 0
