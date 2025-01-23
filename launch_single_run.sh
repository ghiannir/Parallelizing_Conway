#!/bin/bash
#SBATCH --job-name=conway
#SBATCH --partition=cuda
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:2

if [ "$#" -lt 2 ]; then
    echo "Too few arguments, given $#, usage: sbatch launch.sh 'dimensions' 'iterations'"
    exit 1
fi

export N="$1"
export ITER="$2"

echo "Generating matrix"

module load intel/python/3/2017.3.052
python /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/generator.py

echo "Loading CUDA and compiling"

module load nvidia/cudasdk/10.1




## TO LAUNCH CUDA IN DEBUG MODE
# nvcc -O3 -lineinfo -Xcompiler="-fopenmp" /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/main_v2.cu -o main_cuda
# echo "Running CUDA program"
# cuda-memcheck ./main_cuda $SLURM_JOB_ID


nvcc -O3 -Xcompiler="-fopenmp" /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/main_v2.cu -o main_cuda
echo "Running CUDA program"
./main_cuda $SLURM_JOB_ID



nvcc -O3 -Xcompiler="-fopenmp" /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/main_v2_multi_kernel.cu -o main_cuda_multi
echo "Running CUDA program with multi GPU"
./main_cuda_multi $SLURM_JOB_ID



# gcc -fopenmp /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/main_v4.c -o main_openmp
# echo "Running OpenMP program"
# ./main_openmp $SLURM_JOB_ID



gcc /home/hpc_group_04/Drogato/project/Parallelizing_Conway/src/sequential.c -o main_seq
echo "Running sequential program"
./main_seq $SLURM_JOB_ID

# echo "Showing diff between parallel and sequential:"

# diff /home/hpc_group_04/Drogato/project/Parallelizing_Conway/output/mat.txt /home/hpc_group_04/Drogato/project/Parallelizing_Conway/output/mat_seq.txt

exit 0
