#!/bin/bash
#SBATCH --job-name=conway
#SBATCH --partition=cudatemp
#SBATCH --time=00:00:30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1

if [ "$#" -lt 2 ]; then
    echo "Too few arguments, given $#, usage: sbatch launch.sh 'dimensions' 'iterations'"
    exit 1
fi

export N="$1"
export ITER="$2"

echo "Generating matrix"

module load intel/python/3/2017.3.052
python generator.py

echo "Loading CUDA and compiling"

module load nvidia/cudasdk/10.1


# Setting the env variable that specifies how many processors are available
# export N_PROC=$(nproc --all)


nvcc main_v2.cu -o main
echo "Running CUDA program"
./main $SLURM_JOB_ID


gcc -fopenmp main_v4.c -o main
echo "Running OpenMP program"
./main $SLURM_JOB_ID


gcc sequential.c -o test
echo "Running sequential program"
./test $SLURM_JOB_ID

# echo "Showing diff between parallel and sequential:"

# diff ../output/mat.txt ../output/mat_seq.txt

exit 0
