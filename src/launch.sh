#!/bin/bash
#SBATCH --job-name=conway
#SBATCH --partition=cudatemp
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1

if [ "$#" -lt 3 ]; then
    echo "Too few arguments, given $#, usage: sbatch launch.sh 'dimensions' 'iterations' 'distribution'"
    exit 1
fi

export N="$1"
export ITER="$2"
export DIST="../output/$3-dist.csv"
export DIS="$3"



echo "Loading CUDA and compiling"

module load nvidia/cudasdk/10.1



# nvcc main_v3.cu -Wno-deprecated-gpu-targets -lcudadevrt -arch=sm_35 -rdc=true -o  main
# ./main

# Setting the env variable that specifies how many processors are available
# export N_PROC=$(nproc --all)


nvcc main_v2.cu -o main

echo "Generating matrix"

for i in {1..100}
do
    module load intel/python/3/2017.3.052
    python generator.py

    echo "Running CUDA program $i"
    ./main $SLURM_JOB_ID
done


exit 0
