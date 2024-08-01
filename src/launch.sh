#!/bin/bash
#SBATCH --job-name=conway
#SBATCH --partition=cudatemp
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=conway.txt
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1

if [ "$#" -lt 2 ]; then
    echo "Too few arguments, given $#, usage: sbatch launch.sh 'dimensions' 'iterations'"
    exit 1
fi

export N="$1"
export ITER="$2"

module load nvidia/cudasdk/10.1

nvcc main.cu -o main
./main

exit 0