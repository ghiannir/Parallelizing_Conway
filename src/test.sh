#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Too few arguments, given $#, usage: sbatch launch.sh 'dimensions' 'iterations'"
    exit 1
fi

export N="$1"
export ITER="$2"
export DIS=0.5

python3 generator.py

gcc sequential.c -o test
./test 1

gcc main_omp.c -fopenmp -o main_omp 
./main_omp

diff ../output/mat_seq.txt ../output/mat_omp.txt

exit 0