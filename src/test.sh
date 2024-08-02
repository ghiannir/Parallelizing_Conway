#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Too few arguments, given $#, usage: sbatch launch.sh 'dimensions' 'iterations'"
    exit 1
fi

export N="$1"
export ITER="$2"

python3 generator.py

gcc sequential.c -o test
./test

exit 0