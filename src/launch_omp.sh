#!/bin/bash
#SBATCH --job-name=conway_omp
#SBATCH --partition=cudatemp
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=omp.txt

if [ "$#" -lt 3 ]; then
    echo "Too few arguments, given $#, usage: sbatch launch.sh 'dimensions' 'iterations' 'distribution'"
    exit 1
fi

export N="$1"
export ITER="$2"

echo "size,iterations,n_threads,time" > ../output/open_mp/results.csv

gcc -fopenmp -O3 -o main_omp main_omp.c 

for t in {1,2,4,8,10,12}; do
    for i in {50,100,500,1000,2000,5000}; do
        for s in {500,1000,2000,5000,8000,10000}; do
            for i in {1..100}; do
                ./main_omp "$s" "$i" "$t" ../output/open_mp/results.csv
            done
        done
    done
done

exit 0
