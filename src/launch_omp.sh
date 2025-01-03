#!/bin/bash
#SBATCH --job-name=conway_omp
#SBATCH --partition=cudatemp
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=omp.txt
#SBATCH --error=omp_err.txt

# if [ "$#" -lt 3 ]; then
#     echo "Too few arguments, given $#, usage: sbatch launch.sh 'dimensions' 'iterations' 'distribution'"
#     exit 1
# fi

# export N="$1"
# export ITER="$2"

echo "size,iterations,n_threads,time" > ../output/open_mp/results_big_grids.csv

gcc -fopenmp -O3 -o main_omp main_omp.c
module load intel/python/3/2017.3.052

i=100

for t in {1,2,4,8,10,12}; do
    # for i in {50,100,200,500}; do
        for s in {5000,8000,12000,15000}; do
            for j in {1..10}; do
                python generator.py "$s" omp
                ./main_omp "$s" "$i" "$t" ../output/open_mp/results_big_grids.csv
            done
        done
    # done
done

exit 0
