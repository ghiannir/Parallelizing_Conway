#!/bin/bash
#SBATCH --job-name=conway_mpi
#SBATCH --time=04:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --output=mpi.txt
#SBATCH --error=mpi_err.txt

echo "size,iterations,n_threads,time" > ../output/mpi/results.csv

mpicc -O3 -o main_mpi ../src/main_mpi.c
module load intel/python/3/2017.3.052

for t in {2,4,8,10,12}; do
    for i in {50,100,500,1000,2000,5000}; do
        for s in {500,1000,2000,5000,8000,10000}; do
            for j in {1..20}; do
                python generator.py "$s"
                mpirun -np "$t" ./main_mpi "$s" "$i" >> ../output/mpi/results.csv
            done
        done
    done
done

exit 0
