#!/bin/bash
#SBATCH --job-name=conway_mpi
#SBATCH --time=10:00:00
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=12
#SBATCH --output=hyb.txt
#SBATCH --error=hyb_err.txt

echo "size,iterations,nodes,omp_threads,time" > ../output/hyb/results_big_grids.csv

mpicc -fopenmp -O3 -o main_hyb ../src/main_hyb.c
module load intel/python/3/2017.3.052

# for o in {2,4,8,10,12}; do
#     for t in {2,4,6}; do
#         for i in {50,100,500,1000,2000,5000}; do
#             for s in {500,1000,2000,3000}; do
#                 for j in {1..20}; do
#                     python generator.py "$s" hyb
#                     mpirun -np "$t" ./main_hyb "$s" "$i" "$o" >> ../output/hyb/results.csv
#                 done
#             done
#         done
#     done
# done
i=100

for o in {4,8,12}; do
    for t in {2,4,6}; do
        # for i in {50,100,200,500}; do
            for s in {5000,8000,12000,15000}; do
                for j in {1..10}; do
                    python generator.py "$s" hyb
                    echo "Starting with i=$i and s=$s"
                    mpirun -np "$t" ./main_hyb "$s" "$i" "$o" >> ../output/hyb/results_big_grids.csv
                    echo "Finished"
                done
            done
        # done
    done
done

exit 0
