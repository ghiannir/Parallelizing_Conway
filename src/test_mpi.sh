#!/bin/bash

#SBATCH --job-name=test_mpi_conway
#SBATCH --output=../output/output_%j.txt
#SBATCH --error=../output/error_%j.txt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

# Load necessary modules
# module load mpi

# Compile the MPI program
mpicc -O3 -o main_mpi ../src/main_mpi.c

# Compile the sequential program
gcc -o sequential ../src/sequential.c

# Run the MPI program
mpirun -np 4 ./main_mpi 1000 50

# Run the sequential program
./sequential 1000 50

# Compare the outputs
diff ../output/mat_mpi.txt ../output/mat_seq.txt > ../output/diff_output.txt
diff ../output/cnt_mpi.txt ../output/cnt_seq.txt >> ../output/diff_output.txt
diff ../output/streak_mpi.txt ../output/streak_seq.txt >> ../output/diff_output.txt

# Check if the outputs are the same
if [ -s ../output/diff_output.txt ]; then
    echo "Outputs differ. Check ../output/diff_output.txt for details."
else
    echo "Outputs are the same."
fi