#!/bin/bash

#SBATCH --job-name=test_mpi_conway
#SBATCH --output=../output/output_%j.txt
#SBATCH --error=../output/error_%j.txt
#SBATCH --nodes=4
#SBATCH --ntasks=12
#SBATCH --time=01:00:00

# Load necessary modules
# module load mpi
rm ../output/*_mpi.txt

# Compile the MPI program
rm main_mpi
mpicc -O3 -o main_mpi ../src/main_mpi.c

# Compile the sequential program
gcc -o sequential ../src/sequential.c

module load intel/python/3/2017.3.052
N=1000
python3 ../src/generator.py "$N"
# Run the MPI program
mpirun -np 4 ./main_mpi "$N" 50

# Run the sequential program
./sequential "$N" 50

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