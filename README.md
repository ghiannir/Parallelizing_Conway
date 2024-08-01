# Parallelizing_Conway


This repository stores the code used for the project of HPC 2024 named: "Parallelizing Conway's game of life" using Cuda.

- Download it into hactar using:
`git clone https://github.com/ghiannir/Parallelizing_Conway.git`

- In order to run the code go inside src/ and write:
`sbatch launch.sh "dimensions" "iterations"`
substituting "dimensions" with required side of the grid and "iterations" with how many generations you want to simulate.

- It will generate a new input file and results files all stores into output/ directory
