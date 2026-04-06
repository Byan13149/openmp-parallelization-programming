#!/bin/bash
#SBATCH -J cholesky_example
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00

module load gcc
make IMPL=omp3
OMP_NUM_THREADS=64 ./build/cholesky_example 5000