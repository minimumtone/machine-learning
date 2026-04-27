#!/bin/bash
#PBS -N L12_omega_sf
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=48:00:00
#PBS -j oe

# Load modules (adjust to your NIMS environment)
# module load vasp/6.4.0
# module load intel-mpi

cd $PBS_O_WORKDIR

export NPROCS=16
bash run_all.sh 2>&1 | tee run_all.log

echo "Extracting results..."
bash extract_results.sh > l12_results.csv
echo "Done."
