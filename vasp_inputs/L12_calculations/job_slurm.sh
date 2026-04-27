#!/bin/bash
#SBATCH -J L12_omega_sf
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 48:00:00
#SBATCH -o L12_%j.out

# Load modules (adjust to your NIMS environment)
# module load vasp/6.4.0
# module load intel-mpi

cd $SLURM_SUBMIT_DIR

export NPROCS=16
bash run_all.sh 2>&1 | tee run_all.log

echo "Extracting results..."
bash extract_results.sh > l12_results.csv
echo "Done."
