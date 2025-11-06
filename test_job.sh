#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=test_job.out

echo "Job started at $(date)"
hostname
echo "Job finished at $(date)"
