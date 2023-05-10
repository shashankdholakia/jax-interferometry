#!/bin/bash -l
#
#SBATCH --job-name=fitELR_PAVO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G  # memory (MB)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.dholakia@uq.edu.au

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate jaxenv

module load jax
cd /data/uqsdhola/Interferometry

python -m core.elr_fit
