#!/bin/bash
#SBATCH --job-name=privacy_exp
#SBATCH --qos=proj-606
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=privacy_%j.out
#SBATCH --error=privacy_%j.err

module load python/3.11.11

pip install --user scipy

python run_experiments_privacy.py
