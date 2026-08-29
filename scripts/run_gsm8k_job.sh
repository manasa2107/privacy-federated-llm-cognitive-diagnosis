#!/bin/bash
#SBATCH --job-name=gsm8k
#SBATCH --qos=proj-606
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=gsm8k_%j.out
#SBATCH --error=gsm8k_%j.err

module load python/3.11.11

cd /N/lustre/project/proj-606/m_research

python run_experiments_gsm8k.py
