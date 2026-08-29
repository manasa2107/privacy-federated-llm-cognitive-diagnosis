#!/bin/bash
#SBATCH --job-name=uci_student
#SBATCH --qos=proj-606
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=uci_student_%j.out
#SBATCH --error=uci_student_%j.err

module load python/3.11.11

cd /N/lustre/project/proj-606/m_research

python run_experiments_uci_student.py
