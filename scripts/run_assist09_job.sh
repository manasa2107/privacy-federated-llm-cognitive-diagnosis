#!/bin/bash
#SBATCH --job-name=assist09
#SBATCH --qos=proj-606
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=assist09_%j.out
#SBATCH --error=assist09_%j.err

module load python/3.11.11

pip install --user groq openai anthropic pandas numpy

python run_experiments_assist09.py
