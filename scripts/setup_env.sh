#!/bin/bash
#SBATCH --job-name=fed_cdm
#SBATCH --qos=proj-606
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=fed_cdm_%j.out
#SBATCH --error=fed_cdm_%j.err

# Load modules
module load python/3.11.11
module load cuda/12.1.1

# Run your federated learning code
python your_federated_script.py
