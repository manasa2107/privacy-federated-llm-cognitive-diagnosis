#!/bin/bash
#SBATCH --job-name=baseline_cdm
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/N/lustre/project/proj-606/m_research/baselines_cdm/logs/baseline_%j.out
#SBATCH --error=/N/lustre/project/proj-606/m_research/baselines_cdm/logs/baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=boyapaya@mail.uc.edu

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load modules
module purge
module load python/3.11.11
module load cuda/12.1.1
# Set up environmentcd /N/lustre/project/proj-606/m_research/baselines_cdm

# Verify GPU
nvidia-smi

# Install required packages (only needed first time)
echo "Installing required packages..."
pip install torch torchvision --break-system-packages
pip install numpy pandas scikit-learn scipy --break-system-packages

# Run experiment
echo "Starting baseline comparison experiment..."
python run_baseline_comparison.py
#python run_neuralcd_only.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
else
    echo "Experiment failed with error code: $?"
    exit 1
fi

echo "End time: $(date)"
echo "Results saved to: /N/lustre/project/proj-606/m_research/baselines_cdm/results/"
