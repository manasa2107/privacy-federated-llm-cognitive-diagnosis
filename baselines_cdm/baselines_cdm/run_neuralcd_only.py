"""
NeuralCD-only baseline comparison for ASSIST09
Runs only NeuralCD against fixed LLM baseline values from paper
"""

import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import torch

from neuralcd_model import NeuralCDNet, CDMDataset, train_neuralcd, predict_knowledge_states
from torch.utils.data import DataLoader

print("="*60)
print("NeuralCD Baseline Comparison - ASSIST09")
print("="*60)

# Paths
data_path = '/N/lustre/project/proj-606/m_research/baselines_cdm/data'
results_path = '/N/lustre/project/proj-606/m_research/baselines_cdm/results'
os.makedirs(results_path, exist_ok=True)

# Load data
print("\nLoading ASSIST09 data...")
responses_df = pd.read_csv(f'{data_path}/student_responses.csv')
q_matrix_df = pd.read_csv(f'{data_path}/q_matrix.csv')
ground_truth = pd.read_csv(f'{data_path}/ground_truth_knowledge.csv')

print(f"Loaded {len(responses_df)} student responses")
print(f"Students: {responses_df['student_id'].nunique()}")
print(f"Questions: {responses_df['question_id'].nunique()}")

# Prepare NeuralCD data
student_ids = responses_df['student_id'].values
question_ids = responses_df['question_id'].values
responses = responses_df['correct'].values.astype(float)

n_students = responses_df['student_id'].nunique()
n_questions = responses_df['question_id'].nunique()
n_concepts = 4

# Map IDs to continuous range
student_mapping = {sid: i for i, sid in enumerate(sorted(responses_df['student_id'].unique()))}
question_mapping = {qid: i for i, qid in enumerate(sorted(responses_df['question_id'].unique()))}

student_ids_mapped = np.array([student_mapping[sid] for sid in student_ids])
question_ids_mapped = np.array([question_mapping[qid] for qid in question_ids])

# Train-test split
train_indices, test_indices = train_test_split(
    np.arange(len(student_ids_mapped)), 
    test_size=0.2, 
    random_state=42
)

# Create datasets
train_dataset = CDMDataset(
    student_ids_mapped[train_indices],
    question_ids_mapped[train_indices],
    responses[train_indices]
)
test_dataset = CDMDataset(
    student_ids_mapped[test_indices],
    question_ids_mapped[test_indices],
    responses[test_indices]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
print("\nInitializing NeuralCD model...")
model = NeuralCDNet(
    n_students=n_students,
    n_questions=n_questions,
    n_concepts=n_concepts,
    student_dim=64,
    question_dim=64,
    concept_dim=64,
    hidden_dim=128
)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("\nTraining NeuralCD...")
model, history = train_neuralcd(
    model, 
    train_loader, 
    test_loader, 
    n_epochs=100, 
    lr=0.001, 
    device=device
)

# Predict knowledge states
all_student_ids = sorted(student_mapping.values())
predictions = predict_knowledge_states(model, all_student_ids, device=device)

# Align with ground truth
gt_student_ids = ground_truth['student_id'].values
gt_values = ground_truth.drop('student_id', axis=1).values

reverse_mapping = {v: k for k, v in student_mapping.items()}
predicted_states = np.zeros_like(gt_values)

for i, gt_sid in enumerate(gt_student_ids):
    if gt_sid in student_mapping:
        mapped_id = student_mapping[gt_sid]
        predicted_states[i] = predictions[mapped_id]
    else:
        predicted_states[i] = predictions.mean(axis=0)

# Compute MAE
neuralcd_mae = mean_absolute_error(gt_values, predicted_states)
neuralcd_rmse = np.sqrt(mean_squared_error(gt_values, predicted_states))

print(f"\nNeuralCD Results:")
print(f"MAE:  {neuralcd_mae:.4f}")
print(f"RMSE: {neuralcd_rmse:.4f}")

# Fixed baseline values from paper
baseline_mae = 0.2410
federated_mae = 0.2068

# Calculate improvements
neuralcd_improvement = ((baseline_mae - neuralcd_mae) / baseline_mae) * 100
federated_improvement = ((baseline_mae - federated_mae) / baseline_mae) * 100

# Final results
results = {
    "NeuralCD": {"MAE": float(neuralcd_mae)},
    "Single LLM": {"MAE": baseline_mae},
    "Federated LLM": {"MAE": federated_mae}
}

# Save results
results_file = os.path.join(results_path, 'neuralcd_comparison_results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Print final comparison table
print("\n" + "="*60)
print("FINAL RESULTS - NeuralCD vs Federated LLM")
print("="*60)
print(f"\n{'Method':<20} {'MAE':<10} {'vs Baseline':<15}")
print("-" * 50)
print(f"{'NeuralCD':<20} {neuralcd_mae:<10.4f} {neuralcd_improvement:>6.2f}%")
print(f"{'Single LLM':<20} {baseline_mae:<10.4f} {'  0.00%':>15}")
print(f"{'Federated LLM':<20} {federated_mae:<10.4f} {federated_improvement:>6.2f}%")

print(f"\nResults saved to: {results_file}")
print("\nComparison complete!")
