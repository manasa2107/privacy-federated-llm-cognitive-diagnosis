"""
Baseline Comparison Experiment for ASSIST09 Dataset
Compares NeuralCD, IRT, DINA against Federated LLM Framework

Run this script to generate Table 2.5 for the ACL rebuttal
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import warnings
warnings.filterwarnings('ignore')

# Import baseline models
from neuralcd_model import NeuralCDNet, CDMDataset, train_neuralcd, predict_knowledge_states
from irt_dina_models import IRTModel, DINAModel
from torch.utils.data import DataLoader


def load_assist09_data(data_path):
    """
    Load ASSIST09 dataset
    
    Expected format:
    - student_responses.csv: student_id, question_id, correct, concepts
    - q_matrix.csv: question_id, Equations, Percentages, Integers, Conversions
    - ground_truth_knowledge.csv: student_id, Equations, Percentages, Integers, Conversions
    
    Args:
        data_path: Path to data directory
        
    Returns:
        responses_df: DataFrame with student responses
        q_matrix: (n_questions, n_concepts) Q-matrix
        ground_truth: DataFrame with ground truth knowledge states
    """
    print("Loading ASSIST09 data...", flush=True)
    
    # Load response data
    responses_file = os.path.join(data_path, 'student_responses.csv')
    if not os.path.exists(responses_file):
        # Try alternative naming
        responses_file = os.path.join(data_path, 'assist09_responses.csv')
    
    responses_df = pd.read_csv(responses_file)
    print(f"Loaded {len(responses_df)} student responses")
    
    # Load Q-matrix
    q_matrix_file = os.path.join(data_path, 'q_matrix.csv')
    if not os.path.exists(q_matrix_file):
        q_matrix_file = os.path.join(data_path, 'assist09_qmatrix.csv')
    
    q_matrix_df = pd.read_csv(q_matrix_file)
    concepts = ['Equations', 'Percentages', 'Integers', 'Conversions']
    q_matrix = q_matrix_df[concepts].values
    print(f"Loaded Q-matrix: {q_matrix.shape}")
    
    # Load or compute ground truth knowledge states
    gt_file = os.path.join(data_path, 'ground_truth_knowledge.csv')
    if os.path.exists(gt_file):
        ground_truth = pd.read_csv(gt_file)
    else:
        # Compute ground truth from response patterns
        print("Computing ground truth knowledge states from responses...")
        ground_truth = compute_ground_truth_knowledge(responses_df, q_matrix_df, concepts)
        ground_truth.to_csv(gt_file, index=False)
    
    print(f"Ground truth knowledge states: {ground_truth.shape}")
    
    return responses_df, q_matrix, ground_truth


def compute_ground_truth_knowledge(responses_df, q_matrix_df, concepts):
    """
    Compute ground truth knowledge states from response patterns
    Using proportion of correct answers per concept
    
    Args:
        responses_df: DataFrame with student_id, question_id, correct
        q_matrix_df: DataFrame with question_id and concept columns
        concepts: List of concept names
        
    Returns:
        ground_truth_df: DataFrame with student_id and concept mastery scores
    """
    student_ids = responses_df['student_id'].unique()
    n_concepts = len(concepts)
    
    knowledge_states = np.zeros((len(student_ids), n_concepts))
    
    for i, student_id in enumerate(student_ids):
        student_responses = responses_df[responses_df['student_id'] == student_id]
        
        for j, concept in enumerate(concepts):
            # Get questions requiring this concept
            concept_questions = q_matrix_df[q_matrix_df[concept] == 1]['question_id'].values
            
            # Get student's responses to these questions
            concept_responses = student_responses[
                student_responses['question_id'].isin(concept_questions)
            ]
            
            if len(concept_responses) > 0:
                # Knowledge state = proportion correct
                knowledge_states[i, j] = concept_responses['correct'].mean()
            else:
                knowledge_states[i, j] = 0.5  # Default if no responses
    
    # Create DataFrame
    ground_truth_df = pd.DataFrame(knowledge_states, columns=concepts)
    ground_truth_df.insert(0, 'student_id', student_ids)
    
    return ground_truth_df


def run_neuralcd(responses_df, q_matrix, ground_truth, n_concepts=4):
    """
    Run NeuralCD baseline
    
    Returns:
        mae: Mean Absolute Error on test set
        predictions: Predicted knowledge states
    """
    print("\n" + "="*60)
    print("Running NeuralCD Baseline")
    print("="*60)
    
    # Prepare data
    student_ids = responses_df['student_id'].values
    question_ids = responses_df['question_id'].values
    responses = responses_df['correct'].values.astype(float)
    
    # Get unique counts
    n_students = responses_df['student_id'].nunique()
    n_questions = responses_df['question_id'].nunique()
    
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
    
    model, history = train_neuralcd(
        model, 
        train_loader, 
        test_loader, 
        n_epochs=100, 
        lr=0.001, 
        device=device
    )
    
    # Predict knowledge states for all students
    all_student_ids = sorted(student_mapping.values())
    predictions = predict_knowledge_states(model, all_student_ids, device=device)
    
    # Align with ground truth
    gt_student_ids = ground_truth['student_id'].values
    gt_values = ground_truth.drop('student_id', axis=1).values
    
    # Map predictions to ground truth students
    reverse_mapping = {v: k for k, v in student_mapping.items()}
    predicted_states = np.zeros_like(gt_values)
    
    for i, gt_sid in enumerate(gt_student_ids):
        if gt_sid in student_mapping:
            mapped_id = student_mapping[gt_sid]
            predicted_states[i] = predictions[mapped_id]
        else:
            predicted_states[i] = predictions.mean(axis=0)  # Use mean for unseen students
    
    # Compute MAE
    mae = mean_absolute_error(gt_values, predicted_states)
    rmse = np.sqrt(mean_squared_error(gt_values, predicted_states))
    
    print(f"\nNeuralCD Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return mae, predicted_states, history


def run_irt(responses_df, ground_truth, n_concepts=4):
    """
    Run IRT baseline
    
    Returns:
        mae: Mean Absolute Error
        predictions: Predicted knowledge states
    """
    print("\n" + "="*60, flush=True)
    print("Running IRT Baseline", flush=True)
    print("="*60, flush=True)
    
    # Prepare data
    student_ids = responses_df['student_id'].values
    question_ids = responses_df['question_id'].values
    responses = responses_df['correct'].values
    
    n_students = responses_df['student_id'].nunique()
    n_questions = responses_df['question_id'].nunique()
    
    # Map IDs
    student_mapping = {sid: i for i, sid in enumerate(sorted(responses_df['student_id'].unique()))}
    question_mapping = {qid: i for i, qid in enumerate(sorted(responses_df['question_id'].unique()))}
    
    student_ids_mapped = np.array([student_mapping[sid] for sid in student_ids])
    question_ids_mapped = np.array([question_mapping[qid] for qid in question_ids])
    
    # Initialize and fit IRT model
    irt_model = IRTModel(n_students, n_questions)
    irt_model.fit(student_ids_mapped, question_ids_mapped, responses, max_iter=50)
    
    # Predict knowledge states
    all_student_ids = sorted(student_mapping.values())
    predictions = irt_model.predict_knowledge_states(all_student_ids, n_concepts=n_concepts)
    
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
    mae = mean_absolute_error(gt_values, predicted_states)
    rmse = np.sqrt(mean_squared_error(gt_values, predicted_states))
    
    print(f"\nIRT Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return mae, predicted_states


def run_dina(responses_df, q_matrix, ground_truth):
    """
    Run DINA baseline
    
    Returns:
        mae: Mean Absolute Error
        predictions: Predicted knowledge states
    """
    print("\n" + "="*60)
    print("Running DINA Baseline")
    print("="*60)
    
    # Prepare data
    student_ids = responses_df['student_id'].values
    question_ids = responses_df['question_id'].values
    responses = responses_df['correct'].values
    
    n_students = responses_df['student_id'].nunique()
    n_questions = responses_df['question_id'].nunique()
    n_concepts = q_matrix.shape[1]
    
    # Map IDs
    student_mapping = {sid: i for i, sid in enumerate(sorted(responses_df['student_id'].unique()))}
    question_mapping = {qid: i for i, qid in enumerate(sorted(responses_df['question_id'].unique()))}
    
    student_ids_mapped = np.array([student_mapping[sid] for sid in student_ids])
    question_ids_mapped = np.array([question_mapping[qid] for qid in question_ids])
    
    # Initialize and fit DINA model
    dina_model = DINAModel(n_students, n_questions, n_concepts, q_matrix)
    dina_model.fit(student_ids_mapped, question_ids_mapped, responses, max_iter=50)
    
    # Predict knowledge states
    all_student_ids = sorted(student_mapping.values())
    predictions = dina_model.predict_knowledge_states(all_student_ids)
    
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
    mae = mean_absolute_error(gt_values, predicted_states)
    rmse = np.sqrt(mean_squared_error(gt_values, predicted_states))
    
    print(f"\nDINA Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return mae, predicted_states


def main():
    """Main experiment runner"""
    print("="*60, flush=True)
    print("BASELINE COMPARISON EXPERIMENT - ASSIST09", flush=True)
    print("="*60, flush=True)
    
    # Set paths
    data_path = '/N/lustre/project/proj-606/m_research/baselines_cdm/data'
    results_path = '/N/lustre/project/proj-606/m_research/baselines_cdm/results'
    
    # Create results directory
    os.makedirs(results_path, exist_ok=True)
    
    # Load data
    responses_df, q_matrix, ground_truth = load_assist09_data(data_path)
    
    # Run baselines
    results = {}
    
    # IRT
    irt_mae, irt_preds = run_irt(responses_df, ground_truth, n_concepts=4)
    results['IRT'] = {'MAE': irt_mae}
    
    # DINA
    dina_mae, dina_preds = run_dina(responses_df, q_matrix, ground_truth)
    results['DINA'] = {'MAE': dina_mae}
    
    # NeuralCD
    neuralcd_mae, neuralcd_preds, history = run_neuralcd(
        responses_df, q_matrix, ground_truth, n_concepts=4
    )
    results['NeuralCD'] = {'MAE': neuralcd_mae}
    
    # Load federated LLM results (if available)
    # Expected format: {'Single LLM': 0.2410, 'Federated': 0.2068}
    llm_results_file = os.path.join(data_path, 'federated_llm_results.json')
    if os.path.exists(llm_results_file):
        with open(llm_results_file, 'r') as f:
            llm_results = json.load(f)
        results['Single LLM'] = {'MAE': llm_results.get('Single LLM', 0.2410)}
        results['Federated LLM'] = {'MAE': llm_results.get('Federated', 0.2068)}
    else:
        # Use values from paper
        results['Single LLM'] = {'MAE': 0.2410}
        results['Federated LLM'] = {'MAE': 0.2068}
    
    # Print final comparison table
    print("\n" + "="*60)
    print("FINAL RESULTS - TABLE 2.5")
    print("="*60)
    print(f"\n{'Method':<20} {'MAE':<10} {'vs Baseline':<15}")
    print("-" * 50)
    
    baseline_mae = results['Single LLM']['MAE']
    
    for method in ['IRT', 'DINA', 'NeuralCD', 'Single LLM', 'Federated LLM']:
        mae = results[method]['MAE']
        improvement = ((baseline_mae - mae) / baseline_mae) * 100
        sign = '+' if improvement > 0 else ''
        print(f"{method:<20} {mae:<10.4f} {sign}{improvement:>6.2f}%")
    
    # Save results
    results_file = os.path.join(results_path, 'baseline_comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save predictions
    np.save(os.path.join(results_path, 'irt_predictions.npy'), irt_preds)
    np.save(os.path.join(results_path, 'dina_predictions.npy'), dina_preds)
    np.save(os.path.join(results_path, 'neuralcd_predictions.npy'), neuralcd_preds)
    
    print("\nExperiment complete!")
    
    return results


if __name__ == "__main__":
    results = main()
