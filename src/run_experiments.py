import pandas as pd
import numpy as np
from federated_simple import federated_diagnosis, diagnose_student
import time

# Generate synthetic student data
def generate_synthetic_data(n_students=30, n_concepts=4):
    """Generate synthetic student response data"""
    np.random.seed(42)  # Reproducible results
    data = []
    
    concepts = ["addition", "subtraction", "multiplication", "division"]
    
    for i in range(n_students):
        # True knowledge state (ground truth)
        true_state = np.random.rand(n_concepts)
        
        # Generate responses based on knowledge (higher knowledge = more correct)
        responses = []
        for k in true_state:
            # Add some randomness
            is_correct = np.random.rand() < (k * 0.8 + 0.1)
            responses.append("correct" if is_correct else "wrong")
        
        response_string = ", ".join([f"Q{i+1}: {r}" for i, r in enumerate(responses)])
        
        data.append({
            'student_id': i,
            'responses': response_string,
            'true_state': true_state.tolist()
        })
    
    return pd.DataFrame(data), concepts

# Run experiments
def run_baseline_vs_federated(n_students=30):
    """Compare single LLM vs federated approach"""
    
    print("\n" + "="*70)
    print("EXPERIMENT: Baseline vs Federated Cognitive Diagnosis")
    print("="*70)
    
    print(f"\nGenerating test data for {n_students} students...")
    data, concepts = generate_synthetic_data(n_students=n_students)
    
    results = {
        'baseline': [],
        'federated': [],
        'true_states': []
    }
    
    start_time = time.time()
    
    for idx, row in data.iterrows():
        print(f"\n{'='*70}")
        print(f"Processing Student {idx+1}/{n_students}")
        print(f"Responses: {row['responses']}")
        print(f"{'='*70}")
        
        # Baseline: Single LLM
        print("\n[BASELINE] Single LLM diagnosis...")
        baseline_pred = diagnose_student(row['responses'], concepts, entity_id=0)
        
        # Federated: 3 LLMs averaged
        print("\n[FEDERATED] Multi-entity diagnosis...")
        federated_pred, local_preds = federated_diagnosis(row['responses'], concepts, num_entities=3)
        
        results['baseline'].append(baseline_pred)
        results['federated'].append(federated_pred)
        results['true_states'].append(row['true_state'])
        
        print(f"\nTrue state:      {[f'{x:.2f}' for x in row['true_state']]}")
        print(f"Baseline pred:   {[f'{x:.2f}' for x in baseline_pred]}")
        print(f"Federated pred:  {[f'{x:.2f}' for x in federated_pred]}")
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    true_states = np.array(results['true_states'])
    baseline_preds = np.array(results['baseline'])
    federated_preds = np.array(results['federated'])
    
    # Mean Absolute Error
    baseline_mae = np.mean(np.abs(true_states - baseline_preds))
    federated_mae = np.mean(np.abs(true_states - federated_preds))
    
    # Root Mean Squared Error
    baseline_rmse = np.sqrt(np.mean((true_states - baseline_preds) ** 2))
    federated_rmse = np.sqrt(np.mean((true_states - federated_preds) ** 2))
    
    # Per-concept MAE
    baseline_concept_mae = np.mean(np.abs(true_states - baseline_preds), axis=0)
    federated_concept_mae = np.mean(np.abs(true_states - federated_preds), axis=0)
    
    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nDataset: {n_students} students, {len(concepts)} concepts")
    print(f"Concepts: {concepts}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    print("\n--- Overall Metrics ---")
    print(f"Baseline MAE:   {baseline_mae:.4f}")
    print(f"Federated MAE:  {federated_mae:.4f}")
    print(f"Improvement:    {((baseline_mae - federated_mae) / baseline_mae * 100):.2f}%")
    
    print(f"\nBaseline RMSE:  {baseline_rmse:.4f}")
    print(f"Federated RMSE: {federated_rmse:.4f}")
    print(f"Improvement:    {((baseline_rmse - federated_rmse) / baseline_rmse * 100):.2f}%")
    
    print("\n--- Per-Concept MAE ---")
    for i, concept in enumerate(concepts):
        print(f"{concept:15} - Baseline: {baseline_concept_mae[i]:.4f}, Federated: {federated_concept_mae[i]:.4f}")
    
    print("\n" + "="*70)
    
    return {
        'baseline_mae': baseline_mae,
        'federated_mae': federated_mae,
        'baseline_rmse': baseline_rmse,
        'federated_rmse': federated_rmse,
        'improvement': (baseline_mae - federated_mae) / baseline_mae * 100
    }

if __name__ == "__main__":
    # Run with 30 students (change to 50-100 for final experiments)
    results = run_baseline_vs_federated(n_students=30)
