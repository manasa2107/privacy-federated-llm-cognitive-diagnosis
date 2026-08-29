"""
Local Differential Privacy Implementation
Noise is added at each LLM entity BEFORE aggregation
"""

import numpy as np
from simple_llm import get_llm_prediction

def add_local_dp_noise(prediction, epsilon, sensitivity=1.0):
    """
    Add Laplace noise at entity level (Local DP)
    
    Args:
        prediction: Knowledge state vector [0,1]^K
        epsilon: Privacy budget for this entity
        sensitivity: Global sensitivity (default=1.0 for [0,1] predictions)
    """
    K = len(prediction)
    noise = np.random.laplace(0, sensitivity/epsilon, size=K)
    noisy_pred = np.array(prediction) + noise
    return np.clip(noisy_pred, 0, 1).tolist()

def federated_diagnosis_local_dp(student_responses, concepts, epsilon=2.0):
    """
    Federated learning with LOCAL differential privacy
    Each entity adds noise BEFORE sending to aggregator
    """
    print(f"\n{'='*70}")
    print(f"LOCAL DP: Each LLM adds noise before sharing (ε={epsilon})")
    print(f"{'='*70}\n")
    
    llm_models = ['llama', 'gpt4', 'claude']
    noisy_predictions = []
    
    # Step 1: Each LLM makes prediction and adds noise LOCALLY
    for i, model in enumerate(llm_models):
        # Get clean prediction
        clean_pred = get_llm_prediction(student_responses, concepts, model)
        print(f"Entity {i} ({model})")
        print(f"  Clean prediction: {[f'{x:.3f}' for x in clean_pred]}")
        
        # Add noise BEFORE sharing (Local DP)
        noisy_pred = add_local_dp_noise(clean_pred, epsilon)
        print(f"  Noisy prediction: {[f'{x:.3f}' for x in noisy_pred]}")
        print(f"  Noise magnitude:  {[f'{abs(c-n):.3f}' for c, n in zip(clean_pred, noisy_pred)]}\n")
        
        noisy_predictions.append(noisy_pred)
    
    # Step 2: Aggregator receives ONLY noisy predictions
    aggregated = np.mean(noisy_predictions, axis=0).tolist()
    
    print(f"{'='*70}")
    print(f"Aggregated prediction (from noisy inputs): {[f'{x:.3f}' for x in aggregated]}")
    print(f"{'='*70}\n")
    
    return aggregated, noisy_predictions

# Example usage
if __name__ == "__main__":
    responses = "Q1:correct, Q2:wrong, Q3:correct, Q4:correct"
    concepts = ["algebra", "geometry", "probability", "statistics"]
    
    # Compare Central DP vs Local DP
    print("\n" + "="*70)
    print("COMPARING CENTRAL DP vs LOCAL DP")
    print("="*70)
    
    # Local DP (ε=2.0)
    local_dp_result, _ = federated_diagnosis_local_dp(responses, concepts, epsilon=2.0)
    
    print("\nLocal DP Result:", local_dp_result)
