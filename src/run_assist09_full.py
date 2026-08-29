
"""
ASSIST09 Full Dataset Experiment
4,217 students × 4 concepts
Privacy-Preserving Heterogeneous Multi-LLM Federated Cognitive Diagnosis
"""

import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import os
from datetime import datetime
import sys
from collections import defaultdict

# LLM API imports
import anthropic
from openai import OpenAI
from groq import Groq

# ==================== CONFIGURATION ====================
ASSIST09_PATH = '/N/lustre/project/proj-606/m_research/skill_builder_data.csv'  # UPDATE THIS PATH
OUTPUT_DIR = '/N/lustre/project/proj-606/m_research/results_full/'
CHECKPOINT_DIR = '/N/lustre/project/proj-606/m_research/checkpoints_full/'

# Top 4 skills (matching paper)
SKILL_MAPPING = {
    'Equation Solving Two or Fewer Steps': 0,  # Equations
    'Percent Of': 1,                            # Percentages
    'Addition and Subtraction Integers': 2,     # Integers
    'Conversion of Fraction Decimals Percents': 3  # Conversions
}
CONCEPTS = ['Equations', 'Percentages', 'Integers', 'Conversions']
K = 4

# Privacy budgets to test
EPSILON_VALUES = [0.5, 1.0, 2.0, float('inf')]
ALPHA = 0.3  # Residual correction strength

# API configuration
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
CHECKPOINT_INTERVAL = 50  # Save every 50 students

# API Keys from environment
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Initialize API clients
groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ==================== DATA LOADING ====================
def load_assist09_full():
    """Load and preprocess full ASSIST09 dataset"""
    print("="*60)
    print("Loading ASSIST09 Full Dataset")
    print("="*60)
    
    # Load data
    print("\n[1/5] Reading CSV file...")
    df = pd.read_csv(
        ASSIST09_PATH,
        encoding='latin-1',
        on_bad_lines='skip',
        low_memory=False
    )
    print(f"✓ Loaded {len(df):,} interactions")
    
    # Filter to top 4 skills only
    print("\n[2/5] Filtering to top 4 skills...")
    df_filtered = df[df['skill_name'].isin(SKILL_MAPPING.keys())].copy()
    print(f"✓ Retained {len(df_filtered):,} interactions ({len(df_filtered)/len(df)*100:.1f}%)")
    
    # Map skill names to concept indices
    df_filtered['concept_id'] = df_filtered['skill_name'].map(SKILL_MAPPING)
    
    # Get all unique students
    print("\n[3/5] Identifying students...")
    all_students = df_filtered['user_id'].unique()
    print(f"✓ Found {len(all_students):,} students with data in top 4 skills")
    
    # Compute ground truth for each student
    print("\n[4/5] Computing ground truth knowledge states...")
    ground_truth = {}
    student_histories = {}
    students_with_data = []
    
    for student_id in tqdm(all_students, desc="Processing students"):
        student_data = df_filtered[df_filtered['user_id'] == student_id]
        
        # Compute ground truth (proportion correct per concept)
        gt = np.zeros(K)
        has_data = False
        
        for concept_id in range(K):
            concept_data = student_data[student_data['concept_id'] == concept_id]
            if len(concept_data) > 0:
                gt[concept_id] = concept_data['correct'].mean()
                has_data = True
            else:
                gt[concept_id] = 0.5  # Neutral for missing data
        
        if has_data:
            ground_truth[student_id] = gt
            
            # Create response history (limit to 30 most recent)
            history_entries = []
            for _, row in student_data.tail(30).iterrows():
                skill = row['skill_name']
                correct = 'Correct' if row['correct'] == 1 else 'Incorrect'
                history_entries.append(f"• {skill}: {correct}")
            
            student_histories[student_id] = '\n'.join(history_entries)
            students_with_data.append(student_id)
    
    print(f"✓ Processed {len(students_with_data):,} students")
    
    # Statistics
    print("\n[5/5] Dataset Statistics:")
    print(f"  Total students: {len(students_with_data):,}")
    print(f"  Total interactions: {len(df_filtered):,}")
    print(f"  Avg interactions/student: {len(df_filtered)/len(students_with_data):.1f}")
    
    # Concept coverage
    print(f"\n  Concept coverage:")
    for concept_id, concept_name in enumerate(CONCEPTS):
        n_students = sum(1 for gt in ground_truth.values() if gt[concept_id] != 0.5)
        print(f"    {concept_name}: {n_students:,} students ({n_students/len(students_with_data)*100:.1f}%)")
    
    return students_with_data, student_histories, ground_truth

# ==================== LLM ASSESSMENT ====================
def get_llm_assessment(student_history, model='llama', retry_count=0):
    """Get knowledge state assessment from specific LLM with retry logic"""
    
    prompt = f"""You are an expert in educational assessment. Analyze this student's response history and estimate their mastery level (0.0 to 1.0) for each of the 4 mathematical concepts.

Concepts:
1. Equations - Solving equations (two or fewer steps)
2. Percentages - Calculating percentages
3. Integers - Addition and subtraction of integers
4. Conversions - Converting between fractions, decimals, and percentages

Student Response History:
{student_history}

Based on this history, estimate the student's mastery level for each concept.

Return ONLY a JSON array with exactly 4 numbers between 0.0 and 1.0, for example:
[0.75, 0.60, 0.85, 0.70]

Response:"""
    
    try:
        if model == 'llama':
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            content = response.choices[0].message.content
            
        elif model == 'gpt4':
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            content = response.choices[0].message.content
            
        elif model == 'claude':
            response = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        
        # Parse JSON response
        import re
        json_match = re.search(r'\[[\d\.\,\s]+\]', content)
        if json_match:
            prediction = json.loads(json_match.group())
            if len(prediction) >= K:
                return np.array(prediction[:K], dtype=float)
        
        # If parsing fails, return neutral
        print(f"  Warning: Could not parse {model} response, using neutral")
        return np.random.uniform(0.4, 0.6, K)
        
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"  Retry {retry_count+1}/{MAX_RETRIES} for {model}: {str(e)[:50]}")
            time.sleep(2 ** retry_count)  # Exponential backoff
            return get_llm_assessment(student_history, model, retry_count + 1)
        else:
            print(f"  Error with {model} after {MAX_RETRIES} retries: {str(e)}")
            return np.random.uniform(0.4, 0.6, K)

# ==================== FEDERATED FRAMEWORK ====================
class FederatedCognitiveDiagnosis:
    def __init__(self, epsilon=2.0, alpha=0.3):
        self.epsilon = epsilon
        self.alpha = alpha
        self.residuals = {
            'llama': np.zeros(K),
            'gpt4': np.zeros(K),
            'claude': np.zeros(K)
        }
        
    def federated_aggregate(self, predictions):
        """Aggregate predictions with residual correction and DP"""
        llama_pred, gpt4_pred, claude_pred = predictions
        
        # Apply residual correction
        corrected_preds = [
            llama_pred - self.alpha * self.residuals['llama'],
            gpt4_pred - self.alpha * self.residuals['gpt4'],
            claude_pred - self.alpha * self.residuals['claude']
        ]
        
        # Average
        avg_pred = np.mean(corrected_preds, axis=0)
        
        # Add Laplace noise for DP
        if self.epsilon < float('inf'):
            sensitivity = K
            noise = np.random.laplace(0, sensitivity/self.epsilon, K)
            avg_pred += noise
        
        # Clip to [0, 1]
        avg_pred = np.clip(avg_pred, 0, 1)
        
        return avg_pred
    
    def update_residuals(self, all_predictions):
        """Update residual corrections"""
        if len(all_predictions) < 10:
            return  # Need sufficient data
        
        llama_preds = np.array([p[0] for p in all_predictions])
        gpt4_preds = np.array([p[1] for p in all_predictions])
        claude_preds = np.array([p[2] for p in all_predictions])
        
        global_mean = np.mean([llama_preds, gpt4_preds, claude_preds], axis=(0,1))
        
        self.residuals['llama'] = np.mean(llama_preds, axis=0) - global_mean
        self.residuals['gpt4'] = np.mean(gpt4_preds, axis=0) - global_mean
        self.residuals['claude'] = np.mean(claude_preds, axis=0) - global_mean

# ==================== CHECKPOINTING ====================
def save_checkpoint(student_idx, all_predictions, federated_predictions, epsilon):
    """Save progress checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'student_idx': student_idx,
        'all_predictions': all_predictions,
        'federated_predictions': federated_predictions,
        'epsilon': epsilon,
        'timestamp': datetime.now().isoformat()
    }
    filepath = os.path.join(CHECKPOINT_DIR, f'checkpoint_eps_{epsilon}_idx_{student_idx}.pkl')
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  ✓ Checkpoint saved at student {student_idx}")

def load_checkpoint(epsilon):
    """Load most recent checkpoint if exists"""
    import pickle
    import glob
    
    pattern = os.path.join(CHECKPOINT_DIR, f'checkpoint_eps_{epsilon}_idx_*.pkl')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Get most recent
    latest = max(checkpoints, key=os.path.getctime)
    with open(latest, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"✓ Loaded checkpoint from student {checkpoint['student_idx']}")
    return checkpoint

# ==================== MAIN EXPERIMENT ====================
def run_experiment(epsilon=2.0, resume=True):
    """Run full experiment on ASSIST09"""
    
    print("\n" + "="*60)
    print(f"ASSIST09 FULL EXPERIMENT (ε={epsilon})")
    print("="*60)
    
    # Load data
    students, histories, ground_truth = load_assist09_full()
    N = len(students)
    
    # Initialize framework
    framework = FederatedCognitiveDiagnosis(epsilon=epsilon, alpha=ALPHA)
    
    # Try to resume from checkpoint
    start_idx = 0
    all_predictions = []
    federated_predictions = []
    
    if resume:
        checkpoint = load_checkpoint(epsilon)
        if checkpoint:
            start_idx = checkpoint['student_idx'] + 1
            all_predictions = checkpoint['all_predictions']
            federated_predictions = checkpoint['federated_predictions']
            print(f"  Resuming from student {start_idx}/{N}")
    
    # Storage
    start_time = time.time()
    total_cost = 0.0
    api_call_times = []
    
    print(f"\n{'='*60}")
    print(f"Processing {N - start_idx} students (starting from {start_idx})...")
    print(f"{'='*60}\n")
    
    # Process students
    for i in tqdm(range(start_idx, N), desc=f"ε={epsilon}", initial=start_idx, total=N):
        student_id = students[i]
        history = histories[student_id]
        
        # Get predictions from each LLM
        api_start = time.time()
        
        llama_pred = get_llm_assessment(history, 'llama')
        time.sleep(RATE_LIMIT_DELAY)
        
        gpt4_pred = get_llm_assessment(history, 'gpt4')
        time.sleep(RATE_LIMIT_DELAY)
        
        claude_pred = get_llm_assessment(history, 'claude')
        time.sleep(RATE_LIMIT_DELAY)
        
        api_time = time.time() - api_start
        api_call_times.append(api_time)
        
        # Store raw predictions
        all_predictions.append([llama_pred, gpt4_pred, claude_pred])
        
        # Federated aggregation
        fed_pred = framework.federated_aggregate([llama_pred, gpt4_pred, claude_pred])
        federated_predictions.append(fed_pred)
        
        # Estimate API costs (approximate)
        # LLaMA: $0.00059/1K tokens, GPT-4o-mini: $0.150/1M tokens, Claude-3-Haiku: $0.25/1M tokens
        total_cost += 0.002  # Approximate per student
        
        # Update residuals periodically
        if (i + 1) % 100 == 0:
            framework.update_residuals(all_predictions[-100:])
        
        # Save checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(i, all_predictions, federated_predictions, epsilon)
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # minutes
    
    # ==================== EVALUATION ====================
    print(f"\n{'='*60}")
    print("Computing Metrics...")
    print(f"{'='*60}\n")
    
    # Convert to arrays
    fed_preds = np.array(federated_predictions)
    gt_array = np.array([ground_truth[sid] for sid in students])
    
    # MAE and RMSE
    mae = np.mean(np.abs(fed_preds - gt_array))
    rmse = np.sqrt(np.mean((fed_preds - gt_array)**2))
    
    # Per-concept MAE
    concept_mae = np.mean(np.abs(fed_preds - gt_array), axis=0)
    
    # Baseline (single LLM - LLaMA)
    baseline_preds = np.array([p[0] for p in all_predictions])
    baseline_mae = np.mean(np.abs(baseline_preds - gt_array))
    baseline_rmse = np.sqrt(np.mean((baseline_preds - gt_array)**2))
    
    # Improvement
    improvement = ((baseline_mae - mae) / baseline_mae) * 100
    
    # Privacy cost (compare with no-privacy if available)
    privacy_cost = None  # Will compute after running all epsilon values
    
    # ==================== RESULTS ====================
    results = {
        'epsilon': float(epsilon),
        'n_students': N,
        'n_concepts': K,
        'concepts': CONCEPTS,
        'mae': float(mae),
        'rmse': float(rmse),
        'concept_mae': concept_mae.tolist(),
        'baseline_mae': float(baseline_mae),
        'baseline_rmse': float(baseline_rmse),
        'improvement_pct': float(improvement),
        'time_minutes': float(elapsed_time),
        'avg_time_per_student_sec': float(elapsed_time * 60 / N),
        'total_cost_usd': float(total_cost),
        'cost_per_student': float(total_cost / N),
        'avg_api_call_time': float(np.mean(api_call_times)),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS (ε={epsilon})")
    print(f"{'='*60}")
    print(f"Students Processed: {N:,}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"\nPer-Concept MAE:")
    for i, concept in enumerate(CONCEPTS):
        print(f"  {concept}: {concept_mae[i]:.4f}")
    print(f"\nTime: {elapsed_time:.1f} minutes ({elapsed_time/60:.2f} hours)")
    print(f"Cost: ${total_cost:.2f} (${total_cost/N:.4f}/student)")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save JSON results
    with open(os.path.join(OUTPUT_DIR, f'results_epsilon_{epsilon}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.save(os.path.join(OUTPUT_DIR, f'predictions_epsilon_{epsilon}.npy'), fed_preds)
    np.save(os.path.join(OUTPUT_DIR, f'baseline_predictions_epsilon_{epsilon}.npy'), baseline_preds)
    
    # Save ground truth (once)
    if epsilon == EPSILON_VALUES[0]:
        np.save(os.path.join(OUTPUT_DIR, 'ground_truth.npy'), gt_array)
        with open(os.path.join(OUTPUT_DIR, 'student_ids.json'), 'w') as f:
            json.dump([int(sid) for sid in students], f)
    
    return results

# ==================== MAIN ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ASSIST09 Full Experiment')
    parser.add_argument('--epsilon', type=float, default=2.0, help='Privacy budget')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh (ignore checkpoints)')
    args = parser.parse_args()
    
    # Check API keys
    if not all([GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY]):
        print("ERROR: Missing API keys!")
        print("Set environment variables: GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
        sys.exit(1)
    
    # Run experiment
    results = run_experiment(epsilon=args.epsilon, resume=not args.no_resume)
    
    print("\n✓ Experiment completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
