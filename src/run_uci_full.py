"""
UCI Student Performance Full Dataset Experiment
649 students × 5 concepts
Privacy-Preserving Heterogeneous Multi-LLM Federated Cognitive Diagnosis
"""

import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime
from collections import defaultdict
import argparse

# LLM API imports
import anthropic
from openai import OpenAI
from groq import Groq

# ==================== CONFIGURATION ====================
UCI_MAT_PATH = 'student-mat.csv'
UCI_POR_PATH = 'student-por.csv'
OUTPUT_DIR = './results_full/uci/'
CHECKPOINT_DIR = './checkpoints_full/uci/'

# 5 concepts for UCI
CONCEPTS = ['Study Habits', 'Family Support', 'School Engagement', 'Social Factors', 'Academic Foundation']
K = 5

ALPHA = 0.3  # Residual correction strength

# API configuration
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 2.0  # 1 second between API calls
CHECKPOINT_INTERVAL = 50

# API Keys
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==================== DATA LOADING ====================
def load_uci_full():
    """Load and merge UCI Student Performance datasets"""
    print("="*60)
    print("Loading UCI Student Performance Dataset")
    print("="*60)
    
    print("\n[1/5] Reading Math dataset...")
    df_mat = pd.read_csv(UCI_MAT_PATH, sep=';')
    df_mat['course'] = 'Math'
    print(f"✓ Loaded {len(df_mat)} Math students")
    
    print("\n[2/5] Reading Portuguese dataset...")
    df_por = pd.read_csv(UCI_POR_PATH, sep=';')
    df_por['course'] = 'Portuguese'
    print(f"✓ Loaded {len(df_por)} Portuguese students")
    
    print("\n[3/5] Merging datasets (removing duplicates)...")
    # Merge on student attributes, keep first occurrence
    merge_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 
                  'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian']
    
    df_combined = pd.concat([df_mat, df_por], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=merge_cols, keep='first')
    print(f"✓ Total unique students: {len(df_combined)}")
    
    print("\n[4/5] Computing concept scores...")
    students_data = []
    
    for idx, row in df_combined.iterrows():
        # Compute 5 concept scores
        concepts_scores = compute_concept_scores(row)
        
        # Create student profile string for LLM
        profile = create_student_profile(row)
        
        students_data.append({
            'student_id': idx,
            'profile': profile,
            'ground_truth': concepts_scores,
            'final_grade': row['G3']
        })
    
    print(f"✓ Processed {len(students_data)} students")
    
    print("\n[5/5] Dataset Statistics:")
    print(f"  Total students: {len(students_data)}")
    print(f"  Math students: {len(df_mat)}")
    print(f"  Portuguese students: {len(df_por)}")
    print(f"  Unique merged: {len(df_combined)}")
    print(f"  Concepts: {K}")
    
    return students_data

def compute_concept_scores(row):
    """Compute ground truth for 5 concepts based on student attributes"""
    scores = np.zeros(K)
    
    # 1. Study Habits (studytime, failures, absences)
    studytime_norm = row['studytime'] / 4.0
    failures_penalty = max(0, 1 - row['failures'] * 0.25)
    absences_penalty = max(0, 1 - row['absences'] / 50.0)
    scores[0] = (studytime_norm + failures_penalty + absences_penalty) / 3.0
    
    # 2. Family Support (famsup, famrel, Medu, Fedu)
    famsup = 1.0 if row['famsup'] == 'yes' else 0.5
    famrel_norm = row['famrel'] / 5.0
    medu_norm = row['Medu'] / 4.0
    fedu_norm = row['Fedu'] / 4.0
    scores[1] = (famsup + famrel_norm + medu_norm + fedu_norm) / 4.0
    
    # 3. School Engagement (activities, higher, goout)
    activities = 1.0 if row['activities'] == 'yes' else 0.4
    higher = 1.0 if row['higher'] == 'yes' else 0.3
    goout_inv = 1.0 - (row['goout'] / 5.0)  # Less going out = more engagement
    scores[2] = (activities + higher + goout_inv) / 3.0
    
    # 4. Social Factors (romantic, freetime, Dalc, Walc)
    romantic_impact = 0.6 if row['romantic'] == 'yes' else 0.8
    freetime_norm = row['freetime'] / 5.0
    alcohol_penalty = 1.0 - ((row['Dalc'] + row['Walc']) / 10.0)
    scores[3] = (romantic_impact + freetime_norm + alcohol_penalty) / 3.0
    
    # 5. Academic Foundation (G1, G2, schoolsup, paid)
    g1_norm = row['G1'] / 20.0
    g2_norm = row['G2'] / 20.0
    schoolsup = 0.9 if row['schoolsup'] == 'yes' else 0.7
    paid = 0.9 if row['paid'] == 'yes' else 0.7
    scores[4] = (g1_norm + g2_norm + schoolsup + paid) / 4.0
    
    return np.clip(scores, 0, 1)

def create_student_profile(row):
    """Create text profile for LLM assessment"""
    profile = f"""Student Profile:
- Age: {row['age']}, Gender: {row['sex']}
- Study time: {row['studytime']}/4, Failures: {row['failures']}, Absences: {row['absences']}
- Family support: {row['famsup']}, Family relations: {row['famrel']}/5
- Mother education: {row['Medu']}/4, Father education: {row['Fedu']}/4
- Extra activities: {row['activities']}, Wants higher ed: {row['higher']}
- Free time: {row['freetime']}/5, Goes out: {row['goout']}/5
- Romantic relationship: {row['romantic']}
- School support: {row['schoolsup']}, Paid classes: {row['paid']}
- Previous grades: G1={row['G1']}, G2={row['G2']}
- Alcohol consumption: Weekday={row['Dalc']}/5, Weekend={row['Walc']}/5"""
    
    return profile

# ==================== LLM ASSESSMENT ====================
def get_llm_assessment(student_profile, model='llama', retry_count=0):
    """Get knowledge state assessment from specific LLM"""
    
    prompt = f"""You are an educational assessment expert. Based on this student profile, estimate their level (0.0 to 1.0) for 5 key factors:

{student_profile}

Assess these 5 factors:
1. Study Habits - discipline, time management, attendance
2. Family Support - parental involvement, educational background, family relationships
3. School Engagement - participation in activities, higher education aspirations
4. Social Factors - free time management, social relationships, lifestyle balance
5. Academic Foundation - prior performance, support systems, learning resources

Return ONLY a JSON array with exactly 5 numbers between 0.0 and 1.0:
[study_habits, family_support, school_engagement, social_factors, academic_foundation]

Example: [0.75, 0.80, 0.65, 0.70, 0.85]

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
        
        print(f"  Warning: Could not parse {model} response")
        return np.random.uniform(0.4, 0.6, K)
        
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"  Retry {retry_count+1}/{MAX_RETRIES} for {model}: {str(e)[:50]}")
            time.sleep(2 ** retry_count)
            return get_llm_assessment(student_profile, model, retry_count + 1)
        else:
            print(f"  Error with {model} after {MAX_RETRIES} retries: {str(e)[:50]}")
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
        """Aggregate with residual correction and DP"""
        llama_pred, gpt4_pred, claude_pred = predictions
        
        corrected_preds = [
            llama_pred - self.alpha * self.residuals['llama'],
            gpt4_pred - self.alpha * self.residuals['gpt4'],
            claude_pred - self.alpha * self.residuals['claude']
        ]
        
        avg_pred = np.mean(corrected_preds, axis=0)
        
        if self.epsilon < float('inf'):
            sensitivity = K
            noise = np.random.laplace(0, sensitivity/self.epsilon, K)
            avg_pred += noise
        
        return np.clip(avg_pred, 0, 1)
    
    def update_residuals(self, all_predictions):
        """Update residual corrections"""
        if len(all_predictions) < 10:
            return
        
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
    """Load most recent checkpoint"""
    import pickle
    import glob
    
    pattern = os.path.join(CHECKPOINT_DIR, f'checkpoint_eps_{epsilon}_idx_*.pkl')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=os.path.getctime)
    with open(latest, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"✓ Loaded checkpoint from student {checkpoint['student_idx']}")
    return checkpoint

# ==================== MAIN EXPERIMENT ====================
def run_experiment(epsilon=2.0, resume=True):
    """Run full UCI experiment"""
    
    print("\n" + "="*60)
    print(f"UCI STUDENT PERFORMANCE FULL EXPERIMENT (ε={epsilon})")
    print("="*60)
    
    students_data = load_uci_full()
    N = len(students_data)
    
    framework = FederatedCognitiveDiagnosis(epsilon=epsilon, alpha=ALPHA)
    
    # Try to resume
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
    
    start_time = time.time()
    total_cost = 0.0
    
    print(f"\n{'='*60}")
    print(f"Processing {N - start_idx} students (starting from {start_idx})...")
    print(f"{'='*60}\n")
    
    for i in range(start_idx, N):
        student = students_data[i]
        
        if (i + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Processing student {i+1}/{N} ({elapsed:.1f} min elapsed)", flush=True)
        
        # Get predictions
        llama_pred = get_llm_assessment(student['profile'], 'llama')
        time.sleep(RATE_LIMIT_DELAY)
        
        gpt4_pred = get_llm_assessment(student['profile'], 'gpt4')
        time.sleep(RATE_LIMIT_DELAY)
        
        claude_pred = get_llm_assessment(student['profile'], 'claude')
        time.sleep(RATE_LIMIT_DELAY)
        
        all_predictions.append([llama_pred, gpt4_pred, claude_pred])
        
        fed_pred = framework.federated_aggregate([llama_pred, gpt4_pred, claude_pred])
        federated_predictions.append(fed_pred)
        
        total_cost += 0.002
        
        if (i + 1) % 100 == 0:
            framework.update_residuals(all_predictions[-100:])
        
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(i, all_predictions, federated_predictions, epsilon)
    
    elapsed_time = (time.time() - start_time) / 60
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Computing Metrics...")
    print(f"{'='*60}\n")
    
    fed_preds = np.array(federated_predictions)
    gt_array = np.array([s['ground_truth'] for s in students_data])
    
    mae = np.mean(np.abs(fed_preds - gt_array))
    rmse = np.sqrt(np.mean((fed_preds - gt_array)**2))
    concept_mae = np.mean(np.abs(fed_preds - gt_array), axis=0)
    
    baseline_preds = np.array([p[0] for p in all_predictions])
    baseline_mae = np.mean(np.abs(baseline_preds - gt_array))
    baseline_rmse = np.sqrt(np.mean((baseline_preds - gt_array)**2))
    
    improvement = ((baseline_mae - mae) / baseline_mae) * 100
    
    # Results
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
        'total_cost_usd': float(total_cost),
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
    print(f"Cost: ${total_cost:.2f}")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f'results_epsilon_{epsilon}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    np.save(os.path.join(OUTPUT_DIR, f'predictions_epsilon_{epsilon}.npy'), fed_preds)
    np.save(os.path.join(OUTPUT_DIR, f'baseline_predictions_epsilon_{epsilon}.npy'), baseline_preds)
    
    if epsilon == 2.0:
        np.save(os.path.join(OUTPUT_DIR, 'ground_truth.npy'), gt_array)
    
    return results

if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=str, default='2.0')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    
    epsilon = float(args.epsilon) if args.epsilon != 'inf' else float('inf')
    
    if not all([GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY]):
        print("ERROR: Missing API keys!")
        sys.exit(1)
    
    results = run_experiment(epsilon=epsilon, resume=not args.no_resume)
    
    print("\n✓ Experiment completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
