"""
GSM8K Full Dataset Experiment
Grade School Math 8K problems
Privacy-Preserving Heterogeneous Multi-LLM Federated Cognitive Diagnosis
5 concepts: Arithmetic, Fractions, Percentages, Word Problems, Multi-step
"""

import numpy as np
import pandas as pd
import json
import time
import os
from datetime import datetime
from collections import defaultdict
import argparse

# LLM API imports
import anthropic
from openai import OpenAI
from groq import Groq

# ==================== CONFIGURATION ====================
GSM8K_TRAIN_PATH = './train.jsonl'
GSM8K_TEST_PATH = './test.jsonl'
OUTPUT_DIR = './results_full/gsm8k/'
CHECKPOINT_DIR = './checkpoints/gsm8k/'

# 5 math concepts for GSM8K
CONCEPTS = ['Arithmetic', 'Fractions', 'Percentages', 'Word Problems', 'Multi-step']
K = 5

# Privacy budget (from command line)
ALPHA = 0.3  # Residual correction strength

# API configuration
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0  # 1 second between API calls (avoid rate limits)
CHECKPOINT_INTERVAL = 100

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
def load_gsm8k():
    """Load GSM8K dataset"""
    print("\n" + "="*60)
    print("Loading GSM8K Dataset")
    print("="*60)
    
    problems = []
    
    # Load training data
    print("\n[1/3] Loading training set...")
    with open(GSM8K_TRAIN_PATH, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    print(f"✓ Loaded {len(problems)} training problems")
    
    # Load test data
    print("\n[2/3] Loading test set...")
    test_count = 0
    with open(GSM8K_TEST_PATH, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
            test_count += 1
    print(f"✓ Loaded {test_count} test problems")
    
    print(f"\n[3/3] Total problems: {len(problems)}")
    
    return problems

def classify_problem(question, answer):
    """Classify which concepts a problem involves (Q-matrix)"""
    q_vector = np.zeros(K)
    
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Arithmetic operations
    if any(op in question_lower for op in ['+', '-', '*', '/', 'add', 'subtract', 'multiply', 'divide']):
        q_vector[0] = 1
    
    # Fractions
    if any(word in question_lower for word in ['fraction', 'half', 'quarter', 'third', '1/2', '1/3', '1/4']):
        q_vector[1] = 1
    
    # Percentages
    if any(word in question_lower for word in ['percent', '%', 'percentage']):
        q_vector[2] = 1
    
    # Word problems (all GSM8K are word problems)
    q_vector[3] = 1
    
    # Multi-step (if answer has multiple calculation steps)
    if answer.count('<<') >= 2:  # GSM8K format has <<calculation>>
        q_vector[4] = 1
    
    return q_vector

def compute_ground_truth(problems):
    """Compute ground truth knowledge states"""
    print("\n[4/5] Computing ground truth knowledge states...")
    
    student_data = defaultdict(lambda: {'correct': np.zeros(K), 'total': np.zeros(K)})
    
    for idx, prob in enumerate(problems):
        student_id = f"student_{idx % 1000}"  # Group into 1000 virtual students
        q_vector = classify_problem(prob['question'], prob['answer'])
        
        # Simulate correctness (in reality, you'd have student attempts)
        # For this experiment, we'll use the problems as "questions answered"
        for k in range(K):
            if q_vector[k] == 1:
                student_data[student_id]['total'][k] += 1
                student_data[student_id]['correct'][k] += np.random.binomial(1, 0.7)  # Simulate 70% accuracy
    
    # Convert to ground truth vectors
    ground_truth = {}
    for student_id, data in student_data.items():
        gt = np.zeros(K)
        for k in range(K):
            if data['total'][k] > 0:
                gt[k] = data['correct'][k] / data['total'][k]
            else:
                gt[k] = 0.5
        ground_truth[student_id] = gt
    
    print(f"✓ Computed ground truth for {len(ground_truth)} students")
    return ground_truth, student_data

# ==================== LLM INFERENCE ====================
def query_llm(prompt, model, retries=MAX_RETRIES):
    """Query LLM with retry logic"""
    for attempt in range(retries):
        try:
            if model == 'llama':
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                content = response.choices[0].message.content
                
            elif model == 'gpt4':
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                content = response.choices[0].message.content
                
            elif model == 'claude':
                response = anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=150,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            
            # Parse response
            import re
            json_match = re.search(r'\[[\d\.\,\s]+\]', content)
            if json_match:
                prediction = json.loads(json_match.group())
                if len(prediction) >= K:
                    return np.array(prediction[:K], dtype=float)
            
            print(f"  Warning: Could not parse {model} response")
            return np.random.uniform(0.4, 0.6, K)
            
        except Exception as e:
            print(f"  Retry {attempt+1}/{retries} for {model}: {e}")
            time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
    
    print(f"  Error with {model} after {retries} retries")
    return np.random.uniform(0.4, 0.6, K)

def diagnose_student_federated(student_id, problems, epsilon):
    """Federated diagnosis with privacy"""
    
    # Sample problems for this student
    student_problems = [p for i, p in enumerate(problems) if i % 1000 == int(student_id.split('_')[1])][:5]
    
    prompt = f"""Analyze these math problems and estimate student knowledge in 5 areas:
1. Arithmetic (basic operations)
2. Fractions
3. Percentages  
4. Word Problems
5. Multi-step reasoning

Problems:
{student_problems[:2]}

Return knowledge state as: [arithmetic, fractions, percentages, word_problems, multistep]
Each value 0-1. Example: [0.8, 0.6, 0.7, 0.75, 0.65]"""

    # Get predictions from 3 LLMs (store individually for baseline)
    predictions = []
    for model in ['llama', 'gpt4', 'claude']:
        pred = query_llm(prompt, model)
        predictions.append(pred)
        time.sleep(RATE_LIMIT_DELAY)
    
    # Federated aggregation with residual correction
    residuals = [pred - np.mean(predictions, axis=0) for pred in predictions]
    aggregated = np.mean([pred - ALPHA * res for pred, res in zip(predictions, residuals)], axis=0)
    
    # Add differential privacy noise
    if epsilon != float('inf'):
        sensitivity = 1.0 / len(predictions)
        noise = np.random.laplace(0, sensitivity / epsilon, K)
        aggregated = np.clip(aggregated + noise, 0, 1)
    
    # Return: (federated_pred, individual_predictions)
    # individual_predictions[0] = LLaMA baseline
    return aggregated, predictions

# ==================== MAIN EXPERIMENT ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=str, default='inf')
    args = parser.parse_args()
    
    epsilon = float(args.epsilon) if args.epsilon != 'inf' else float('inf')
    
    print("\n" + "="*60)
    print(f"GSM8K FULL EXPERIMENT (ε={epsilon})")
    print("="*60)
    
    # Load data
    problems = load_gsm8k()
    ground_truth, student_data = compute_ground_truth(problems)
    
    student_ids = list(ground_truth.keys())
    n_students = len(student_ids)
    
    print(f"\n[5/5] Dataset Statistics:")
    print(f"  Total students: {n_students}")
    print(f"  Total problems: {len(problems)}")
    print(f"  Concepts: {K}")
    
    # Process students
    print(f"\n{'='*60}")
    print(f"Processing {n_students} students...")
    print(f"{'='*60}\n")
    
    federated_preds = []
    all_individual_preds = []  # Store [llama, gpt4, claude] for each student
    ground_truth_all = []
    start_time = time.time()
    
    for i, student_id in enumerate(student_ids):
        if (i + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Processing student {i+1}/{n_students} ({elapsed:.1f} min elapsed)", flush=True)
        
        fed_pred, individual_preds = diagnose_student_federated(student_id, problems, epsilon)
        federated_preds.append(fed_pred)
        all_individual_preds.append(individual_preds)
        ground_truth_all.append(ground_truth[student_id])
        
        if (i + 1) % 50 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"[{i+1}/{n_students}] {elapsed:.1f} min elapsed", flush=True)
    
    # Compute metrics
    federated_preds = np.array(federated_preds)
    ground_truth_all = np.array(ground_truth_all)
    
    # Baseline = LLaMA-only (first prediction)
    baseline_preds = np.array([p[0] for p in all_individual_preds])
    
    mae = np.mean(np.abs(federated_preds - ground_truth_all))
    rmse = np.sqrt(np.mean((federated_preds - ground_truth_all) ** 2))
    baseline_mae = np.mean(np.abs(baseline_preds - ground_truth_all))
    baseline_rmse = np.sqrt(np.mean((baseline_preds - ground_truth_all) ** 2))
    improvement = ((baseline_mae - mae) / baseline_mae) * 100
    
    # Save results
    results = {
        'epsilon': epsilon,
        'students': n_students,
        'mae': float(mae),
        'rmse': float(rmse),
        'baseline_mae': float(baseline_mae),
        'baseline_rmse': float(baseline_rmse),
        'improvement': float(improvement),
        'time_minutes': (time.time() - start_time) / 60
    }
    
    output_file = os.path.join(OUTPUT_DIR, f'results_eps_{epsilon}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (ε={epsilon})")
    print(f"{'='*60}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Time: {results['time_minutes']:.1f} minutes")
    print(f"{'='*60}\n")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
