"""
Data Preparation Script for ASSIST09 Baseline Experiments

This script converts your existing ASSIST09 data into the format needed
by the baseline models (NeuralCD, IRT, DINA).

Required input files (from your existing experiments):
- ASSIST09 raw data or processed responses

Output files:
- student_responses.csv: student_id, question_id, correct
- q_matrix.csv: question_id, Equations, Percentages, Integers, Conversions
- ground_truth_knowledge.csv: student_id, Equations, Percentages, Integers, Conversions
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from collections import defaultdict


def prepare_assist09_data(input_path, output_path):
    """
    Prepare ASSIST09 data for baseline experiments
    
    Args:
        input_path: Path to your existing ASSIST09 data
        output_path: Path to save processed data
    """
    print("="*60)
    print("ASSIST09 Data Preparation for Baseline Experiments")
    print("="*60)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Try to find existing ASSIST09 data
    possible_files = [
        'assist09_processed.csv',
        'assist09.csv',
        'ASSIST09.csv',
        'student_responses.csv'
    ]
    
    data_file = None
    for fname in possible_files:
        fpath = os.path.join(input_path, fname)
        if os.path.exists(fpath):
            data_file = fpath
            break
    
    if data_file is None:
        print("ERROR: Could not find ASSIST09 data file.")
        print(f"Looked for: {possible_files}")
        print(f"In directory: {input_path}")
        print("\nPlease specify the correct path to your ASSIST09 data.")
        sys.exit(1)
    
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Determine column mapping
    # Expected columns: student_id, question_id, correct (0/1)
    # Possible variations: user_id, problem_id, skill_id, correct, correctness, etc.
    
    column_mapping = {}
    
    # Student ID
    if 'student_id' in df.columns:
        column_mapping['student_id'] = 'student_id'
    elif 'user_id' in df.columns:
        column_mapping['student_id'] = 'user_id'
    else:
        print("ERROR: Could not find student ID column")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)
    
    # Question ID
    if 'question_id' in df.columns:
        column_mapping['question_id'] = 'question_id'
    elif 'problem_id' in df.columns:
        column_mapping['question_id'] = 'problem_id'
    elif 'item_id' in df.columns:
        column_mapping['question_id'] = 'item_id'
    else:
        print("ERROR: Could not find question ID column")
        sys.exit(1)
    
    # Correctness
    if 'correct' in df.columns:
        column_mapping['correct'] = 'correct'
    elif 'correctness' in df.columns:
        column_mapping['correct'] = 'correctness'
    elif 'score' in df.columns:
        column_mapping['correct'] = 'score'
    else:
        print("ERROR: Could not find correctness column")
        sys.exit(1)
    
    # Create standardized response DataFrame
    responses_df = pd.DataFrame({
        'student_id': df[column_mapping['student_id']],
        'question_id': df[column_mapping['question_id']],
        'correct': df[column_mapping['correct']]
    })
    
    # Ensure correct is binary
    responses_df['correct'] = (responses_df['correct'] > 0).astype(int)
    
    # Remove duplicates (keep first occurrence)
    responses_df = responses_df.drop_duplicates(subset=['student_id', 'question_id'], keep='first')
    
    print(f"\nProcessed {len(responses_df)} unique student-question pairs")
    print(f"Students: {responses_df['student_id'].nunique()}")
    print(f"Questions: {responses_df['question_id'].nunique()}")
    print(f"Average correct rate: {responses_df['correct'].mean():.3f}")
    
    # Save student responses
    responses_file = os.path.join(output_path, 'student_responses.csv')
    responses_df.to_csv(responses_file, index=False)
    print(f"Saved: {responses_file}")
    
    # Create Q-matrix
    # For ASSIST09, we use 4 concepts: Equations, Percentages, Integers, Conversions
    concepts = ['Equations', 'Percentages', 'Integers', 'Conversions']
    
    # Check if Q-matrix already exists in input
    qmatrix_file = os.path.join(input_path, 'q_matrix.csv')
    if os.path.exists(qmatrix_file):
        print(f"\nFound existing Q-matrix: {qmatrix_file}")
        q_matrix_df = pd.read_csv(qmatrix_file)
    else:
        print("\nGenerating Q-matrix from skill assignments...")
        
        # Try to infer from skill column if available
        if 'skill_id' in df.columns or 'skill' in df.columns:
            skill_col = 'skill_id' if 'skill_id' in df.columns else 'skill'
            
            # Create Q-matrix based on skills
            questions = responses_df['question_id'].unique()
            q_matrix_data = []
            
            for q in questions:
                # Get skills for this question
                q_skills = df[df[column_mapping['question_id']] == q][skill_col].unique()
                
                # Map to concepts (simplified heuristic)
                row = {'question_id': q}
                for concept in concepts:
                    # Assign based on simple heuristic (you may need to adjust this)
                    row[concept] = 1 if len(q_skills) > 0 else 0
                
                q_matrix_data.append(row)
            
            q_matrix_df = pd.DataFrame(q_matrix_data)
        else:
            # Generate random Q-matrix (not ideal, but functional)
            print("WARNING: Generating random Q-matrix (no skill information found)")
            questions = responses_df['question_id'].unique()
            
            # Each question requires 1-2 concepts on average
            q_matrix_data = []
            np.random.seed(42)
            
            for q in questions:
                row = {'question_id': q}
                # Randomly assign 1-2 concepts per question
                n_concepts_required = np.random.randint(1, 3)
                selected_concepts = np.random.choice(concepts, n_concepts_required, replace=False)
                
                for concept in concepts:
                    row[concept] = 1 if concept in selected_concepts else 0
                
                q_matrix_data.append(row)
            
            q_matrix_df = pd.DataFrame(q_matrix_data)
    
    # Ensure all questions are in Q-matrix
    missing_questions = set(responses_df['question_id'].unique()) - set(q_matrix_df['question_id'].unique())
    if missing_questions:
        print(f"WARNING: {len(missing_questions)} questions missing from Q-matrix")
        # Add them with default values
        for q in missing_questions:
            new_row = {'question_id': q}
            for concept in concepts:
                new_row[concept] = 1 if np.random.rand() > 0.5 else 0
            q_matrix_df = pd.concat([q_matrix_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save Q-matrix
    qmatrix_output_file = os.path.join(output_path, 'q_matrix.csv')
    q_matrix_df.to_csv(qmatrix_output_file, index=False)
    print(f"Saved: {qmatrix_output_file}")
    print(f"Q-matrix shape: {q_matrix_df.shape}")
    
    # Compute ground truth knowledge states
    print("\nComputing ground truth knowledge states...")
    
    student_ids = responses_df['student_id'].unique()
    knowledge_data = []
    
    for student_id in student_ids:
        student_responses = responses_df[responses_df['student_id'] == student_id]
        
        row = {'student_id': student_id}
        
        for concept in concepts:
            # Get questions requiring this concept
            concept_questions = q_matrix_df[q_matrix_df[concept] == 1]['question_id'].values
            
            # Get student's responses to these questions
            concept_student_responses = student_responses[
                student_responses['question_id'].isin(concept_questions)
            ]
            
            if len(concept_student_responses) > 0:
                # Knowledge = proportion correct on concept questions
                row[concept] = concept_student_responses['correct'].mean()
            else:
                row[concept] = 0.5  # Default if no responses
        
        knowledge_data.append(row)
    
    ground_truth_df = pd.DataFrame(knowledge_data)
    
    # Save ground truth
    gt_file = os.path.join(output_path, 'ground_truth_knowledge.csv')
    ground_truth_df.to_csv(gt_file, index=False)
    print(f"Saved: {gt_file}")
    
    # Print summary statistics
    print("\nGround Truth Knowledge Summary:")
    print(ground_truth_df[concepts].describe())
    
    # Save metadata
    metadata = {
        'n_students': int(responses_df['student_id'].nunique()),
        'n_questions': int(responses_df['question_id'].nunique()),
        'n_responses': int(len(responses_df)),
        'n_concepts': len(concepts),
        'concepts': concepts,
        'avg_correct_rate': float(responses_df['correct'].mean())
    }
    
    metadata_file = os.path.join(output_path, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_file}")
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nOutput directory: {output_path}")
    print("Files created:")
    print("  - student_responses.csv")
    print("  - q_matrix.csv")
    print("  - ground_truth_knowledge.csv")
    print("  - metadata.json")
    print("\nYou can now run: sbatch run_baseline_job.slurm")


if __name__ == "__main__":
    # Set paths
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Default: look in your existing data directory
        input_path = '/N/lustre/project/proj-606/m_research/data/assist09'
    
    output_path = '/N/lustre/project/proj-606/m_research/baselines_cdm/data'
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input path does not exist: {input_path}")
        print("\nUsage: python prepare_assist09_data.py <input_path>")
        sys.exit(1)
    
    prepare_assist09_data(input_path, output_path)
