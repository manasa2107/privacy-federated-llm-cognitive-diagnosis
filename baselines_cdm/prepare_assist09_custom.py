"""
Custom ASSIST09 Data Preparation
Handles the processed format with responses string and true_state
"""

import os
import pandas as pd
import numpy as np
import json
import ast

def prepare_assist09_custom(input_file, output_path):
    print("="*60)
    print("ASSIST09 Custom Data Preparation")
    print("="*60)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Load processed data
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} students")
    
    # Parse responses into student-question-correct format
    all_responses = []
    
    for idx, row in df.iterrows():
        student_id = row['student_id']
        responses_str = row['responses']
        
        # Parse "Q1: wrong, Q2: correct, ..." format
        response_items = responses_str.split(', ')
        
        for item in response_items:
            parts = item.split(': ')
            if len(parts) == 2:
                question_id = int(parts[0].replace('Q', ''))
                correct = 1 if parts[1] == 'correct' else 0
                
                all_responses.append({
                    'student_id': student_id,
                    'question_id': question_id,
                    'correct': correct
                })
    
    responses_df = pd.DataFrame(all_responses)
    
    print(f"\nProcessed {len(responses_df)} student-question pairs")
    print(f"Students: {responses_df['student_id'].nunique()}")
    print(f"Questions: {responses_df['question_id'].nunique()}")
    
    # Save student responses
    responses_file = os.path.join(output_path, 'student_responses.csv')
    responses_df.to_csv(responses_file, index=False)
    print(f"Saved: {responses_file}")
    
    # Create Q-matrix (4 concepts for ASSIST09)
    concepts = ['Equations', 'Percentages', 'Integers', 'Conversions']
    n_questions = responses_df['question_id'].max()
    
    # Simple Q-matrix: distribute questions across concepts
    q_matrix_data = []
    for q in range(1, n_questions + 1):
        row = {'question_id': q}
        
        # Simple heuristic: assign 1-2 concepts per question
        # Q1-5: Equations, Q6-10: Percentages, Q11-15: Integers, Q16-20: Conversions
        if q <= 5:
            row['Equations'] = 1
            row['Percentages'] = 0
            row['Integers'] = 0
            row['Conversions'] = 0
        elif q <= 10:
            row['Equations'] = 0
            row['Percentages'] = 1
            row['Integers'] = 0
            row['Conversions'] = 0
        elif q <= 15:
            row['Equations'] = 0
            row['Percentages'] = 0
            row['Integers'] = 1
            row['Conversions'] = 0
        else:
            row['Equations'] = 0
            row['Percentages'] = 0
            row['Integers'] = 0
            row['Conversions'] = 1
        
        q_matrix_data.append(row)
    
    q_matrix_df = pd.DataFrame(q_matrix_data)
    qmatrix_file = os.path.join(output_path, 'q_matrix.csv')
    q_matrix_df.to_csv(qmatrix_file, index=False)
    print(f"Saved: {qmatrix_file}")
    
    # Parse ground truth knowledge states
    print("\nParsing ground truth knowledge states...")
    
    knowledge_data = []
    for idx, row in df.iterrows():
        student_id = row['student_id']
        
        # Parse true_state: "[0.5, 0.5, 0.8, 0.5454545454545454]"
        true_state_str = row['true_state']
        try:
            knowledge_values = ast.literal_eval(true_state_str)
            
            knowledge_data.append({
                'student_id': student_id,
                'Equations': knowledge_values[0],
                'Percentages': knowledge_values[1],
                'Integers': knowledge_values[2],
                'Conversions': knowledge_values[3]
            })
        except:
            print(f"Warning: Could not parse true_state for student {student_id}")
    
    ground_truth_df = pd.DataFrame(knowledge_data)
    gt_file = os.path.join(output_path, 'ground_truth_knowledge.csv')
    ground_truth_df.to_csv(gt_file, index=False)
    print(f"Saved: {gt_file}")
    
    print("\nGround Truth Knowledge Summary:")
    print(ground_truth_df[concepts].describe())
    
    # Save metadata
    metadata = {
        'n_students': int(responses_df['student_id'].nunique()),
        'n_questions': int(n_questions),
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


if __name__ == "__main__":
    input_file = '/N/lustre/project/proj-606/m_research/assist09_processed_full.csv'
    output_path = '/N/lustre/project/proj-606/m_research/baselines_cdm/data'
    
    prepare_assist09_custom(input_file, output_path)
