import pandas as pd
import numpy as np
import json
import os

# Load raw data
df = pd.read_csv('/N/lustre/project/proj-606/m_research/skill_builder_data.csv', 
                 encoding='latin1', on_bad_lines='skip')
print(f"Loaded {len(df)} rows, {df['user_id'].nunique()} students")

# Create response data
responses_df = df[['user_id', 'problem_id', 'correct']].copy()
responses_df.columns = ['student_id', 'question_id', 'correct']
responses_df = responses_df.drop_duplicates()

# Save
output_path = 'data'
responses_df.to_csv(f'{output_path}/student_responses.csv', index=False)

# Create simple Q-matrix (4 concepts, distribute evenly)
questions = responses_df['question_id'].unique()
q_matrix = pd.DataFrame({
    'question_id': questions,
    'Equations': [(q % 4) == 0 for q in questions],
    'Percentages': [(q % 4) == 1 for q in questions],
    'Integers': [(q % 4) == 2 for q in questions],
    'Conversions': [(q % 4) == 3 for q in questions]
})
q_matrix.to_csv(f'{output_path}/q_matrix.csv', index=False)

# Create ground truth (proportion correct per concept)
students = responses_df['student_id'].unique()
gt_data = []
for sid in students:
    s_data = responses_df[responses_df['student_id'] == sid]
    gt_data.append({
        'student_id': sid,
        'Equations': s_data[s_data['question_id'].isin(q_matrix[q_matrix['Equations']]['question_id'])]['correct'].mean(),
        'Percentages': s_data[s_data['question_id'].isin(q_matrix[q_matrix['Percentages']]['question_id'])]['correct'].mean(),
        'Integers': s_data[s_data['question_id'].isin(q_matrix[q_matrix['Integers']]['question_id'])]['correct'].mean(),
        'Conversions': s_data[s_data['question_id'].isin(q_matrix[q_matrix['Conversions']]['question_id'])]['correct'].mean()
    })

pd.DataFrame(gt_data).fillna(0.5).to_csv(f'{output_path}/ground_truth_knowledge.csv', index=False)
print("Done!")
# Save metadata
metadata = {
    'n_students': len(students),
    'n_questions': len(questions),
    'n_responses': len(responses_df),
    'n_concepts': 4,
    'concepts': ['Equations', 'Percentages', 'Integers', 'Conversions'],
    'avg_correct_rate': float(responses_df['correct'].mean())
}
with open(f'{output_path}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
