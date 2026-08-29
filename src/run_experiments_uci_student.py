import pandas as pd
import numpy as np
from federated_simple import federated_diagnosis
from simple_llm import diagnose_student
import time

def load_uci_student_data(n_students=100):
    """Load UCI Student Performance (Math) dataset"""
    
    # Load math performance data
    df = pd.read_csv('student-mat.csv', sep=';')
    print(f"Loaded {len(df)} students from UCI dataset")
    
    # Define cognitive/performance concepts
    concepts = [
        'study_habits',           # studytime, failures
        'family_support',         # Medu, Fedu, famsup
        'school_engagement',      # absences, schoolsup, activities
        'social_factors',         # goout, romantic, freetime
        'academic_foundation'     # G1, G2 (prior grades)
    ]
    
    # Sample students
    df_sample = df.sample(n=min(n_students, len(df)), random_state=42)
    
    students = []
    np.random.seed(42)
    
    for idx, row in df_sample.iterrows():
        # Create response text from student attributes
        study_quality = "high" if row['studytime'] >= 3 else "medium" if row['studytime'] == 2 else "low"
        failures_text = "failed before" if row['failures'] > 0 else "no failures"
        family_edu = "educated" if (row['Medu'] + row['Fedu']) >= 6 else "moderate"
        attendance = "poor" if row['absences'] > 10 else "good"
        support = "supported" if row['schoolsup'] == 'yes' or row['famsup'] == 'yes' else "unsupported"
        
        responses = f"Study time: {study_quality}, Past performance: {failures_text}, Family background: {family_edu}, Attendance: {attendance}, Support: {support}, Activities: {row['activities']}"
        
        # Calculate true mastery based on actual performance
        # G3 is final grade (0-20 scale)
        final_grade = row['G3']
        grade_normalized = final_grade / 20.0  # Convert to 0-1
        
        # True state based on various factors
        true_state = [
            min(1.0, (row['studytime'] / 4.0) * (1 - row['failures'] * 0.2)),  # study_habits
            min(1.0, ((row['Medu'] + row['Fedu']) / 8.0)),  # family_support
            min(1.0, (1 - row['absences'] / 50.0) * (0.5 if row['schoolsup'] == 'yes' else 0.3) + 0.5),  # school_engagement
            min(1.0, 0.8 - (row['goout'] / 5.0) * 0.3),  # social_factors (inverse of going out)
            grade_normalized  # academic_foundation
        ]
        true_state = [max(0.0, min(1.0, x)) for x in true_state]
        
        students.append({
            'student_id': idx,
            'responses': responses,
            'true_state': true_state
        })
    
    return students, concepts

def run_uci_experiments():
    print("\n" + "="*70)
    print("UCI STUDENT PERFORMANCE - HETEROGENEOUS FEDERATION")
    print("="*70)
    
    students, concepts = load_uci_student_data(n_students=100)
    print(f"\nUCI Student Performance: {len(students)} students")
    print(f"Concepts: {concepts}")
    
    results = {
        'baseline': [],
        'fed_no_privacy': [],
        'fed_epsilon_2.0': [],
        'fed_epsilon_1.0': [],
        'true_states': []
    }
    
    start_time = time.time()
    
    for idx, student in enumerate(students):
        print(f"\nStudent {idx+1}/{len(students)} (ID: {student['student_id']})")
        
        true_state = student['true_state']
        
        # Baseline
        baseline_pred = diagnose_student(student['responses'], concepts, entity_id=0)
        
        # Federated no privacy
        fed_no_priv, _ = federated_diagnosis(student['responses'], concepts, num_entities=3, use_privacy=False)
        
        # Federated epsilon=2.0
        fed_eps_2, _ = federated_diagnosis(student['responses'], concepts, num_entities=3, use_privacy=True, epsilon=2.0)
        
        # Federated epsilon=1.0
        fed_eps_1, _ = federated_diagnosis(student['responses'], concepts, num_entities=3, use_privacy=True, epsilon=1.0)
        
        results['baseline'].append(baseline_pred)
        results['fed_no_privacy'].append(fed_no_priv)
        results['fed_epsilon_2.0'].append(fed_eps_2)
        results['fed_epsilon_1.0'].append(fed_eps_1)
        results['true_states'].append(true_state)
        
        print(f"True:             {[f'{x:.2f}' for x in true_state]}")
        print(f"Baseline:         {[f'{x:.2f}' for x in baseline_pred]}")
        print(f"Fed (no privacy): {[f'{x:.2f}' for x in fed_no_priv]}")
    
    # Calculate metrics
    true_states = np.array(results['true_states'])
    baseline_preds = np.array(results['baseline'])
    fed_no_priv_preds = np.array(results['fed_no_privacy'])
    fed_eps_2_preds = np.array(results['fed_epsilon_2.0'])
    fed_eps_1_preds = np.array(results['fed_epsilon_1.0'])
    
    baseline_mae = np.mean(np.abs(true_states - baseline_preds))
    fed_no_priv_mae = np.mean(np.abs(true_states - fed_no_priv_preds))
    fed_eps_2_mae = np.mean(np.abs(true_states - fed_eps_2_preds))
    fed_eps_1_mae = np.mean(np.abs(true_states - fed_eps_1_preds))
    
    baseline_rmse = np.sqrt(np.mean((true_states - baseline_preds) ** 2))
    fed_no_priv_rmse = np.sqrt(np.mean((true_states - fed_no_priv_preds) ** 2))
    fed_eps_2_rmse = np.sqrt(np.mean((true_states - fed_eps_2_preds) ** 2))
    fed_eps_1_rmse = np.sqrt(np.mean((true_states - fed_eps_1_preds) ** 2))
    
    print("\n" + "="*70)
    print("UCI STUDENT PERFORMANCE FINAL RESULTS")
    print("="*70)
    print(f"Models: LLaMA-3.3 + GPT-4o + Claude-3-Haiku")
    print(f"Students: {len(students)}")
    print(f"Time: {(time.time()-start_time)/60:.1f} min")
    print(f"\nBaseline MAE: {baseline_mae:.4f}")
    print(f"Fed (No Privacy): {fed_no_priv_mae:.4f} ({((baseline_mae-fed_no_priv_mae)/baseline_mae*100):.2f}%)")
    print(f"Fed (epsilon=2.0): {fed_eps_2_mae:.4f} ({((baseline_mae-fed_eps_2_mae)/baseline_mae*100):.2f}%)")
    print(f"Fed (epsilon=1.0): {fed_eps_1_mae:.4f} ({((baseline_mae-fed_eps_1_mae)/baseline_mae*100):.2f}%)")
    print("="*70)
    
    # Save results
    with open('uci_student_FINAL_RESULTS.txt', 'w') as f:
        f.write(f"baseline: MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}\n")
        f.write(f"fed_no_privacy: MAE={fed_no_priv_mae:.4f}, RMSE={fed_no_priv_rmse:.4f}\n")
        f.write(f"fed_epsilon_2.0: MAE={fed_eps_2_mae:.4f}, RMSE={fed_eps_2_rmse:.4f}\n")
        f.write(f"fed_epsilon_1.0: MAE={fed_eps_1_mae:.4f}, RMSE={fed_eps_1_rmse:.4f}\n")
    
    print("\nResults saved to: uci_student_FINAL_RESULTS.txt")

if __name__ == "__main__":
    run_uci_experiments()
