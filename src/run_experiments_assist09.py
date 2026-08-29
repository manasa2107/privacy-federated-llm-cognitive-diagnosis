import pandas as pd
import numpy as np
from federated_simple import federated_diagnosis
from simple_llm import diagnose_student
import time

def run_assist09_experiments():
    print("\n" + "="*70)
    print("ASSIST09 BENCHMARK - HETEROGENEOUS FEDERATION")
    print("="*70)
    
    data = pd.read_csv('assist09_processed.csv')
    concepts = ["equations", "percentages", "integers", "conversions"]
    
    print(f"\nReal ASSIST09 data: {len(data)} students")
    print(f"Concepts: Equation Solving, Percentages, Integers, Conversions")
    
    results = {'baseline': [], 'federated': [], 'true_states': []}
    start_time = time.time()
    
    for idx, row in data.iterrows():
        print(f"\n{'='*70}")
        print(f"Student {idx+1}/{len(data)} (ID: {row['student_id']})")
        
        true_state = eval(row['true_state']) if isinstance(row['true_state'], str) else row['true_state']
        
        baseline_pred = diagnose_student(row['responses'], concepts, entity_id=0)
        federated_pred, _ = federated_diagnosis(row['responses'], concepts, num_entities=3)
        
        results['baseline'].append(baseline_pred)
        results['federated'].append(federated_pred)
        results['true_states'].append(true_state)
        
        print(f"True:      {[f'{x:.2f}' for x in true_state]}")
        print(f"Baseline:  {[f'{x:.2f}' for x in baseline_pred]}")
        print(f"Federated: {[f'{x:.2f}' for x in federated_pred]}")
    
    true_states = np.array(results['true_states'])
    baseline_preds = np.array(results['baseline'])
    federated_preds = np.array(results['federated'])
    
    baseline_mae = np.mean(np.abs(true_states - baseline_preds))
    federated_mae = np.mean(np.abs(true_states - federated_preds))
    baseline_rmse = np.sqrt(np.mean((true_states - baseline_preds) ** 2))
    federated_rmse = np.sqrt(np.mean((true_states - federated_preds) ** 2))
    
    print("\n" + "="*70)
    print("ASSIST09 FINAL RESULTS")
    print("="*70)
    print(f"Models: LLaMA-3.3 + GPT-4o + Claude-3-Haiku")
    print(f"Students: {len(data)}")
    print(f"Time: {(time.time()-start_time)/60:.1f} min")
    print(f"\nBaseline MAE:   {baseline_mae:.4f}")
    print(f"Federated MAE:  {federated_mae:.4f}")
    print(f"Improvement:    {((baseline_mae-federated_mae)/baseline_mae*100):.2f}%")
    print(f"\nBaseline RMSE:  {baseline_rmse:.4f}")
    print(f"Federated RMSE: {federated_rmse:.4f}")
    print(f"Improvement:    {((baseline_rmse-federated_rmse)/baseline_rmse*100):.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_assist09_experiments()
