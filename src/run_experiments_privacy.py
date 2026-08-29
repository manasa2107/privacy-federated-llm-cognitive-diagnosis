import pandas as pd
import numpy as np
from federated_simple import federated_diagnosis
from simple_llm import diagnose_student
import time

def run_privacy_experiments():
    """
    Privacy-Utility Tradeoff Experiment
    
    Compares:
    1. Baseline (single LLM)
    2. Federated without privacy
    3. Federated with ε=0.5 (high privacy)
    4. Federated with ε=1.0 (medium privacy)
    5. Federated with ε=2.0 (low privacy)
    """
    
    print("\n" + "="*70)
    print("DIFFERENTIAL PRIVACY TRADEOFF EXPERIMENT - ASSIST09")
    print("="*70)
    
    data = pd.read_csv('assist09_processed.csv')
    concepts = ["equations", "percentages", "integers", "conversions"]
    
    print(f"\nDataset: {len(data)} students from ASSIST09")
    print(f"Privacy Mechanism: Laplace Mechanism (Central DP)")
    print(f"Testing ε values: [No privacy, 2.0, 1.0, 0.5]")
    
    results = {
        'baseline': [],
        'fed_no_privacy': [],
        'fed_epsilon_2.0': [],
        'fed_epsilon_1.0': [],
        'fed_epsilon_0.5': [],
        'true_states': []
    }
    
    start_time = time.time()
    
    for idx, row in data.iterrows():
        print(f"\n{'='*70}")
        print(f"Student {idx+1}/{len(data)} (ID: {row['student_id']})")
        print(f"{'='*70}")
        
        true_state = eval(row['true_state']) if isinstance(row['true_state'], str) else row['true_state']
        
        # 1. Baseline
        print("\n[BASELINE - Single LLM]")
        baseline_pred = diagnose_student(row['responses'], concepts, entity_id=0)
        
        # 2. Federated without privacy
        print("\n[FEDERATED - No Privacy]")
        fed_no_priv, _ = federated_diagnosis(row['responses'], concepts, num_entities=3, use_privacy=False)
        
        # 3. Federated with ε=2.0 (low privacy, less noise)
        print("\n[FEDERATED - ε=2.0]")
        fed_eps_2, _ = federated_diagnosis(row['responses'], concepts, num_entities=3, use_privacy=True, epsilon=2.0)
        
        # 4. Federated with ε=1.0 (medium privacy)
        print("\n[FEDERATED - ε=1.0]")
        fed_eps_1, _ = federated_diagnosis(row['responses'], concepts, num_entities=3, use_privacy=True, epsilon=1.0)
        
        # 5. Federated with ε=0.5 (high privacy, more noise)
        print("\n[FEDERATED - ε=0.5]")
        fed_eps_0p5, _ = federated_diagnosis(row['responses'], concepts, num_entities=3, use_privacy=True, epsilon=0.5)
        
        results['baseline'].append(baseline_pred)
        results['fed_no_privacy'].append(fed_no_priv)
        results['fed_epsilon_2.0'].append(fed_eps_2)
        results['fed_epsilon_1.0'].append(fed_eps_1)
        results['fed_epsilon_0.5'].append(fed_eps_0p5)
        results['true_states'].append(true_state)
        
        print(f"\nTrue:             {[f'{x:.2f}' for x in true_state]}")
        print(f"Baseline:         {[f'{x:.2f}' for x in baseline_pred]}")
        print(f"Fed (no privacy): {[f'{x:.2f}' for x in fed_no_priv]}")
        print(f"Fed (ε=2.0):      {[f'{x:.2f}' for x in fed_eps_2]}")
        print(f"Fed (ε=1.0):      {[f'{x:.2f}' for x in fed_eps_1]}")
        print(f"Fed (ε=0.5):      {[f'{x:.2f}' for x in fed_eps_0p5]}")
    
    # Calculate metrics
    true_states = np.array(results['true_states'])
    
    metrics = {}
    for key in ['baseline', 'fed_no_privacy', 'fed_epsilon_2.0', 'fed_epsilon_1.0', 'fed_epsilon_0.5']:
        preds = np.array(results[key])
        mae = np.mean(np.abs(true_states - preds))
        rmse = np.sqrt(np.mean((true_states - preds) ** 2))
        metrics[key] = {'mae': mae, 'rmse': rmse}
    
    # Display results
    print("\n" + "="*70)
    print("PRIVACY-UTILITY TRADEOFF RESULTS")
    print("="*70)
    print(f"Dataset: ASSIST09 ({len(data)} students, 4 concepts)")
    print(f"Models: LLaMA-3.3-70B + GPT-4o-mini + Claude-3-Haiku")
    print(f"Privacy: Laplace Mechanism (Central DP)")
    print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
    
    baseline_mae = metrics['baseline']['mae']
    
    print(f"\n{'Configuration':<30} {'MAE':<10} {'RMSE':<10} {'Improvement':<12} {'Privacy Cost'}")
    print("-"*90)
    print(f"{'Baseline (Single LLM)':<30} {metrics['baseline']['mae']:.4f}     {metrics['baseline']['rmse']:.4f}     -            -")
    
    for config, label in [
        ('fed_no_privacy', 'Federated (No Privacy)'),
        ('fed_epsilon_2.0', 'Federated (ε=2.0)'),
        ('fed_epsilon_1.0', 'Federated (ε=1.0)'),
        ('fed_epsilon_0.5', 'Federated (ε=0.5)')
    ]:
        mae = metrics[config]['mae']
        rmse = metrics[config]['rmse']
        improvement = ((baseline_mae - mae) / baseline_mae * 100)
        
        if config == 'fed_no_privacy':
            privacy_cost = "-"
        else:
            no_priv_mae = metrics['fed_no_privacy']['mae']
            privacy_cost = f"{((mae - no_priv_mae) / no_priv_mae * 100):+.2f}%"
        
        print(f"{label:<30} {mae:.4f}     {rmse:.4f}     {improvement:>6.2f}%      {privacy_cost}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Privacy-utility tradeoff
    no_priv_improvement = ((baseline_mae - metrics['fed_no_privacy']['mae']) / baseline_mae * 100)
    eps1_improvement = ((baseline_mae - metrics['fed_epsilon_1.0']['mae']) / baseline_mae * 100)
    utility_retention = (eps1_improvement / no_priv_improvement * 100)
    
    print(f"• Without privacy: {no_priv_improvement:.2f}% improvement")
    print(f"• With ε=1.0 privacy: {eps1_improvement:.2f}% improvement")
    print(f"• Utility retention: {utility_retention:.1f}% (only {100-utility_retention:.1f}% loss for privacy)")
    
    # Statistical significance
    try:
        from scipy import stats
        baseline_errors = np.abs(true_states - np.array(results['baseline'])).flatten()
        fed_eps1_errors = np.abs(true_states - np.array(results['fed_epsilon_1.0'])).flatten()
        t_stat, p_value = stats.ttest_rel(baseline_errors, fed_eps1_errors)
        
        print(f"\n• Statistical significance (ε=1.0): p = {p_value:.6f}", end="")
        if p_value < 0.001:
            print(" ✓ Highly significant (p < 0.001)")
        elif p_value < 0.05:
            print(" ✓ Significant (p < 0.05)")
        else:
            print(" (not significant)")
    except:
        print("\n• scipy not available for statistical testing")
    
    print("="*70)
    
    # Save summary
    with open('privacy_tradeoff_summary.txt', 'w') as f:
        f.write("PRIVACY-UTILITY TRADEOFF SUMMARY\n")
        f.write("="*70 + "\n\n")
        for config in ['baseline', 'fed_no_privacy', 'fed_epsilon_2.0', 'fed_epsilon_1.0', 'fed_epsilon_0.5']:
            f.write(f"{config}: MAE={metrics[config]['mae']:.4f}, RMSE={metrics[config]['rmse']:.4f}\n")
    
    print("\n✓ Summary saved to: privacy_tradeoff_summary.txt")

if __name__ == "__main__":
    run_privacy_experiments()
