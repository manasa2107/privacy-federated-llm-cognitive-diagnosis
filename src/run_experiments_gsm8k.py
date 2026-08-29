import json
import numpy as np
from federated_simple import federated_diagnosis
from simple_llm import diagnose_student
import time

def load_gsm8k_students(n_students=100):
    with open('train.jsonl', 'r') as f:
        problems = [json.loads(line) for line in f][:500]
    concepts = ["addition_subtraction", "multiplication_division", "fractions_percentages", "multi_step_reasoning", "word_problems"]
    np.random.seed(42)
    students = []
    for sid in range(n_students):
        true_mastery = [np.random.beta(2, 2) for _ in concepts]
        responses = []
        for i in range(10):
            prob = problems[np.random.randint(len(problems))]
            text = prob['question'].lower()
            req_concepts = []
            if any(w in text for w in ['add', 'sum', 'total']): req_concepts.append(0)
            if any(w in text for w in ['multiply', 'times']): req_concepts.append(1)
            if any(w in text for w in ['percent', 'fraction']): req_concepts.append(2)
            if len(text) > 100: req_concepts.append(3)
            req_concepts.append(4)
            if req_concepts:
                avg_mastery = np.mean([true_mastery[c] for c in req_concepts])
                is_correct = np.random.random() < (0.1 + 0.8 * avg_mastery)
            else:
                is_correct = np.random.random() < 0.5
            responses.append(f"Q{i+1}: {'correct' if is_correct else 'wrong'}")
        students.append({'student_id': sid, 'responses': ', '.join(responses), 'true_state': true_mastery})
    return students, concepts

def run_gsm8k_experiments():
    print("\n" + "="*70)
    print("GSM8K BENCHMARK")
    print("="*70)
    students, concepts = load_gsm8k_students(n_students=100)
    print(f"\nGSM8K dataset: {len(students)} students")
    results = {'baseline': [], 'fed_no_privacy': [], 'fed_epsilon_2.0': [], 'fed_epsilon_1.0': [], 'true_states': []}
    start_time = time.time()
    for idx, student in enumerate(students):
        print(f"\nStudent {idx+1}/{len(students)}")
        true_state = student['true_state']
        baseline_pred = diagnose_student(student['responses'], concepts, entity_id=0)
        fed_no_priv, _ = federated_diagnosis(student['responses'], concepts, num_entities=3, use_privacy=False)
        fed_eps_2, _ = federated_diagnosis(student['responses'], concepts, num_entities=3, use_privacy=True, epsilon=2.0)
        fed_eps_1, _ = federated_diagnosis(student['responses'], concepts, num_entities=3, use_privacy=True, epsilon=1.0)
        results['baseline'].append(baseline_pred)
        results['fed_no_privacy'].append(fed_no_priv)
        results['fed_epsilon_2.0'].append(fed_eps_2)
        results['fed_epsilon_1.0'].append(fed_eps_1)
        results['true_states'].append(true_state)
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
    print("GSM8K FINAL RESULTS")
    print("="*70)
    print(f"Students: {len(students)}")
    print(f"Time: {(time.time()-start_time)/60:.1f} min")
    print(f"\nBaseline MAE: {baseline_mae:.4f}")
    print(f"Fed (No Privacy): {fed_no_priv_mae:.4f} ({((baseline_mae-fed_no_priv_mae)/baseline_mae*100):.2f}%)")
    print(f"Fed (e=2.0): {fed_eps_2_mae:.4f} ({((baseline_mae-fed_eps_2_mae)/baseline_mae*100):.2f}%)")
    print(f"Fed (e=1.0): {fed_eps_1_mae:.4f} ({((baseline_mae-fed_eps_1_mae)/baseline_mae*100):.2f}%)")
    print("="*70)
    with open('gsm8k_FINAL_RESULTS.txt', 'w') as f:
        f.write(f"baseline: MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}\n")
        f.write(f"fed_no_privacy: MAE={fed_no_priv_mae:.4f}, RMSE={fed_no_priv_rmse:.4f}\n")
        f.write(f"fed_epsilon_2.0: MAE={fed_eps_2_mae:.4f}, RMSE={fed_eps_2_rmse:.4f}\n")
        f.write(f"fed_epsilon_1.0: MAE={fed_eps_1_mae:.4f}, RMSE={fed_eps_1_rmse:.4f}\n")

if __name__ == "__main__":
    run_gsm8k_experiments()
