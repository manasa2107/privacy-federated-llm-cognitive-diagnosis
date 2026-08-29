import os
"""
NEW BASELINES for ACL 2026 Rebuttal
Addresses: 8S5L W4, 8Bf2 W3, 8Bf3 W2
Baselines:
  1. Simple 3-Model Ensemble (no DP, no RC)
  2. w/o Residual Correction (LDP eps=2.0, no RC)
  3. Per-FedAvg (personalized residual per entity)
  4. Gaussian-DP (Gaussian noise, Renyi-DP accounting)
Runs on: ASSIST09, GSM8K, UCI
"""

import numpy as np
import pandas as pd
import time, json, os, re
from datetime import datetime
from collections import defaultdict
from groq import Groq
from openai import OpenAI
from anthropic import Anthropic

# ── API CLIENTS ──────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

groq_client      = Groq(api_key=GROQ_API_KEY)
openai_client    = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

BASE = '/N/lustre/project/proj-606/m_research'
OUT  = f'{BASE}/results_new_baselines'
CKPT = f'{BASE}/checkpoints_new_baselines'
os.makedirs(OUT,  exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

ALPHA = 0.3
CHECKPOINT_EVERY = 50

# ── LLM CALL ─────────────────────────────────────────────
def call_llm(entity_id, student_responses, concepts):
    n = len(concepts)
    base = f"""Given student responses: {student_responses}
And knowledge concepts: {concepts}
Estimate knowledge state as a probability vector [0-1] for EXACTLY {n} concepts.
Return ONLY a Python list with EXACTLY {n} numbers, nothing else.
Example: {[0.5]*n}"""
    try:
        if entity_id == 0:
            for _attempt in range(10):
                try:
                    r = groq_client.chat.completions.create(
                        messages=[{"role":"user","content":"You are an educational assessor.\n"+base}],
                        model="llama-3.3-70b-versatile", temperature=0.5)
                    text = r.choices[0].message.content
                    break
                except Exception as _e:
                    if "rate_limit" in str(_e).lower() or "429" in str(_e):
                        import time as _t
                        wait = 130
                        print(f"  Groq rate limit hit, waiting {wait}s...")
                        _t.sleep(wait)
                    else:
                        raise
            else:
                raise Exception("Groq rate limit: max retries exceeded")
        elif entity_id == 1:
            for attempt in range(5):
                try:
                    r = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":"You are an educational assessor with strong mathematical reasoning.\n"+base}],
                        temperature=0.5)
                    text = r.choices[0].message.content
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() and attempt < 4:
                        time.sleep(2**attempt)
                    else:
                        raise
        else:
            r = anthropic_client.messages.create(
                model="claude-3-haiku-20240307", max_tokens=200, temperature=0.5,
                messages=[{"role":"user","content":"You are a precise educational assessor.\n"+base}])
            text = r.content[0].text

        text = text.replace("```python","").replace("```","").strip()
        m = re.search(r'\[[\d\.,\s]+\]', text)
        ks = eval(m.group() if m else text.strip())
        if len(ks) != n:
            ks = (ks + [0.5]*n)[:n]
        return np.clip(ks, 0, 1)
    except Exception as e:
        print(f"  Entity {entity_id} error: {e}")
        return np.array([0.5]*len(concepts))

# ── GET ALL 3 PREDICTIONS ────────────────────────────────
def get_three_preds(responses, concepts):
    p0 = call_llm(0, responses, concepts)
    p1 = call_llm(1, responses, concepts)
    p2 = call_llm(2, responses, concepts)
    return np.array(p0), np.array(p1), np.array(p2)

# ── 4 NEW BASELINES ──────────────────────────────────────
def simple_ensemble(p0, p1, p2):
    """Baseline 1: plain average, no DP, no RC"""
    return np.clip(np.mean([p0,p1,p2], axis=0), 0, 1)

def no_rc_ldp(p0, p1, p2, K, eps=2.0):
    """Baseline 2: LDP noise per entity, then average — NO residual correction"""
    noisy = []
    for p in [p0, p1, p2]:
        noise = np.random.laplace(0, K/eps, K)
        noisy.append(np.clip(p + noise, 0, 1))
    return np.clip(np.mean(noisy, axis=0), 0, 1)

def per_fedavg(p0, p1, p2, residuals):
    """Baseline 3: personalized residual per entity (Per-FedAvg style)"""
    corrected = [
        p0 - ALPHA * residuals[0],
        p1 - ALPHA * residuals[1],
        p2 - ALPHA * residuals[2],
    ]
    return np.clip(np.mean(corrected, axis=0), 0, 1)

def gaussian_dp(p0, p1, p2, K, eps=2.0, delta=1e-5):
    """Baseline 4: Gaussian mechanism (Renyi-DP accounting)"""
    # sigma calibration for (eps, delta)-DP via analytic Gaussian mechanism
    sigma = np.sqrt(2 * np.log(1.25/delta)) * (K / eps)
    noisy = []
    for p in [p0, p1, p2]:
        noise = np.random.normal(0, sigma, K)
        noisy.append(np.clip(p + noise, 0, 1))
    return np.clip(np.mean(noisy, axis=0), 0, 1)

def update_residuals(all_preds_list, K):
    """Compute per-entity residuals from collected predictions"""
    if len(all_preds_list) < 10:
        return [np.zeros(K)]*3
    arr = np.array(all_preds_list)   # shape (N, 3, K)
    global_mean = arr.mean(axis=(0,1))
    return [arr[:,j,:].mean(axis=0) - global_mean for j in range(3)]

# ── CHECKPOINT HELPERS ───────────────────────────────────
def save_ckpt(dataset, idx, data):
    path = f"{CKPT}/{dataset}_ckpt_{idx}.json"
    with open(path,'w') as f:
        json.dump({k: (v.tolist() if isinstance(v,np.ndarray) else v)
                   for k,v in data.items()}, f)

def load_ckpt(dataset):
    files = sorted([f for f in os.listdir(CKPT) if f.startswith(dataset+'_ckpt_')])
    if not files:
        return None, 0
    with open(f"{CKPT}/{files[-1]}") as f:
        data = json.load(f)
    idx = data['last_idx']
    print(f"  Resuming {dataset} from student {idx}")
    return data, idx

# ── RESULT PRINTER ───────────────────────────────────────
def print_results(dataset, gt, res):
    gt = np.array(gt)
    print(f"\n{'='*60}")
    print(f"RESULTS — {dataset}")
    print(f"{'='*60}")
    baseline_mae = np.mean(np.abs(gt - np.array(res['baseline'])))
    names = ['simple_ensemble','no_rc_ldp','per_fedavg','gaussian_dp']
    labels = ['Simple 3-Model Ensemble (no DP, no RC)',
              'w/o Residual Correction  (LDP ε=2.0)  ',
              'Per-FedAvg               (personalized)',
              'Gaussian-DP              (Rényi-DP)   ']
    print(f"  Baseline (LLaMA single)     MAE={baseline_mae:.4f}")
    for name,label in zip(names,labels):
        mae = np.mean(np.abs(gt - np.array(res[name])))
        imp = (baseline_mae - mae)/baseline_mae*100
        print(f"  {label}  MAE={mae:.4f}  ({imp:+.2f}%)")
    print(f"{'='*60}\n")

def save_results(dataset, gt, res):
    gt = np.array(gt)
    baseline_mae = np.mean(np.abs(gt - np.array(res['baseline'])))
    out = {'dataset': dataset, 'n_students': len(gt), 'baseline_mae': baseline_mae, 'baselines': {}}
    for name in ['simple_ensemble','no_rc_ldp','per_fedavg','gaussian_dp']:
        mae = np.mean(np.abs(gt - np.array(res[name])))
        out['baselines'][name] = {'mae': mae, 'improvement_pct': (baseline_mae-mae)/baseline_mae*100}
    with open(f"{OUT}/{dataset}_new_baselines.json",'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {OUT}/{dataset}_new_baselines.json")

# ════════════════════════════════════════════════════════
# DATASET RUNNERS
# ════════════════════════════════════════════════════════

def run_assist09():
    print("\n" + "="*60)
    print("DATASET: ASSIST09")
    print("="*60)
    SKILL_MAPPING = {
        'Equation Solving Two or Fewer Steps': 0,
        'Percent Of': 1,
        'Addition and Subtraction Integers': 2,
        'Conversion of Fraction Decimals Percents': 3
    }
    CONCEPTS = ['Equations','Percentages','Integers','Conversions']
    K = 4

    df = pd.read_csv(f'{BASE}/skill_builder_data.csv',
                     encoding='latin-1', on_bad_lines='skip', low_memory=False)
    df = df[df['skill_name'].isin(SKILL_MAPPING.keys())].copy()
    df['concept_id'] = df['skill_name'].map(SKILL_MAPPING)
    students = [s for s in df['user_id'].unique()
                if df[df['user_id']==s]['concept_id'].nunique() >= 2]
    print(f"  Students with ≥2 concepts: {len(students)}")

    ckpt, start = load_ckpt('assist09')
    results = ckpt['results'] if ckpt else {k:[] for k in ['baseline','simple_ensemble','no_rc_ldp','per_fedavg','gaussian_dp']}
    gt_list = ckpt['gt_list'] if ckpt else []
    all_preds_list = ckpt.get('all_preds_list', []) if ckpt else []
    residuals = [np.zeros(K)]*3

    t0 = time.time()
    for idx, sid in enumerate(students[start:], start=start):
        sdf = df[df['user_id']==sid]
        gt = np.array([sdf[sdf['concept_id']==c]['correct'].mean()
                       if len(sdf[sdf['concept_id']==c])>0 else 0.5 for c in range(K)])
        history = '\n'.join([f"• {r['skill_name']}: {'Correct' if r['correct']==1 else 'Incorrect'}"
                             for _,r in sdf.tail(30).iterrows()])

        print(f"  [{idx+1}/{len(students)}] Student {sid}  elapsed={((time.time()-t0)/60):.1f}m")
        p0,p1,p2 = get_three_preds(history, CONCEPTS)

        # update residuals every 50 students
        all_preds_list.append([p0.tolist(), p1.tolist(), p2.tolist()])
        if len(all_preds_list) % 50 == 0:
            residuals = update_residuals(all_preds_list, K)

        results['baseline'].append(p0.tolist())
        results['simple_ensemble'].append(simple_ensemble(p0,p1,p2).tolist())
        results['no_rc_ldp'].append(no_rc_ldp(p0,p1,p2,K).tolist())
        results['per_fedavg'].append(per_fedavg(p0,p1,p2,residuals).tolist())
        results['gaussian_dp'].append(gaussian_dp(p0,p1,p2,K).tolist())
        gt_list.append(gt.tolist())

        if (idx+1) % CHECKPOINT_EVERY == 0:
            save_ckpt('assist09', idx+1, {'last_idx':idx+1,'results':results,
                                          'gt_list':gt_list,'all_preds_list':all_preds_list})

    print_results('ASSIST09', gt_list, results)
    save_results('assist09', gt_list, results)


def run_gsm8k():
    print("\n" + "="*60)
    print("DATASET: GSM8K")
    print("="*60)
    CONCEPTS = ['Problem_Setup','Arithmetic','Multi_Step_Reasoning','Answer_Verification']
    K = 4

    # Load from jsonl
    path = f'{BASE}/gsm8k_test.jsonl'
    if not os.path.exists(path):
        print(f"  GSM8K file not found at {path}, skipping.")
        return
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line.strip()))
    print(f"  Problems loaded: {len(problems)}")

    ckpt, start = load_ckpt('gsm8k')
    results = ckpt['results'] if ckpt else {k:[] for k in ['baseline','simple_ensemble','no_rc_ldp','per_fedavg','gaussian_dp']}
    gt_list = ckpt['gt_list'] if ckpt else []
    all_preds_list = ckpt.get('all_preds_list', []) if ckpt else []
    residuals = [np.zeros(K)]*3

    t0 = time.time()
    for idx, prob in enumerate(problems[start:], start=start):
        question = prob.get('question','')
        answer   = prob.get('answer','')
        # Extract correct final answer after ####
        import re as _re
        final_ans = _re.search(r'####\s*([\d\.\-]+)', answer)
        correct_answer = final_ans.group(1).strip() if final_ans else ""
        # Ground truth: 1.0 if correct answer is present/solvable, spread across concepts
        # Use number of reasoning steps as proxy for concept coverage
        n_steps = len(_re.findall(r'<<', answer))  # count calc steps in solution
        step_score = min(n_steps / 5.0, 1.0)  # normalize by expected 5 steps
        gt = np.array([
            min(step_score * 1.0, 1.0),   # Problem_Setup
            min(step_score * 1.0, 1.0),   # Arithmetic
            min(step_score * 0.9, 1.0),   # Multi_Step_Reasoning
            min(step_score * 0.8, 1.0),   # Answer_Verification
        ])
        history = f"Question: {question}\nCorrect Answer: {correct_answer}\nSolution steps: {n_steps}"

        print(f"  [{idx+1}/{len(problems)}] Problem {idx+1}  elapsed={((time.time()-t0)/60):.1f}m")
        p0,p1,p2 = get_three_preds(history, CONCEPTS)

        all_preds_list.append([p0.tolist(), p1.tolist(), p2.tolist()])
        if len(all_preds_list) % 50 == 0:
            residuals = update_residuals(all_preds_list, K)

        results['baseline'].append(p0.tolist())
        results['simple_ensemble'].append(simple_ensemble(p0,p1,p2).tolist())
        results['no_rc_ldp'].append(no_rc_ldp(p0,p1,p2,K).tolist())
        results['per_fedavg'].append(per_fedavg(p0,p1,p2,residuals).tolist())
        results['gaussian_dp'].append(gaussian_dp(p0,p1,p2,K).tolist())
        gt_list.append(gt.tolist())

        if (idx+1) % CHECKPOINT_EVERY == 0:
            save_ckpt('gsm8k', idx+1, {'last_idx':idx+1,'results':results,
                                       'gt_list':gt_list,'all_preds_list':all_preds_list})

    print_results('GSM8K', gt_list, results)
    save_results('gsm8k', gt_list, results)


def run_uci():
    print("\n" + "="*60)
    print("DATASET: UCI Student Performance")
    print("="*60)
    CONCEPTS = ['Study_Habits','Family_Support','School_Engagement',
                'Social_Factors','Academic_Foundation']
    K = 5

    path = f'{BASE}/student-por.csv'
    if not os.path.exists(path):
        path = f'{BASE}/student-mat.csv'
    if not os.path.exists(path):
        print(f"  UCI file not found, skipping.")
        return
    df = pd.read_csv(path, sep=';')
    print(f"  Students: {len(df)}")

    # Ground truth: normalize final grade G3 and key attributes to [0,1]
    def uci_gt(row):
        g3   = row['G3']/20.0
        stdy = min(row['studytime']/4.0, 1.0)
        fams = (row['famrel']-1)/4.0
        schl = (row['goout']-1)/4.0  # inverse — less going out = more engaged
        socl = (row['freetime']-1)/4.0
        return np.array([stdy, fams, 1-schl, socl, g3])

    ckpt, start = load_ckpt('uci')
    results = ckpt['results'] if ckpt else {k:[] for k in ['baseline','simple_ensemble','no_rc_ldp','per_fedavg','gaussian_dp']}
    gt_list = ckpt['gt_list'] if ckpt else []
    all_preds_list = ckpt.get('all_preds_list', []) if ckpt else []
    residuals = [np.zeros(K)]*3

    t0 = time.time()
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx < start:
            continue
        gt = uci_gt(row)
        history = (f"Student: age={row['age']}, studytime={row['studytime']}, "
                   f"failures={row['failures']}, famrel={row['famrel']}, "
                   f"goout={row['goout']}, G1={row['G1']}, G2={row['G2']}")

        print(f"  [{idx+1}/{len(df)}] Student {idx+1}  elapsed={((time.time()-t0)/60):.1f}m")
        p0,p1,p2 = get_three_preds(history, CONCEPTS)

        all_preds_list.append([p0.tolist(), p1.tolist(), p2.tolist()])
        if len(all_preds_list) % 50 == 0:
            residuals = update_residuals(all_preds_list, K)

        results['baseline'].append(p0.tolist())
        results['simple_ensemble'].append(simple_ensemble(p0,p1,p2).tolist())
        results['no_rc_ldp'].append(no_rc_ldp(p0,p1,p2,K).tolist())
        results['per_fedavg'].append(per_fedavg(p0,p1,p2,residuals).tolist())
        results['gaussian_dp'].append(gaussian_dp(p0,p1,p2,K).tolist())
        gt_list.append(gt.tolist())

        if (idx+1) % CHECKPOINT_EVERY == 0:
            save_ckpt('uci', idx+1, {'last_idx':idx+1,'results':results,
                                     'gt_list':gt_list,'all_preds_list':all_preds_list})

    print_results('UCI', gt_list, results)
    save_results('uci', gt_list, results)


# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"\nStarted: {datetime.now()}")
    run_assist09()
    run_gsm8k()
    run_uci()
    print(f"\nAll done: {datetime.now()}")
