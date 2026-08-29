import os
"""
Quick experiment: Individual LLM MAE on all 3 datasets
Gets real per-model MAE for GPT-4o-mini and Claude-3-Haiku
"""
import numpy as np
import pandas as pd
import time, json, os, re
from groq import Groq
from openai import OpenAI
from anthropic import Anthropic

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

groq_client      = Groq(api_key=GROQ_API_KEY)
openai_client    = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

BASE = '/N/lustre/project/proj-606/m_research'
OUT  = f'{BASE}/results_individual_mae'
os.makedirs(OUT, exist_ok=True)

def call_llm(entity_id, student_responses, concepts):
    n = len(concepts)
    base = f"""Given student responses: {student_responses}
And knowledge concepts: {concepts}
Estimate knowledge state as a probability vector [0-1] for EXACTLY {n} concepts.
Return ONLY a Python list with EXACTLY {n} numbers, nothing else.
Example: {[0.5]*n}"""
    try:
        if entity_id == 0:
            for attempt in range(10):
                try:
                    r = groq_client.chat.completions.create(
                        messages=[{"role":"user","content":"You are an educational assessor.\n"+base}],
                        model="llama-3.3-70b-versatile", temperature=0.5)
                    text = r.choices[0].message.content
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        print(f"  Groq rate limit, waiting 130s...")
                        time.sleep(130)
                    else:
                        raise
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
        return np.array([0.5]*n)

# ── ASSIST09 ──────────────────────────────────────────────────────
def run_assist09_individual():
    print("\n=== ASSIST09 Individual MAE ===")
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
                if df[df['user_id']==s]['concept_id'].nunique() >= 2][:200]
    print(f"  Using {len(students)} students (subset for speed)")

    preds = {0:[], 1:[], 2:[]}
    gt_list = []

    for idx, sid in enumerate(students):
        sdf = df[df['user_id']==sid]
        gt = np.array([sdf[sdf['concept_id']==c]['correct'].mean()
                       if len(sdf[sdf['concept_id']==c])>0 else 0.5 for c in range(K)])
        history = '\n'.join([f"• {r['skill_name']}: {'Correct' if r['correct']==1 else 'Incorrect'}"
                             for _,r in sdf.tail(20).iterrows()])
        print(f"  [{idx+1}/{len(students)}] Student {sid}")
        for eid in [0,1,2]:
            p = call_llm(eid, history, CONCEPTS)
            preds[eid].append(p.tolist())
        gt_list.append(gt.tolist())

    gt = np.array(gt_list)
    results = {}
    names = {0:'LLaMA-3.3-70B', 1:'GPT-4o-mini', 2:'Claude-3-Haiku'}
    print(f"\n--- ASSIST09 Results ({len(students)} students) ---")
    for eid in [0,1,2]:
        mae = np.mean(np.abs(gt - np.array(preds[eid])))
        results[names[eid]] = float(mae)
        print(f"  {names[eid]}: MAE={mae:.4f}")
    print(f"\n  Per-concept MAE breakdown:")
    concept_names = CONCEPTS
    for eid in [0,1,2]:
        per_concept = np.mean(np.abs(gt - np.array(preds[eid])), axis=0)
        best_concept = concept_names[np.argmin(per_concept)]
        print(f"  {names[eid]}: best on '{best_concept}' (MAE={np.min(per_concept):.4f})")
        results[f'{names[eid]}_best_concept'] = best_concept
        results[f'{names[eid]}_per_concept'] = per_concept.tolist()
    with open(f'{OUT}/assist09_individual.json','w') as f:
        json.dump(results, f, indent=2)

# ── GSM8K ─────────────────────────────────────────────────────────
def run_gsm8k_individual():
    print("\n=== GSM8K Individual MAE ===")
    CONCEPTS = ['Problem_Setup','Arithmetic','Multi_Step_Reasoning','Answer_Verification']
    K = 4

    problems = []
    with open(f'{BASE}/gsm8k_test.jsonl') as f:
        for line in f:
            problems.append(json.loads(line.strip()))
    problems = problems[:200]
    print(f"  Using {len(problems)} problems (subset for speed)")

    preds = {0:[], 1:[], 2:[]}
    gt_list = []

    for idx, prob in enumerate(problems):
        question = prob.get('question','')
        answer   = prob.get('answer','')
        import re as _re
        final_ans = _re.search(r'####\s*([\d\.\-]+)', answer)
        correct_answer = final_ans.group(1).strip() if final_ans else ""
        n_steps = len(_re.findall(r'<<', answer))
        step_score = min(n_steps / 5.0, 1.0)
        gt = np.array([min(step_score*1.0,1.0), min(step_score*1.0,1.0),
                       min(step_score*0.9,1.0), min(step_score*0.8,1.0)])
        history = f"Question: {question}\nCorrect Answer: {correct_answer}\nSolution steps: {n_steps}"

        print(f"  [{idx+1}/{len(problems)}] Problem {idx+1}")
        for eid in [0,1,2]:
            p = call_llm(eid, history, CONCEPTS)
            preds[eid].append(p.tolist())
        gt_list.append(gt.tolist())

    gt = np.array(gt_list)
    results = {}
    names = {0:'LLaMA-3.3-70B', 1:'GPT-4o-mini', 2:'Claude-3-Haiku'}
    print(f"\n--- GSM8K Results ({len(problems)} problems) ---")
    for eid in [0,1,2]:
        mae = np.mean(np.abs(gt - np.array(preds[eid])))
        results[names[eid]] = float(mae)
        print(f"  {names[eid]}: MAE={mae:.4f}")
    print(f"\n  Per-concept MAE breakdown:")
    concept_names = CONCEPTS
    for eid in [0,1,2]:
        per_concept = np.mean(np.abs(gt - np.array(preds[eid])), axis=0)
        best_concept = concept_names[np.argmin(per_concept)]
        print(f"  {names[eid]}: best on '{best_concept}' (MAE={np.min(per_concept):.4f})")
        results[f'{names[eid]}_best_concept'] = best_concept
        results[f'{names[eid]}_per_concept'] = per_concept.tolist()
    with open(f'{OUT}/gsm8k_individual.json','w') as f:
        json.dump(results, f, indent=2)

# ── UCI ───────────────────────────────────────────────────────────
def run_uci_individual():
    print("\n=== UCI Individual MAE ===")
    CONCEPTS = ['Study_Habits','Family_Support','School_Engagement',
                'Social_Factors','Academic_Foundation']
    K = 5

    path = f'{BASE}/student-por.csv'
    df = pd.read_csv(path, sep=';')
    print(f"  Using all {len(df)} students")

    def uci_gt(row):
        g3   = row['G3']/20.0
        stdy = min(row['studytime']/4.0, 1.0)
        fams = (row['famrel']-1)/4.0
        schl = (row['goout']-1)/4.0
        socl = (row['freetime']-1)/4.0
        return np.array([stdy, fams, 1-schl, socl, g3])

    preds = {0:[], 1:[], 2:[]}
    gt_list = []

    for idx, (_, row) in enumerate(df.iterrows()):
        gt = uci_gt(row)
        history = (f"Student: age={row['age']}, studytime={row['studytime']}, "
                   f"failures={row['failures']}, famrel={row['famrel']}, "
                   f"goout={row['goout']}, G1={row['G1']}, G2={row['G2']}")
        print(f"  [{idx+1}/{len(df)}] Student {idx+1}")
        for eid in [0,1,2]:
            p = call_llm(eid, history, CONCEPTS)
            preds[eid].append(p.tolist())
        gt_list.append(gt.tolist())

    gt = np.array(gt_list)
    results = {}
    names = {0:'LLaMA-3.3-70B', 1:'GPT-4o-mini', 2:'Claude-3-Haiku'}
    print(f"\n--- UCI Results ({len(df)} students) ---")
    for eid in [0,1,2]:
        mae = np.mean(np.abs(gt - np.array(preds[eid])))
        results[names[eid]] = float(mae)
        print(f"  {names[eid]}: MAE={mae:.4f}")
    print(f"\n  Per-concept MAE breakdown:")
    concept_names = CONCEPTS
    for eid in [0,1,2]:
        per_concept = np.mean(np.abs(gt - np.array(preds[eid])), axis=0)
        best_concept = concept_names[np.argmin(per_concept)]
        print(f"  {names[eid]}: best on '{best_concept}' (MAE={np.min(per_concept):.4f})")
        results[f'{names[eid]}_best_concept'] = best_concept
        results[f'{names[eid]}_per_concept'] = per_concept.tolist()
    with open(f'{OUT}/uci_individual.json','w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    from datetime import datetime
    print(f"Started: {datetime.now()}")
    run_assist09_individual()
    run_gsm8k_individual()
    run_uci_individual()
    print(f"\nAll done: {datetime.now()}")
