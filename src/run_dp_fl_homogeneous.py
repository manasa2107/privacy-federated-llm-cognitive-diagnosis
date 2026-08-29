import numpy as np
import pandas as pd
import json, os, re, time
from datetime import datetime
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY)

BASE = '/N/lustre/project/proj-606/m_research'
OUT  = f'{BASE}/results_dp_fl_homogeneous'
CKPT = f'{BASE}/checkpoints_dp_fl_homo'
os.makedirs(OUT,  exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

ALPHA = 0.3
EPS   = 2.0
M     = 3
CHECKPOINT_EVERY = 50

def call_llama(prompt_text, concepts):
    K = len(concepts)
    base = (f"Given student information: {prompt_text}\n"
            f"And knowledge concepts: {concepts}\n"
            f"Estimate knowledge mastery as a probability vector [0-1] "
            f"for EXACTLY {K} concepts.\n"
            f"Return ONLY a Python list with EXACTLY {K} numbers, nothing else.\n"
            f"Example: {[0.5]*K}")
    for attempt in range(10):
        try:
            r = groq_client.chat.completions.create(
                messages=[{"role":"user","content":"You are an educational assessor.\n"+base}],
                model="llama-3.3-70b-versatile", temperature=0.7)
            text = r.choices[0].message.content
            text = text.replace("```python","").replace("```","").strip()
            m = re.search(r'\[[\d\.,\s]+\]', text)
            ks = eval(m.group() if m else text.strip())
            if len(ks) != K:
                ks = (list(ks) + [0.5]*K)[:K]
            return np.clip(np.array(ks, dtype=float), 0, 1)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"  Groq rate limit, waiting 130s...")
                time.sleep(130)
            else:
                print(f"  Error attempt {attempt+1}: {e}")
                time.sleep(2)
    return np.array([0.5]*K)

def dp_fl_homogeneous(prompt_text, concepts, residuals):
    K = len(concepts)
    preds = []
    for _ in range(M):
        p = call_llama(prompt_text, concepts)
        noise = np.random.laplace(0, K/EPS, K)
        preds.append(np.clip(p + noise, 0, 1))
    corrected = [preds[j] - ALPHA * residuals[j] for j in range(M)]
    return np.clip(np.mean(corrected, axis=0), 0, 1)

def update_residuals(all_preds_list, K):
    if len(all_preds_list) < 10:
        return [np.zeros(K)]*M
    arr = np.array(all_preds_list)
    global_mean = arr.mean(axis=(0,1))
    return [arr[:,j,:].mean(axis=0) - global_mean for j in range(M)]

def save_ckpt(dataset, idx, data):
    path = f"{CKPT}/{dataset}_ckpt_{idx}.json"
    with open(path,'w') as f:
        json.dump({k:(v.tolist() if isinstance(v,np.ndarray) else v)
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

def save_results(dataset, gt_list, single_preds, homo_preds):
    gt = np.array(gt_list)
    single_mae = float(np.mean(np.abs(gt - np.array(single_preds))))
    homo_mae   = float(np.mean(np.abs(gt - np.array(homo_preds))))
    improvement = (single_mae - homo_mae) / single_mae * 100
    result = {
        "dataset": dataset,
        "n_students": len(gt_list),
        "single_llm_mae": single_mae,
        "dp_fl_homogeneous_mae": homo_mae,
        "improvement_over_single_pct": improvement,
        "epsilon": EPS, "M": M
    }
    out_path = f"{OUT}/{dataset}_dp_fl_homo.json"
    with open(out_path,'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n{'='*50}")
    print(f"RESULTS -- {dataset.upper()}")
    print(f"  Single LLM (LLaMA):    MAE = {single_mae:.4f}")
    print(f"  DP-FL Homogeneous:     MAE = {homo_mae:.4f}  ({improvement:+.2f}%)")
    print(f"  Saved -> {out_path}")
    print(f"{'='*50}\n")
    return result

def run_assist09():
    print("\n"+"="*50+"\nDP-FL HOMOGENEOUS -- ASSIST09\n"+"="*50)
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
    print(f"  Students: {len(students)}")
    ckpt, start = load_ckpt('assist09')
    single_preds   = ckpt['single_preds']   if ckpt else []
    homo_preds     = ckpt['homo_preds']     if ckpt else []
    gt_list        = ckpt['gt_list']        if ckpt else []
    all_preds_list = ckpt.get('all_preds_list',[]) if ckpt else []
    residuals      = [np.zeros(K)]*M
    t0 = time.time()
    for idx, sid in enumerate(students[start:], start=start):
        sdf = df[df['user_id']==sid]
        gt  = np.array([sdf[sdf['concept_id']==c]['correct'].mean()
                        if len(sdf[sdf['concept_id']==c])>0 else 0.5 for c in range(K)])
        history = '\n'.join([f"* {r['skill_name']}: {'Correct' if r['correct']==1 else 'Incorrect'}"
                             for _,r in sdf.tail(30).iterrows()])
        print(f"  [{idx+1}/{len(students)}] Student {sid}  {(time.time()-t0)/60:.1f}m")
        sp = call_llama(history, CONCEPTS)
        residuals = update_residuals(all_preds_list, K)
        hp = dp_fl_homogeneous(history, CONCEPTS, residuals)
        three_preds = [call_llama(history, CONCEPTS) for _ in range(M)]
        all_preds_list.append([p.tolist() for p in three_preds])
        single_preds.append(sp.tolist())
        homo_preds.append(hp.tolist())
        gt_list.append(gt.tolist())
        if (idx+1) % CHECKPOINT_EVERY == 0:
            save_ckpt('assist09', idx+1, {'last_idx':idx+1,
                'single_preds':single_preds,'homo_preds':homo_preds,
                'gt_list':gt_list,'all_preds_list':all_preds_list})
    return save_results('assist09', gt_list, single_preds, homo_preds)

def run_gsm8k():
    print("\n"+"="*50+"\nDP-FL HOMOGENEOUS -- GSM8K\n"+"="*50)
    CONCEPTS = ['Problem_Setup','Arithmetic','Multi_Step_Reasoning','Answer_Verification']
    K = 4
    path = f'{BASE}/gsm8k_test.jsonl'
    if not os.path.exists(path):
        print(f"  Not found: {path}"); return None
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line.strip()))
    print(f"  Problems: {len(problems)}")
    ckpt, start = load_ckpt('gsm8k')
    single_preds   = ckpt['single_preds']   if ckpt else []
    homo_preds     = ckpt['homo_preds']     if ckpt else []
    gt_list        = ckpt['gt_list']        if ckpt else []
    all_preds_list = ckpt.get('all_preds_list',[]) if ckpt else []
    residuals      = [np.zeros(K)]*M
    t0 = time.time()
    for idx, prob in enumerate(problems[start:], start=start):
        question = prob.get('question','')
        answer   = prob.get('answer','')
        final_ans = re.search(r'####\s*([\d\.\-]+)', answer)
        correct_ans = final_ans.group(1).strip() if final_ans else ""
        n_steps = len(re.findall(r'<<', answer))
        step_score = min(n_steps/5.0, 1.0)
        gt = np.array([min(step_score*1.0,1.0),min(step_score*1.0,1.0),
                       min(step_score*0.9,1.0),min(step_score*0.8,1.0)])
        history = f"Question: {question}\nCorrect Answer: {correct_ans}\nSteps: {n_steps}"
        print(f"  [{idx+1}/{len(problems)}] Problem {idx+1}  {(time.time()-t0)/60:.1f}m")
        sp = call_llama(history, CONCEPTS)
        residuals = update_residuals(all_preds_list, K)
        hp = dp_fl_homogeneous(history, CONCEPTS, residuals)
        three_preds = [call_llama(history, CONCEPTS) for _ in range(M)]
        all_preds_list.append([p.tolist() for p in three_preds])
        single_preds.append(sp.tolist())
        homo_preds.append(hp.tolist())
        gt_list.append(gt.tolist())
        if (idx+1) % CHECKPOINT_EVERY == 0:
            save_ckpt('gsm8k', idx+1, {'last_idx':idx+1,
                'single_preds':single_preds,'homo_preds':homo_preds,
                'gt_list':gt_list,'all_preds_list':all_preds_list})
    return save_results('gsm8k', gt_list, single_preds, homo_preds)

def run_uci():
    print("\n"+"="*50+"\nDP-FL HOMOGENEOUS -- UCI\n"+"="*50)
    CONCEPTS = ['Study_Habits','Family_Support','School_Engagement',
                'Social_Factors','Academic_Foundation']
    K = 5
    path = f'{BASE}/student-por.csv'
    if not os.path.exists(path):
        path = f'{BASE}/student-mat.csv'
    if not os.path.exists(path):
        print("  UCI not found"); return None
    df = pd.read_csv(path, sep=';')
    print(f"  Students: {len(df)}")
    def uci_gt(row):
        return np.array([min(row['studytime']/4.0,1.0),(row['famrel']-1)/4.0,
                         1-(row['goout']-1)/4.0,(row['freetime']-1)/4.0,row['G3']/20.0])
    ckpt, start = load_ckpt('uci')
    single_preds   = ckpt['single_preds']   if ckpt else []
    homo_preds     = ckpt['homo_preds']     if ckpt else []
    gt_list        = ckpt['gt_list']        if ckpt else []
    all_preds_list = ckpt.get('all_preds_list',[]) if ckpt else []
    residuals      = [np.zeros(K)]*M
    t0 = time.time()
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx < start: continue
        gt = uci_gt(row)
        history = (f"Student: age={row['age']}, studytime={row['studytime']}, "
                   f"failures={row['failures']}, famrel={row['famrel']}, "
                   f"goout={row['goout']}, G1={row['G1']}, G2={row['G2']}")
        print(f"  [{idx+1}/{len(df)}] Student {idx+1}  {(time.time()-t0)/60:.1f}m")
        sp = call_llama(history, CONCEPTS)
        residuals = update_residuals(all_preds_list, K)
        hp = dp_fl_homogeneous(history, CONCEPTS, residuals)
        three_preds = [call_llama(history, CONCEPTS) for _ in range(M)]
        all_preds_list.append([p.tolist() for p in three_preds])
        single_preds.append(sp.tolist())
        homo_preds.append(hp.tolist())
        gt_list.append(gt.tolist())
        if (idx+1) % CHECKPOINT_EVERY == 0:
            save_ckpt('uci', idx+1, {'last_idx':idx+1,
                'single_preds':single_preds,'homo_preds':homo_preds,
                'gt_list':gt_list,'all_preds_list':all_preds_list})
    return save_results('uci', gt_list, single_preds, homo_preds)

if __name__ == '__main__':
    print(f"Started: {datetime.now()}")
    r1 = run_assist09()
    r2 = run_gsm8k()
    r3 = run_uci()
    print("\n=== FINAL SUMMARY ===")
    for r, ds in [(r1,'ASSIST09'),(r2,'GSM8K'),(r3,'UCI')]:
        if r:
            print(f"  {ds}: single={r['single_llm_mae']:.4f}  "
                  f"homo={r['dp_fl_homogeneous_mae']:.4f}  "
                  f"({r['improvement_over_single_pct']:+.2f}%)")
    print(f"Done: {datetime.now()}")
