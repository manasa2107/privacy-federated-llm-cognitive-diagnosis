# Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis

[![EMNLP 2026 Findings](https://img.shields.io/badge/EMNLP_2026-Findings-blue)](https://2026.emnlp.org)
[![License: MIT](https://img.shields.io/badge/Code_License-MIT-yellow.svg)](LICENSE)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper_License-CC_BY_4.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

Official implementation of our **EMNLP 2026 Findings** paper:

> **Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis**
> Yagna Manasa Boyapati, Chong Yu, Tianyu Jiang, Justin Zhan
> Department of Computer Science, University of Cincinnati

## Framework

Our framework combines four key components:
- **Heterogeneous LLMs**: LLaMA-3.3-70B (Groq API), GPT-4o-mini (OpenAI API), Claude-3-Haiku (Anthropic API)
- **Federated Inference**: Prediction-level aggregation — no model weights or gradients shared
- **Local Differential Privacy**: Laplace mechanism with ε ∈ {0.5, 1.0, 2.0, ∞}
- **Residual Correction**: Calibration bias correction for heterogeneous LLM outputs

## Results

| Dataset  | Domain   | Baseline MAE | Fed MAE (ε=2.0) | Improv. | Privacy Cost |
|----------|----------|-------------|-----------------|---------|--------------|
| ASSIST09 | Math     | 0.2410      | 0.2068          | 14.19%  | 0.19%        |
| GSM8K    | Word Prob| 0.2328      | 0.2156          | 7.39%   | 0.27%        |
| UCI      | Holistic | 0.1132      | 0.0969          | 14.40%  | 0.94%        |
| **Avg**  | --       | 0.1957      | 0.1731          | **11.99%** | **0.46%** |

## Installation

```bash
pip install -r requirements.txt
```

## API Keys

Set the following environment variables before running:

```bash
export GROQ_API_KEY="your_groq_key"
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## Datasets

Download and place in the `data/` directory:

| Dataset | Source | File |
|---------|--------|------|
| ASSIST09 | [ASSISTments](https://sites.google.com/site/assistmentsdata/) | `skill_builder_data.csv` |
| GSM8K    | [HuggingFace](https://huggingface.co/datasets/openai/gsm8k) | `gsm8k_test.jsonl` |
| UCI Student Performance | [UCI ML Repository](https://archive.ics.uci.edu/dataset/320/student+performance) | `student-por.csv` |

## Repository Structure

```
├── src/
│   ├── federated_simple.py
│   ├── federated_local_dp.py
│   ├── simple_llm.py
│   ├── run_experiments_assist09.py
│   ├── run_experiments_gsm8k.py
│   ├── run_experiments_uci_student.py
│   ├── run_experiments_privacy.py
│   ├── run_experiments.py
│   ├── run_dp_fl_homogeneous.py
│   ├── run_individual_llm_mae.py
│   ├── run_new_baselines.py
│   ├── run_assist09_full.py
│   ├── run_gsm8k_full.py
│   └── run_uci_full.py
├── scripts/
│   ├── run_assist09_job.sh
│   ├── run_gsm8k_job.sh
│   ├── run_uci_job.sh
│   ├── run_privacy_job.sh
│   └── setup_env.sh
├── baselines_cdm/
│   ├── irt_dina_models.py
│   ├── neuralcd_model.py
│   ├── run_baseline_comparison.py
│   ├── run_neuralcd_only.py
│   ├── prepare_assist09_data.py
│   ├── prepare_assist09_custom.py
│   ├── prepare_full_assist09.py
│   ├── run_baseline_job.slurm
│   └── data/
│       └── metadata.json
├── data/
│   └── sample/
│       └── README.md              # Dataset download instructions
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
## Running Experiments

```bash
python src/run_experiments_assist09.py
python src/run_experiments_gsm8k.py
python src/run_experiments_uci_student.py
```

### Privacy Tradeoff Analysis

```bash
python src/run_experiments_privacy.py
```

### Full Dataset on HPC Cluster (SLURM)

```bash
sbatch scripts/run_assist09_job.sh
sbatch scripts/run_gsm8k_job.sh
sbatch scripts/run_uci_job.sh
```

### CDM Baselines (IRT, DINA, NeuralCD)

```bash
python baselines_cdm/run_baseline_comparison.py
```

## Infrastructure

Experiments were conducted on the University of Cincinnati ARCC2 GPU cluster using Tesla V100S GPUs (32GB). Full-scale experiments required approximately 24 GPU-hours across all datasets and privacy configurations.

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{boyapati2026privacy,
  title={Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis},
  author={Boyapati, Yagna Manasa and Yu, Chong and Jiang, Tianyu and Zhan, Justin},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

## License

- **Code**: MIT License — see [LICENSE](LICENSE)
- **Paper**: Creative Commons Attribution 4.0 (CC BY 4.0) — as published in ACL Anthology

## Contact

- Yagna Manasa Boyapati — boyapaya@mail.uc.edu
- Chong Yu — yuc5@ucmail.uc.edu
- Tianyu Jiang — tianyu.jiang@uc.edu
- Justin Zhan — zhanjt@ucmail.uc.edu
