# vi-uq-bench

Code for the paper *Uncertainty Quantification for Hallucination Detection in
Vietnamese LLMs: A Benchmark and Certified Selective Answering*.

The repository contains the full pipeline: generating answers with eight open
LLMs, building the NLI graphs, judging correctness with two LLM judges,
scoring eight uncertainty methods and a length control, and certifying
selective answering with Learn-then-Test. It contains **no data and no run
outputs**. The dataset is on the Hugging Face Hub with gated access
(below); everything else is produced by running the pipeline.

## Dataset

The benchmark has 1,200 Vietnamese questions, 100 for each of 12 task types.
It is at [huggingface.co/datasets/vndee/vi-uq-bench](https://huggingface.co/datasets/vndee/vi-uq-bench).
To keep it out of the training data of future models, access is gated and
approved by hand: request it on that page with your name, affiliation and
intended use. See [data/README.md](data/README.md) for where to put the file.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Generation was run with `torch==2.6.0` (CUDA 12.4) and `transformers==4.57.6`.
The analysis only needs the CPU packages at the top of `requirements.txt`.

Paths default to `data/`, `artifacts/` and `results/` inside the repository
and can be changed with `VIUQ_DATASET`, `VIUQ_ARTIFACTS` and `VIUQ_RESULTS`
(see `viuq/config.py`).

API keys are read from the environment only: `HF_TOKEN` (gated models),
`OPENAI_API_KEY` (judge J1), `GEMINI_API_KEY` (judge J2).

## Pipeline

| Step | Command | Hardware | Output (`artifacts/`) |
|---|---|---|---|
| 1. Generate | `make generate` | 1 GPU, 48 GB (A40) | `generations/` |
| 2. NLI graphs | `make nli` | GPU (about 1 h for 8 models on an A40); on Apple MPS set `BATCH=16` | `nli/` |
| 3. Judge | `make judge-j1`, `make judge-j2` | API (batch jobs) | `judge/` |
| 4. Analysis | `make paper-tables` | CPU | `results/` |

**1. Generation** (`pipeline/generate.py`).
For every question each model gives one greedy answer and 10 sampled answers.
The script stores the texts, the token log-probs and per-step entropies of
the greedy answer, and the sequence log-probs of the samples. Sampling settings and system prompts per model are in
`configs/models.tsv`; token budgets are 256, with 640 for math, 384 for
summarization and 768 for unanswerable items. Phi-4 (14B, bf16) is the largest
model and fits on a 48 GB card. Generation resumes after an interruption.

**2. NLI graphs** (`pipeline/nli_matrices.py`). Entailment and contradiction
probabilities between every pair of sampled answers, from
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (max length 512).

**3. Judging** (`pipeline/make_manifest.py`, `pipeline/judge_openai.py`,
`pipeline/judge_gemini.py`). Each greedy answer is graded against the
reference by J1 (`gpt-5.6-luna`, reasoning effort low) and J2
(`gemini-3.6-flash`, thinking budget 128) with the rubric in
`pipeline/rubric.py`. Both clients use batch APIs; run `build`, `submit`,
then `status` until done, then `fetch`.

**4. Analysis** (`analysis/`). Run `benchmark.py` first: it writes the
per-question scores that every other script reads, and
`fig_risk_coverage.py` needs `cert_final.py`; `make paper-tables` runs them
in order. Shared splits, loaders and the certificate are in `protocol.py`.

## Where each result comes from

| Paper result | Script | Output (`results/`) |
|---|---|---|
| Table 2, pooled AUROC (both judges) | `benchmark.py` | `benchmark.json` |
| Table 2, within-task AUROC and TaskPrior | `task_control.py` | `task_control.json` |
| Pooled gaps and per-task intervals | `bootstrap_ci.py` | `bootstrap_ci.json` |
| Within-task gaps and their intervals | `within_task_bench.py` | `within_task_bench.json` |
| Table 3, per-task AUROC; judge agreement (Sec. 5.1) | `per_task.py` | `per_task.json` |
| Table 4, exceedance rates, threshold transfer, per-task error, Mondrian at delta/12 | `cert_final.py` | `cert_final.json` |
| Figure 1, risk-coverage curves | `fig_risk_coverage.py` | `fig_risk_coverage.pdf` |
| Certification procedures and truncation (Sec. 5.2-5.3) | `cert_sensitivity.py` | `cert_sensitivity.json` |
| Pair-weighted, cross-task and standardized AUROC; non-empty certificates; complement check; transfer by direction; per-task oracle; Mondrian at delta; first-level check | `checks.py` | `checks.json` |
| 500-split rates with intervals: self-calibrated exceedance on the test split and on the full question pool, transfer by direction, per-task exceedance | `split_robustness.py` | `split_robustness.json` |
| Within-task AUROC without sparse (model, task) cells; per-task oracle for KLE, MSP and MTE; raw vs. retention-level transfer; per-task certification with fewer levels and by task group | `review_checks.py` | `review_checks.json` |
| Bonferroni vs. fixed-sequence exceedance (500 splits); transfer measured on the full pool; two-level (questions x splits) intervals; TaskPrior + KLE and per-task score choice; certificates built on MSP / MTE | `review_checks2.py` | `review_checks2.json` |

All splits are by question and shared across models: 20 calibration/test
splits of 500/700 questions (500 in `split_robustness.py`), with 200 of the 700 test
questions kept unlabelled to fix the certification thresholds.

## Code layout

```
configs/models.tsv   models, sampling settings, system prompts
pipeline/            generation, NLI graphs, judges
analysis/            scoring, certification, checks, figure
viuq/config.py       paths
viuq/fingerprint.py  dataset hash, checked before generations are reused
viuq/lmpg/           the lm-polygraph estimators used (KLE, DegMat, NumSemSets, SE)
scripts/             generation runner
```

The semantic estimators are lm-polygraph's own classes, vendored in
`viuq/lmpg/` (MIT) so the scores do not depend on the installed release.

## Citation

```bibtex
@inproceedings{huynh2027viuq,
  title     = {Uncertainty Quantification for Hallucination Detection in
               Vietnamese {LLMs}: A Benchmark and Certified Selective Answering},
  author    = {Huynh, Duy V. and Quang-Hung, Nguyen and Nguyen, Hien T.},
  booktitle = {<venue>},
  year      = {2027}
}
```

## License

Code: MIT (see `LICENSE`). The dataset is not part of this repository and is
shared separately under its own terms.
