# vi-uq-bench

Code for the paper *Uncertainty Quantification for Hallucination Detection in
Vietnamese LLMs: A Benchmark and Certified Selective Answering*.

The repository contains the full pipeline: generating answers with eight open
LLMs, building the NLI graphs, judging correctness with two LLM judges,
scoring nine uncertainty methods, certifying selective answering with
Learn-then-Test, and training hidden-state probes. It contains **no data and
no run outputs**. The dataset is available on request (below); everything
else is produced by running the pipeline.

## Dataset

The benchmark has 1,200 Vietnamese questions, 100 for each of 12 task types.
To keep it out of the training data of future models, it is not posted
online. Email `duyhv@pm.me` with your name, affiliation and intended use
to get a copy, and see [data/README.md](data/README.md) for where to put it.

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

**1. Generation** (`pipeline/generate.py`, `pipeline/attention_pass.py`).
For every question each model gives one greedy answer and 10 sampled answers.
The script stores the texts, token log-probs, per-step entropies and
mean-pooled hidden states of every layer; a second pass in eager attention
computes AttentionScore. Sampling settings and system prompts per model are in
`configs/models.tsv`; token budgets are 256, with 640 for math, 384 for
summarization and 768 for unanswerable items. Phi-4 (14B, bf16) is the largest
model and fits on a 48 GB card. Both steps resume after an interruption.

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
per-question scores that every other script reads. `cert_final.py` needs
`cert_sensitivity.py`, and `fig_risk_coverage.py` needs `cert_final.py`;
`make paper-tables` runs them in order.

## Where each result comes from

| Paper result | Script | Output (`results/`) |
|---|---|---|
| Benchmark, pooled AUROC (both judges) | `benchmark.py` | `benchmark.json` |
| Pooled gaps and per-task intervals | `bootstrap_ci.py` | `bootstrap_ci.json` |
| Within-task AUROC and its intervals | `within_task_bench.py` | `within_task_bench.json` |
| TaskPrior reference, pooled vs. within task | `task_control.py` | `task_control.json` |
| Per-task AUROC, judge agreement, error rates | `tiers.py` | `tiers.json` |
| Probes: layer sweep, leave-one-task-out, calibration | `tiers.py` | `tiers.json` |
| Certified coverage, threshold transfer, per-task failures | `cert_final.py` | `cert_final.json` |
| Procedure and truncation checks | `cert_sensitivity.py` | `cert_sensitivity.json` |
| Risk-coverage figure | `fig_risk_coverage.py` | `fig_risk_coverage.pdf` |
| Probe vs. supervised baseline | `supervised_baseline.py`, `probe_control.py` | `supervised_baseline.json`, `probe_control.json` |
| Task x score interaction baseline | `sup_interaction.py` | `sup_interaction.json` |
| Label budget for certification | `budget_control.py` | `budget_control.json` |
| Probe transfer across models | `lineage_within.py` | `lineage_within.json` |

All splits are by question and shared across models: 20 calibration/test
splits of 500/700 questions (5 for the probes), with 200 of the 700 test
questions kept unlabelled to fix the certification thresholds.

## Code layout

```
configs/models.tsv   models, sampling settings, system prompts
pipeline/            generation, attention pass, NLI graphs, judges
analysis/            scoring, certification, probes, figure
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
