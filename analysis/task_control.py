"""Task-identity control.

Error rates differ about tenfold across tasks, so a score that only
recognises the task already gets AUROC above 0.5 when tasks are pooled.
Two controls:

  1. TaskPrior: score each test item by its task's error rate on the
     calibration split.
  2. Within-task AUROC: AUROC inside each task, averaged over tasks, where
     task identity cannot help.

Writes results/task_control.json.
Usage: python analysis/task_control.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import SCORES  # noqa: E402
from benchmark import KEYS, METHODS  # noqa: E402
from protocol import OUT_DIR, auc, load_model, splits  # noqa: E402


def within_task(err, s, task):
    v = [auc(err[task == t], s[task == t]) for t in np.unique(task)]
    return float(np.nanmean(v))


def task_prior(err, task, cal, tst):
    rate = {t: np.nanmean(err[cal][task[cal] == t]) for t in np.unique(task)}
    return np.array([rate[t] for t in task[tst]])


def benchmark_full(models, scores):
    """Full-data pooled vs within-task AUROC for every benchmark method, both
    judges, plus the task-prior baseline on test splits."""
    SP = splits()
    out = {}
    for j in ("luna", "gemini"):
        res = {m: {"pooled": [], "within": []} for m in METHODS}
        res["TaskPrior"] = {"pooled": [], "within": []}
        for k in KEYS:
            M = models[k]
            e, task = M["err"][j], M["task"]
            ok = np.isfinite(e)
            for m in METHODS:
                s = scores[f"{k}/{m}"]
                res[m]["pooled"].append(auc(e[ok], s[ok]))
                res[m]["within"].append(within_task(e[ok], s[ok], task[ok]))
            tp = []
            for cal, tst in SP:
                cal, tst = cal[ok[cal]], tst[ok[tst]]
                tp.append(auc(e[tst], task_prior(e, task, cal, tst)))
            res["TaskPrior"]["pooled"].append(float(np.mean(tp)))
            res["TaskPrior"]["within"].append(0.5)
        out[j] = {m: {"pooled": float(np.nanmean(v["pooled"])),
                      "within": float(np.nanmean(v["within"])),
                      "per_model_within": dict(zip(KEYS, v["within"]))}
                  for m, v in res.items()}
    return out


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    out = {"benchmark": benchmark_full(models, scores)}
    for j in ("luna", "gemini"):
        print(f"\n=== benchmark, judge {j}: pooled vs within-task macro ===")
        for m, v in sorted(out["benchmark"][j].items(),
                           key=lambda kv: -kv[1]["within"]):
            print(f"  {m:11s} pooled {v['pooled']:.3f}  within {v['within']:.3f}")
    p = os.path.join(OUT_DIR, "task_control.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
