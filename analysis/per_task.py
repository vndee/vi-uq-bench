"""Per-task AUROC (Table 3) and agreement between the two judges.

  per_task_benchmark  macro-over-models AUROC per task for every score (J1),
                      with each task's error rate
  judge_agreement     J1-J2 agreement and Cohen's kappa, pooled, per model and
                      per task, and the share of disagreements where J2 is the
                      lenient one
  single_class        share of (model, task, test split) cells that contain
                      only correct or only wrong answers

Writes results/per_task.json.
Usage: python analysis/per_task.py
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


def per_task_benchmark(models, scores):
    tasks = sorted(set(next(iter(models.values()))["task"]))
    out = {}
    for t in tasks:
        out[t] = {"err_rate": float(np.mean([np.nanmean(M["err"]["luna"][M["task"] == t])
                                             for M in models.values()]))}
        for m in METHODS:
            v = []
            for k, M in models.items():
                sel = (M["task"] == t) & np.isfinite(M["err"]["luna"])
                v.append(auc(M["err"]["luna"][sel], scores[f"{k}/{m}"][sel]))
            out[t][m] = float(np.nanmean(v))
    return out


def judge_agreement(models):
    def kappa(a, b):
        po = np.mean(a == b)
        pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
        return float((po - pe) / (1 - pe))
    A, Bj, T = [], [], []
    per_model = {}
    for k, M in models.items():
        ok = np.isfinite(M["err"]["luna"]) & np.isfinite(M["err"]["gemini"])
        a, b = M["err"]["luna"][ok], M["err"]["gemini"][ok]
        per_model[k] = {"agree": float(np.mean(a == b)), "kappa": kappa(a, b),
                        "correct_luna": float(1 - a.mean()),
                        "correct_gemini": float(1 - b.mean())}
        A.append(a)
        Bj.append(b)
        T.append(M["task"][ok])
    A, Bj, T = map(np.concatenate, (A, Bj, T))
    per_task = {t: {"agree": float(np.mean(A[T == t] == Bj[T == t])),
                    "kappa": kappa(A[T == t], Bj[T == t]),
                    "gemini_lenient_share": float(np.mean(
                        (A[T == t] == 1)[A[T == t] != Bj[T == t]]))
                    if np.any(A[T == t] != Bj[T == t]) else None}
                for t in sorted(set(T))}
    dis = A != Bj
    return {"pooled": {"n": int(len(A)), "agree": float(np.mean(A == Bj)),
                       "kappa": kappa(A, Bj), "n_disagree": int(dis.sum()),
                       "gemini_lenient_share": float(np.mean(A[dis] == 1))},
            "per_model": per_model, "per_task": per_task}


def single_class(models, SP):
    v = []
    for M in models.values():
        e, task = M["err"]["luna"], M["task"]
        for _, tst in SP:
            for t in np.unique(task):
                m = tst[task[tst] == t]
                v.append(len(np.unique(e[m][np.isfinite(e[m])])) < 2)
    return float(np.mean(v))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    out = {"per_task_benchmark": per_task_benchmark(models, scores),
           "judge_agreement": judge_agreement(models),
           "single_class_test_cells": single_class(models, splits())}
    print("judges:", json.dumps(out["judge_agreement"]["pooled"]))
    for t, v in out["per_task_benchmark"].items():
        print(f"  {t:26s} err {v['err_rate']:.2f} KLE {v['KLE']:.3f} "
              f"DegMat {v['DegMat']:.3f} MTE {v['MTE']:.3f} MSP {v['MSP']:.3f} "
              f"len {v['MeanLength']:.3f}  "
              f"judge-agree {out['judge_agreement']['per_task'][t]['agree']:.3f}")
    print("single-class test cells:", round(out["single_class_test_cells"], 4))
    p = os.path.join(OUT_DIR, "per_task.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
