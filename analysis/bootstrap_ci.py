"""Paired bootstrap intervals for the pooled benchmark gaps.

All eight models answer the same 1,200 questions, so a difference in macro
AUROC is resampled by drawing questions jointly for all models, recomputing
each model's AUROC and averaging. Per-task gaps resample that task's 100
questions the same way.

Writes results/bootstrap_ci.json.
Usage: python analysis/bootstrap_ci.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import SCORES  # noqa: E402
from benchmark import KEYS, N  # noqa: E402
from tiers import OUT_DIR, auc, load_model  # noqa: E402

B = 2000
PAIRS = [("KLE", "DegMat"), ("KLE", "MTE"), ("KLE", "MeanLength"),
         ("KLE", "SE"), ("DegMat", "MTE")]
TASK_PAIRS = [("true_false", "MTE", "KLE"), ("information_retrieval", "MTE", "KLE"),
              ("sentiment_classification", "MSP", "KLE"), ("ner", "MSP", "KLE"),
              ("commonsense", "KLE", "MTE"), ("factual_qa", "KLE", "MTE"),
              ("multiple_choice", "KLE", "MTE"), ("math_reasoning", "KLE", "MTE"),
              ("unanswerable", "KLE", "MTE"), ("multihop_reasoning", "KLE", "MTE"),
              ("translation", "DegMat", "KLE"), ("summarization", "MeanLength", "KLE")]


def macro_diff(models, scores, a, b, j, idx):
    v = []
    for k in KEYS:
        e = models[k]["err"][j][idx]
        v.append(auc(e, scores[f"{k}/{a}"][idx]) - auc(e, scores[f"{k}/{b}"][idx]))
    return 100 * float(np.nanmean(v))


def ci(models, scores, a, b, j, pool, rng):
    point = macro_diff(models, scores, a, b, j, pool)
    bs = np.array([macro_diff(models, scores, a, b, j,
                              rng.choice(pool, len(pool), replace=True))
                   for _ in range(B)])
    return {"delta": point, "lo": float(np.nanquantile(bs, 0.025)),
            "hi": float(np.nanquantile(bs, 0.975)),
            "p_le0": float(np.mean(bs <= 0))}


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    task = models[KEYS[0]]["task"]
    rng = np.random.default_rng(0)
    out = {"macro": {}, "per_task": {}}
    allq = np.arange(N)
    for j in ("luna", "gemini"):
        for a, b in PAIRS:
            r = ci(models, scores, a, b, j, allq, rng)
            out["macro"][f"{j}:{a}-{b}"] = r
            print(f"{j:6s} {a:>7s}-{b:<10s} {r['delta']:+5.1f} "
                  f"[{r['lo']:+5.1f},{r['hi']:+5.1f}] p={r['p_le0']:.4f}", flush=True)
    for t, a, b in TASK_PAIRS:
        r = ci(models, scores, a, b, "luna", np.where(task == t)[0], rng)
        out["per_task"][f"{t}:{a}-{b}"] = r
        print(f"luna   {t:26s} {a}-{b}: {r['delta']:+5.1f} "
              f"[{r['lo']:+5.1f},{r['hi']:+5.1f}] p={r['p_le0']:.4f}", flush=True)
    p = os.path.join(OUT_DIR, "bootstrap_ci.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
