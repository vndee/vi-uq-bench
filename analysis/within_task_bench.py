"""Within-task AUROC and paired bootstrap intervals for its gaps.

Within-task AUROC is computed inside each of the 12 tasks, averaged over
tasks, then over models. Intervals for the gaps between KLE and the next
scores resample questions within task, jointly across the eight models.

Writes results/within_task_bench.json.
Usage: python analysis/within_task_bench.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import SCORES  # noqa: E402
from benchmark import KEYS  # noqa: E402
from probe_control import fauc  # noqa: E402
from task_control import METHODS  # noqa: E402
from tiers import OUT_DIR, load_model  # noqa: E402

B = 2000
PAIRS = (("KLE", "SE"), ("KLE", "MTE"), ("KLE", "DegMat"), ("KLE", "MSP"),
         ("KLE", "MeanLength"))


def within(e, s, task, tasks, idx):
    return float(np.nanmean([fauc(e[i], s[i]) for i in
                             (idx[task[idx] == t] for t in tasks)]))


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    task = models[KEYS[0]]["task"]
    tasks = np.unique(task)
    groups = [np.where(task == t)[0] for t in tasks]
    full = np.arange(len(task))
    out = {}
    for j in ("luna", "gemini"):
        S = {m: {k: scores[f"{k}/{m}"].astype(float) for k in KEYS} for m in METHODS}
        E = {k: models[k]["err"][j] for k in KEYS}
        per = {m: {k: within(E[k], S[m][k], task, tasks, full) for k in KEYS}
               for m in METHODS}
        macro = {m: float(np.mean(list(per[m].values()))) for m in METHODS}
        rank = sorted(METHODS, key=lambda m: -macro[m])
        rng = np.random.default_rng(0)
        bs = {f"{a}-{b}": [] for a, b in PAIRS}
        for _ in range(B):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            for a, b in PAIRS:
                d = [within(E[k], S[a][k], task, tasks, idx)
                     - within(E[k], S[b][k], task, tasks, idx) for k in KEYS]
                bs[f"{a}-{b}"].append(100 * np.mean(d))
        gaps = {}
        for a, b in PAIRS:
            v = np.array(bs[f"{a}-{b}"])
            gaps[f"{a}-{b}"] = {"delta": 100 * (macro[a] - macro[b]),
                                "lo": float(np.quantile(v, .025)),
                                "hi": float(np.quantile(v, .975)),
                                "models_a_ahead": int(sum(per[a][k] > per[b][k]
                                                          for k in KEYS))}
        out[j] = {"per_model": per, "macro": macro, "rank": rank, "gaps": gaps}
        print(f"=== {j}: " + "  ".join(f"{m} {macro[m]:.3f}" for m in rank))
        for p, g in gaps.items():
            print(f"   {p:15s} {g['delta']:+.1f} [{g['lo']:+.1f},{g['hi']:+.1f}] "
                  f"KLE ahead on {g['models_a_ahead']}/8", flush=True)
    p = os.path.join(OUT_DIR, "within_task_bench.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
