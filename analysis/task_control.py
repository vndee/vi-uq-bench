"""Task-identity control.

Error rates differ about tenfold across tasks, so a score that only
recognises the task already gets AUROC above 0.5 when tasks are pooled.
Two controls:

  1. TaskPrior: score each test item by its task's error rate on the
     calibration split.
  2. Within-task AUROC: AUROC inside each task, averaged over tasks, where
     task identity cannot help.

Probes follow the tiers.py protocol; results are given for 5 and 20 splits.

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
from benchmark import DISPLAY, KEYS, N  # noqa: E402
from tiers import (OUT_DIR, PROBE_DEPTH, auc, fit_predict,  # noqa: E402
                        load_hidden, load_model, splits)

METHODS = ["KLE", "DegMat", "SE", "NumSets", "MTE", "PPL", "MC-SE", "MSP",
           "MeanLength", "Attn"]


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


def probes(models, scores, n_splits):
    SP = splits()[:n_splits]
    out = {}
    for k in KEYS:
        M = models[k]
        e, task = M["err"]["luna"], M["task"]
        ok = np.isfinite(e)
        Hans, Hpr = load_hidden(k)
        L = int(round(PROBE_DEPTH * (Hans.shape[1] - 1)))
        acc = {f"{n}_{w}": [] for n in ("probe", "prompt", "kle", "mte", "len",
                                        "taskprior")
               for w in ("pooled", "within")}
        for cal, tst in SP:
            cal, tst = cal[ok[cal]], tst[ok[tst]]
            et, tt = e[tst], task[tst]
            sc = {"probe": fit_predict(Hans[cal, L], e[cal], Hans[tst, L]),
                  "prompt": fit_predict(Hpr[cal, L], e[cal], Hpr[tst, L]),
                  "kle": scores[f"{k}/KLE"][tst],
                  "mte": scores[f"{k}/MTE"][tst],
                  "len": scores[f"{k}/MeanLength"][tst],
                  "taskprior": task_prior(e, task, cal, tst)}
            for n, s in sc.items():
                acc[f"{n}_pooled"].append(auc(et, s))
                acc[f"{n}_within"].append(within_task(et, s, tt)
                                          if n != "taskprior" else 0.5)
        out[k] = {n: float(np.nanmean(v)) for n, v in acc.items()}
        r = out[k]
        print(f"  {DISPLAY[k]:10s} pooled probe {r['probe_pooled']:.3f} "
              f"prompt {r['prompt_pooled']:.3f} KLE {r['kle_pooled']:.3f} "
              f"taskprior {r['taskprior_pooled']:.3f} | within probe "
              f"{r['probe_within']:.3f} prompt {r['prompt_within']:.3f} "
              f"KLE {r['kle_within']:.3f} MTE {r['mte_within']:.3f}", flush=True)
    macro = {n: float(np.mean([out[k][n] for k in KEYS]))
             for n in out[KEYS[0]]}
    wins = {f"probe>kle_{w}": int(sum(out[k][f"probe_{w}"] > out[k][f"kle_{w}"]
                                      for k in KEYS)) for w in ("pooled", "within")}
    wins.update({f"prompt>kle_{w}": int(sum(out[k][f"prompt_{w}"] >
                                            out[k][f"kle_{w}"] for k in KEYS))
                 for w in ("pooled", "within")})
    return {"per_model": out, "macro": macro, "wins": wins}


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    out = {"benchmark": benchmark_full(models, scores)}
    for j in ("luna", "gemini"):
        print(f"\n=== benchmark, judge {j}: pooled vs within-task macro ===")
        for m, v in sorted(out["benchmark"][j].items(),
                           key=lambda kv: -kv[1]["within"]):
            print(f"  {m:11s} pooled {v['pooled']:.3f}  within {v['within']:.3f}")
    for ns in (5, 20):
        print(f"\n=== probes, {ns} splits (judge luna, test split) ===")
        out[f"probes_{ns}"] = probes(models, scores, ns)
        print("macro:", json.dumps({k: round(v, 3) for k, v in
                                    out[f"probes_{ns}"]["macro"].items()}))
        print("wins:", out[f"probes_{ns}"]["wins"])
    p = os.path.join(OUT_DIR, "task_control.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
