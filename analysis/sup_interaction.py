"""Task x score interaction baseline.

In uq_tn+task (supervised_baseline.py) task identity only shifts the
intercept, so within a task the scores share one set of weights. This arm
adds every task x score product, giving per-task weights with shared
regularisation, to check whether the probe's within-task lead is just
task-specific score selection. Same protocol as supervised_baseline.py.

Writes results/sup_interaction.json.
Usage: python analysis/sup_interaction.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import SCORES  # noqa: E402
from benchmark import DISPLAY, KEYS  # noqa: E402
from supervised_baseline import (boot_within, impute, onehot,  # noqa: E402
                                      score_matrix, task_norm)
from task_control import within_task  # noqa: E402
from tiers import (N_PROBE_SPLITS, OUT_DIR, PROBE_DEPTH, auc,  # noqa: E402
                        fit_predict, load_hidden, load_model, splits)


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits()[:N_PROBE_SPLITS]
    arms = ("uq_tn+task", "uq_int+task", "probe")
    per, preds = {}, {}
    for k in KEYS:
        M = models[k]
        e, task = M["err"]["luna"], M["task"]
        ok = np.isfinite(e)
        Hans, _ = load_hidden(k)
        H = Hans[:, int(round(PROBE_DEPTH * (Hans.shape[1] - 1)))].astype(np.float32)
        del Hans
        S, T = score_matrix(k, scores), onehot(task)
        res = {a: {"pooled": [], "within": []} for a in arms}
        preds[k] = {a: [] for a in arms}
        for cal, tst in SP:
            cal, tst = cal[ok[cal]], tst[ok[tst]]
            Stn = task_norm(impute(S, cal), task, cal)
            inter = (Stn[:, :, None] * T[:, None, :]).reshape(len(task), -1)
            feats = {"uq_tn+task": np.hstack([Stn, T]),
                     "uq_int+task": np.hstack([Stn, inter, T]), "probe": H}
            for a in arms:
                p = fit_predict(feats[a][cal], e[cal], feats[a][tst])
                res[a]["pooled"].append(auc(e[tst], p))
                res[a]["within"].append(within_task(e[tst], p, task[tst]))
                preds[k][a].append((tst, p))
        per[k] = {a: {w: float(np.nanmean(v)) for w, v in r.items()} for a, r in res.items()}
        print(f"{DISPLAY[k]:10s} " + "  ".join(
            f"{a} {v['pooled']:.3f}/{v['within']:.3f}" for a, v in per[k].items()), flush=True)
    macro = {a: {w: float(np.mean([per[k][a][w] for k in KEYS])) for w in ("pooled", "within")}
             for a in arms}
    wins = int(sum(per[k]["probe"]["within"] > per[k]["uq_int+task"]["within"] for k in KEYS))
    print("macro pooled/within:", {a: (round(v['pooled'], 3), round(v['within'], 3))
                                   for a, v in macro.items()})
    print("probe > uq_int+task within on", wins, "of 8")
    rng = np.random.default_rng(0)
    boot = {}
    for a, b in (("probe", "uq_int+task"), ("uq_int+task", "uq_tn+task")):
        boot[f"{a}-{b}"] = r = boot_within(models, preds, a, b, SP, rng)
        print(f"within {a}-{b} (split 0): {r['delta']:+.1f} [{r['lo']:+.1f},{r['hi']:+.1f}]")
    p = os.path.join(OUT_DIR, "sup_interaction.json")
    json.dump({"per_model": per, "macro": macro, "probe_wins_within": wins,
               "boot_within_split0": boot}, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
