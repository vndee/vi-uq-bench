"""Supervised baseline for the hidden-state probe.

The probe uses 500 correctness labels and KLE uses none. To separate "the
probe has labels" from "hidden states add signal", the same labels train a
logistic regression on the benchmark scores, on the same splits.

Arms
  task           task one-hot only
  uq             the ten scores
  uq+task        ten scores + task one-hot
  uq_tn+task     scores z-scored within task (cal statistics) + task one-hot
  probe          answer-mean hidden state at 54% depth
  probe+uq+task  probe features + uq_tn + task

Pooled and within-task AUROC, macro over models. 5 splits, J1, C = 0.05,
standardised inputs.

Writes results/supervised_baseline.json.
Usage: python analysis/supervised_baseline.py
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
from task_control import METHODS, within_task  # noqa: E402
from tiers import (N_PROBE_SPLITS, OUT_DIR, PROBE_DEPTH, auc,  # noqa: E402
                        fit_predict, load_hidden, load_model, splits)

B = 2000


def score_matrix(k, scores):
    return np.column_stack([scores[f"{k}/{m}"] for m in METHODS]).astype(float)


def impute(X, cal):
    """Fill non-finite entries with the cal-split column median."""
    X = X.copy()
    med = np.nanmedian(np.where(np.isfinite(X[cal]), X[cal], np.nan), axis=0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(med, np.where(bad)[1])
    return X


def task_norm(X, task, cal):
    """z-score each column within each task, using cal statistics."""
    Z = np.zeros_like(X)
    for t in np.unique(task):
        m, mc = task == t, cal[task[cal] == t]
        mu, sd = X[mc].mean(0), X[mc].std(0) + 1e-9
        Z[m] = (X[m] - mu) / sd
    return Z


def onehot(task):
    ts = np.unique(task)
    return (task[:, None] == ts[None, :]).astype(float)


def run_model(k, M, scores, SP):
    e, task = M["err"]["luna"], M["task"]
    ok = np.isfinite(e)
    Hans, _ = load_hidden(k)
    L = int(round(PROBE_DEPTH * (Hans.shape[1] - 1)))
    H = Hans[:, L].astype(np.float32)
    S, T = score_matrix(k, scores), onehot(task)
    arms = ("task", "uq", "uq+task", "uq_tn+task", "probe", "probe+uq+task",
            "kle")
    res = {a: {"pooled": [], "within": []} for a in arms}
    preds = {a: [] for a in arms}
    for cal, tst in SP:
        cal, tst = cal[ok[cal]], tst[ok[tst]]
        Si = impute(S, cal)
        Stn = task_norm(Si, task, cal)
        feats = {"task": T, "uq": Si, "uq+task": np.hstack([Si, T]),
                 "uq_tn+task": np.hstack([Stn, T]), "probe": H,
                 "probe+uq+task": np.hstack([H, Stn, T])}
        for a in arms:
            if a == "kle":
                p = M["kle"][tst]
            else:
                X = feats[a]
                p = fit_predict(X[cal], e[cal], X[tst])
            res[a]["pooled"].append(auc(e[tst], p))
            res[a]["within"].append(within_task(e[tst], p, task[tst]))
            preds[a].append((tst, p))
    return {a: {w: float(np.nanmean(v)) for w, v in r.items()}
            for a, r in res.items()}, preds


def boot_within(models, preds, a, b, SP, rng):
    """Paired bootstrap of the within-task macro difference a-b on split 0's
    test questions (resampled jointly across models, within each task)."""
    tst0 = None
    per = {}
    for k in KEYS:
        ta, pa = preds[k][a][0]
        tb, pb = preds[k][b][0]
        assert np.array_equal(ta, tb)
        per[k] = (ta, pa, pb)
    pool = sorted(set.intersection(*[set(v[0].tolist()) for v in per.values()]))
    pool = np.array(pool)
    task = models[KEYS[0]]["task"]

    def stat(idx):
        d = []
        for k in KEYS:
            ta, pa, pb = per[k]
            pos = {q: i for i, q in enumerate(ta)}
            ii = np.array([pos[q] for q in idx])
            e = models[k]["err"]["luna"][idx]
            tt = task[idx]
            d.append(within_task(e, pa[ii], tt) - within_task(e, pb[ii], tt))
        return 100 * float(np.nanmean(d))

    point = stat(pool)
    groups = [pool[task[pool] == t] for t in np.unique(task[pool])]
    bs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        bs.append(stat(idx))
    bs = np.array(bs)
    return {"delta": point, "lo": float(np.nanquantile(bs, .025)),
            "hi": float(np.nanquantile(bs, .975)),
            "p_le0": float(np.mean(bs <= 0))}


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits()[:N_PROBE_SPLITS]
    out, preds = {"per_model": {}}, {}
    for k in KEYS:
        r, preds[k] = run_model(k, models[k], scores, SP)
        out["per_model"][k] = r
        print(f"{DISPLAY[k]:10s} " + "  ".join(
            f"{a} {v['pooled']:.3f}/{v['within']:.3f}" for a, v in r.items()),
            flush=True)
    arms = list(out["per_model"][KEYS[0]])
    out["macro"] = {a: {w: float(np.mean([out["per_model"][k][a][w]
                                          for k in KEYS]))
                        for w in ("pooled", "within")} for a in arms}
    out["wins_probe_over_uq_tn_task"] = {
        w: int(sum(out["per_model"][k]["probe"][w] >
                   out["per_model"][k]["uq_tn+task"][w] for k in KEYS))
        for w in ("pooled", "within")}
    print("\nmacro (pooled / within-task):")
    for a, v in out["macro"].items():
        print(f"  {a:14s} {v['pooled']:.3f} / {v['within']:.3f}")
    print("probe > uq_tn+task on", out["wins_probe_over_uq_tn_task"], "of 8")
    rng = np.random.default_rng(0)
    out["boot_within_split0"] = {}
    for a, b in (("probe", "uq_tn+task"), ("uq_tn+task", "kle"),
                 ("probe", "kle"), ("probe+uq+task", "probe")):
        r = boot_within(models, {k: preds[k] for k in KEYS}, a, b, SP, rng)
        out["boot_within_split0"][f"{a}-{b}"] = r
        print(f"within-task {a}-{b} (split 0): {r['delta']:+.1f} "
              f"[{r['lo']:+.1f},{r['hi']:+.1f}] p={r['p_le0']:.4f}", flush=True)
    p = os.path.join(OUT_DIR, "supervised_baseline.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
