"""Probe evaluation against baselines that see the same labels.

  1. Main comparison: KLE, TaskPrior, supervised scores (+ task one-hot),
     answer probe, prompt probe, label-free probe, probe + scores + task;
     pooled and within-task macro over the eight models.
  2. Bootstrap over the 5 probe splits: each split's test questions are
     resampled within task, jointly for the eight models.
  3. Label budget: the probe trained on 50 to 500 labels.

tiers.py protocol: cal 500 / test 700, 5 splits, C = 0.05, 54% depth,
standardised inputs. Both judges.

Writes results/probe_control.json.
Usage: python analysis/probe_control.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import SCORES  # noqa: E402
from benchmark import DISPLAY, KEYS, N  # noqa: E402
from supervised_baseline import impute, onehot, score_matrix  # noqa: E402
from task_control import task_prior  # noqa: E402
from tiers import (N_PROBE_SPLITS, OUT_DIR, PROBE_DEPTH,  # noqa: E402
                        fit_predict, load_hidden, load_model, splits)

B = 1000
BUDGETS = (50, 100, 200, 300, 500)
JUDGES = ("luna", "gemini")


def fauc(y, s):
    """Mann-Whitney AUROC; nan if a class is missing."""
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within(y, s, task, tasks):
    return float(np.nanmean([fauc(y[task == t], s[task == t]) for t in tasks]))


def full(idx, p):
    out = np.full(N, np.nan)
    out[idx] = p
    return out


def run_judge(j, models, scores, SP, hidden):
    tasks = np.unique(models[KEYS[0]]["task"])
    arms = ("kle", "taskprior", "sup", "probe", "prompt", "labelfree",
            "probe_sup")
    preds = {k: {a: [] for a in arms} for k in KEYS}
    budget = {k: {nb: [] for nb in BUDGETS} for k in KEYS}
    budget_sup = {k: {nb: [] for nb in BUDGETS} for k in KEYS}
    for k in KEYS:
        M = models[k]
        e, task = M["err"][j], M["task"]
        ok = np.isfinite(e)
        H, Hp = hidden[k]
        S, T = score_matrix(k, scores), onehot(task)
        for s, (cal, tst) in enumerate(SP):
            cal, tst = cal[ok[cal]], tst[ok[tst]]
            Xs = np.hstack([impute(S, cal), T])
            p = {"kle": M["kle"][tst],
                 "taskprior": task_prior(e, task, cal, tst),
                 "sup": fit_predict(Xs[cal], e[cal], Xs[tst]),
                 "probe": fit_predict(H[cal], e[cal], H[tst]),
                 "prompt": fit_predict(Hp[cal], e[cal], Hp[tst])}
            yk = (M["kle"] > np.median(M["kle"][cal])).astype(float)
            p["labelfree"] = fit_predict(H[cal], yk[cal], H[tst])
            Xb = np.hstack([H, Xs])
            p["probe_sup"] = fit_predict(Xb[cal], e[cal], Xb[tst])
            for a in arms:
                preds[k][a].append(full(tst, p[a]))
            for nb in BUDGETS:
                tr = np.random.default_rng(100 + s).choice(cal, min(nb, len(cal)),
                                                           replace=False)
                if len(np.unique(e[tr])) < 2:
                    continue
                budget[k][nb].append(full(tst, fit_predict(H[tr], e[tr], H[tst])))
                budget_sup[k][nb].append(full(tst, fit_predict(Xs[tr], e[tr],
                                                               Xs[tst])))

    def metric(k, arr_list, kind):
        e, task = models[k]["err"][j], models[k]["task"]
        v = []
        for a in arr_list:
            m = np.isfinite(a)
            v.append(fauc(e[m], a[m]) if kind == "pooled"
                     else within(e[m], a[m], task[m], tasks))
        return float(np.nanmean(v))

    res = {"per_model": {}, "macro": {}, "wins_within": {}}
    for k in KEYS:
        res["per_model"][k] = {a: {w: metric(k, preds[k][a], w)
                                   for w in ("pooled", "within")} for a in arms}
    for a in arms:
        res["macro"][a] = {w: float(np.mean([res["per_model"][k][a][w]
                                             for k in KEYS]))
                           for w in ("pooled", "within")}
    for a, b in (("probe", "kle"), ("prompt", "kle"), ("probe", "sup"),
                 ("sup", "kle"), ("labelfree", "kle")):
        res["wins_within"][f"{a}>{b}"] = int(sum(
            res["per_model"][k][a]["within"] > res["per_model"][k][b]["within"]
            for k in KEYS))
    res["wins_pooled_probe>sup"] = int(sum(
        res["per_model"][k]["probe"]["pooled"] > res["per_model"][k]["sup"]["pooled"]
        for k in KEYS))

    bud = {}
    for nb in BUDGETS:
        pm = {k: {"pooled": metric(k, budget[k][nb], "pooled"),
                  "within": metric(k, budget[k][nb], "within"),
                  "sup_within": metric(k, budget_sup[k][nb], "within")}
              for k in KEYS}
        kle_w = {k: res["per_model"][k]["kle"]["within"] for k in KEYS}
        bud[str(nb)] = {
            "pooled": float(np.mean([v["pooled"] for v in pm.values()])),
            "within": float(np.mean([v["within"] for v in pm.values()])),
            "sup_within": float(np.mean([v["sup_within"] for v in pm.values()])),
            "wins_within_vs_kle": int(sum(pm[k]["within"] > kle_w[k] for k in KEYS)),
            "wins_within_vs_sup": int(sum(pm[k]["within"] > pm[k]["sup_within"]
                                          for k in KEYS))}
    res["budget"] = bud

    rng = np.random.default_rng(0)
    task = models[KEYS[0]]["task"]
    split_groups = [[tst[task[tst] == t] for t in tasks] for _, tst in SP]

    def stat(a, b, kind, idx_per_split):
        d = []
        for k in KEYS:
            e = models[k]["err"][j]
            vs = []
            for s, idx in enumerate(idx_per_split):
                pa, pb = preds[k][a][s][idx], preds[k][b][s][idx]
                ee, tt = e[idx], task[idx]
                if kind == "pooled":
                    vs.append(fauc(ee, pa) - fauc(ee, pb))
                else:
                    vs.append(within(ee, pa, tt, tasks) - within(ee, pb, tt, tasks))
            d.append(np.nanmean(vs))
        return 100 * float(np.nanmean(d))

    res["bootstrap"] = {}
    for a, b, kind in (("probe", "sup", "within"), ("probe", "kle", "within"),
                       ("prompt", "kle", "within"), ("sup", "kle", "within"),
                       ("probe", "sup", "pooled"), ("probe_sup", "probe", "within")):
        point = stat(a, b, kind, [tst for _, tst in SP])
        bs = []
        for _ in range(B):
            idx = [np.concatenate([rng.choice(g, len(g), replace=True) for g in G])
                   for G in split_groups]
            bs.append(stat(a, b, kind, idx))
        bs = np.array(bs)
        r = {"delta": point, "lo": float(np.nanquantile(bs, .025)),
             "hi": float(np.nanquantile(bs, .975)), "p_le0": float(np.mean(bs <= 0))}
        res["bootstrap"][f"{kind}:{a}-{b}"] = r
        print(f"  [{j}] {kind:6s} {a}-{b}: {r['delta']:+.1f} "
              f"[{r['lo']:+.1f},{r['hi']:+.1f}] p={r['p_le0']:.4f}", flush=True)
    return res


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits()[:N_PROBE_SPLITS]
    hidden = {}
    for k in KEYS:
        Ha, Hp = load_hidden(k)
        L = int(round(PROBE_DEPTH * (Ha.shape[1] - 1)))
        hidden[k] = (Ha[:, L].astype(np.float32), Hp[:, L].astype(np.float32))
        del Ha, Hp
    out = {}
    for j in JUDGES:
        print(f"=== judge {j}", flush=True)
        r = run_judge(j, models, scores, SP, hidden)
        out[j] = r
        for a, v in r["macro"].items():
            print(f"  {a:10s} pooled {v['pooled']:.3f} within {v['within']:.3f}")
        print("  wins_within", r["wins_within"], "pooled probe>sup",
              r["wins_pooled_probe>sup"])
        for nb, v in r["budget"].items():
            print(f"  budget {nb}: {v}")
        for k in KEYS:
            pm = r["per_model"][k]
            print(f"  {DISPLAY[k]:10s} " + " ".join(
                f"{a}={pm[a]['within']:.3f}" for a in pm))
    p = os.path.join(OUT_DIR, "probe_control.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
