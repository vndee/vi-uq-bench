"""Label budget: train a score or calibrate KLE?

With 500 labels, either calibrate KLE on all 500 or train a score on 200
and calibrate it on the other 300. Four arms on the same 200/300 split:

  kle        KLE calibrated on all 500
  probe      hidden-state probe trained on the 200
  sup        logistic regression on the scores + task one-hot, on the 200
  taskprior  each question's task error rate on the 200

Fixed sequence from 20% retention, delta 0.1, J1. For each cell: mean
coverage, share of non-empty certificates, median coverage when non-empty,
and how often the certificate fails on NER and unanswerable items.

Writes results/budget_control.json.
Usage: python analysis/budget_control.py
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
from cert_sensitivity import certify  # noqa: E402
from supervised_baseline import impute, onehot, score_matrix  # noqa: E402
from tiers import (OUT_DIR, PROBE_DEPTH, apply_threshold, fit_predict,  # noqa: E402
                        grid_split, load_hidden, load_model, splits)

ALPHAS = (0.1, 0.15, 0.2)
ARMS = ("kle", "probe", "sup", "taskprior")
WATCH = ("ner", "unanswerable")


def task_rate(e, task, tr, idx):
    rate = {t: float(np.mean(e[tr][task[tr] == t])) if np.any(task[tr] == t)
            else float(np.mean(e[tr])) for t in np.unique(task)}
    return np.array([rate[t] for t in task[idx]])


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits()
    cov = {a: {arm: np.zeros((len(KEYS), len(SP))) for arm in ARMS} for a in ALPHAS}
    pviol = {a: {arm: {t: [] for t in WATCH} for arm in ARMS} for a in ALPHAS}
    for mi, k in enumerate(KEYS):
        M = models[k]
        e, task = M["err"]["luna"], M["task"]
        ok = np.isfinite(e)
        Hans, _ = load_hidden(k)
        H = Hans[:, int(round(PROBE_DEPTH * (Hans.shape[1] - 1)))].astype(np.float32)
        del Hans
        S, T = score_matrix(k, scores), onehot(task)
        for sd, (cal, tst) in enumerate(SP):
            grid, tst = grid_split(tst, sd)
            cal, tst = cal[ok[cal]], tst[ok[tst]]
            perm = np.random.default_rng(200 + sd).permutation(cal)
            tr, cc = np.sort(perm[:200]), np.sort(perm[200:])
            Xs = np.hstack([impute(S, tr), T])
            sc = {"kle": (cal, M["kle"][cal], M["kle"][grid], M["kle"][tst]),
                  "probe": (cc, fit_predict(H[tr], e[tr], H[cc]),
                            fit_predict(H[tr], e[tr], H[grid]),
                            fit_predict(H[tr], e[tr], H[tst])),
                  "sup": (cc, fit_predict(Xs[tr], e[tr], Xs[cc]),
                          fit_predict(Xs[tr], e[tr], Xs[grid]),
                          fit_predict(Xs[tr], e[tr], Xs[tst])),
                  "taskprior": (cc, task_rate(e, task, tr, cc),
                                task_rate(e, task, tr, grid),
                                task_rate(e, task, tr, tst))}
            for a in ALPHAS:
                for arm, (ci, s_c, s_g, s_t) in sc.items():
                    tau = certify(s_c, e[ci], a, s_g, "asc20")
                    c, _ = apply_threshold(tau, s_t, e[tst])
                    cov[a][arm][mi, sd] = c
                    if tau is not None:
                        for t in WATCH:
                            m = task[tst] == t
                            r = s_t[m] <= tau
                            if r.any():
                                pviol[a][arm][t].append(float(e[tst][m][r].mean() > a))
        print(f"  {DISPLAY[k]} done", flush=True)
    out = {}
    for a in ALPHAS:
        out[str(a)] = {}
        for arm in ARMS:
            C = cov[a][arm]
            nz = C[C > 0]
            out[str(a)][arm] = {
                "coverage": float(C.mean()),
                "p_nonempty": float((C > 0).mean()),
                "median_if_nonempty": float(np.median(nz)) if nz.size else 0.0,
                "per_task_violation": {t: (float(np.mean(v)) if v else None)
                                       for t, v in pviol[a][arm].items()}}
        for arm in ("probe", "sup", "taskprior"):
            d = (cov[a][arm] - cov[a]["kle"]).mean(0)
            dp = (cov[a]["probe"] - cov[a][arm]).mean(0) if arm != "probe" else None
            out[str(a)][arm]["minus_kle"] = {
                "diff": float(d.mean()), "lo": float(np.quantile(d, .025)),
                "hi": float(np.quantile(d, .975)),
                "models_ahead": int(((cov[a][arm] - cov[a]["kle"]).mean(1) > 0).sum())}
            if dp is not None:
                out[str(a)][arm]["probe_minus"] = {
                    "diff": float(dp.mean()), "lo": float(np.quantile(dp, .025)),
                    "hi": float(np.quantile(dp, .975)),
                    "models_probe_ahead": int(((cov[a]["probe"] - cov[a][arm]).mean(1)
                                               > 0).sum())}
        print(f"=== alpha {a}")
        for arm, v in out[str(a)].items():
            print(f"  {arm:9s} cov {v['coverage']:.3f} nonempty {v['p_nonempty']:.2f} "
                  f"median|nonempty {v['median_if_nonempty']:.2f} "
                  f"viol ner/unans {v['per_task_violation']} "
                  + (f"| -KLE {v['minus_kle']['diff']:+.3f} [{v['minus_kle']['lo']:+.3f},"
                     f"{v['minus_kle']['hi']:+.3f}] {v['minus_kle']['models_ahead']}/8 "
                     if "minus_kle" in v else "")
                  + (f"| probe-this {v['probe_minus']['diff']:+.3f} [{v['probe_minus']['lo']:+.3f},"
                     f"{v['probe_minus']['hi']:+.3f}] {v['probe_minus']['models_probe_ahead']}/8"
                     if "probe_minus" in v else ""))
    p = os.path.join(OUT_DIR, "budget_control.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
