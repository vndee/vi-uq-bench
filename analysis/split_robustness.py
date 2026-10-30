"""Certification results with 500 splits instead of 20, with intervals.

Same protocol as protocol.py (500 calibration questions, 200 unlabelled
grid questions, the rest for testing; fixed sequence from 20% retention,
delta 0.1, judge J1), only with split seeds 0..499. Draws share questions
and splits are shared across models, so intervals come from a bootstrap
over splits (all draws of a split resampled together), B = 2000.

  self       exceedance of the self-calibrated certificate over the cells
             (model, alpha) with base error above alpha and coverage above
             zero: on the test split, over all draws and over non-empty
             certificates; and the population check: the selective error
             of the certified threshold on all 1,200 questions (the pool
             the draws come from), and on the 700 outside the calibration
             set
  transfer   threshold transfer at alpha 0.2 onto the models with base
             error above alpha, over non-empty certificates, by direction
             (source with lower / higher base error), with and without
             VinaLLaMA
  per_task   per-task exceedance of the marginal certificate at alpha 0.2
             on NER, unanswerable and multi-hop

Writes results/split_robustness.json.
Usage: python analysis/split_robustness.py
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
from protocol import (N_CAL, OUT_DIR, apply_threshold, certify,  # noqa: E402
                      grid_split, load_model)

J = "luna"
N_SPLITS = 500
B = 2000
ALPHAS = (0.1, 0.15, 0.2, 0.3)
HARD_TASKS = ("ner", "unanswerable", "multihop_reasoning")


def splits(n):
    out = []
    for seed in range(n):
        perm = np.random.default_rng(seed).permutation(N)
        out.append((np.sort(perm[:N_CAL]), np.sort(perm[N_CAL:])))
    return out


def ratio_ci(num, den, rng):
    """num, den: per-split counts. Ratio of sums, bootstrap over splits."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    est = num.sum() / max(den.sum(), 1)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(num), len(num))
        d = den[i].sum()
        if d > 0:
            bs.append(num[i].sum() / d)
    lo, hi = (np.quantile(bs, [.025, .975]) if bs else (np.nan, np.nan))
    return {"rate": float(est) if den.sum() else float("nan"),
            "lo": float(lo), "hi": float(hi),
            "num": int(num.sum()), "den": int(den.sum())}


def run_self(models, SP, rng):
    S = len(SP)
    rec = {}
    for k in KEYS:
        M = models[k]
        e, s = M["err"][J], M["kle"]
        ok = np.isfinite(e)
        base = float(np.nanmean(e))
        for a in ALPHAS:
            if base <= a:
                continue
            r = {n: np.zeros(S) for n in ("draws", "nonempty", "test_exc",
                                          "pool_exc", "comp_exc")}
            for sd, (cal, tst) in enumerate(SP):
                grid, test = grid_split(tst, sd)
                c, t, comp = cal[ok[cal]], test[ok[test]], tst[ok[tst]]
                tau = certify(s[c], e[c], a, s[grid], "asc20")
                r["draws"][sd] = 1
                if tau is None:
                    continue
                r["nonempty"][sd] = 1
                _, se = apply_threshold(tau, s[t], e[t])
                r["test_exc"][sd] = float(np.isfinite(se) and se > a)
                _, se = apply_threshold(tau, s[ok], e[ok])
                r["pool_exc"][sd] = float(se > a)
                _, se = apply_threshold(tau, s[comp], e[comp])
                r["comp_exc"][sd] = float(np.isfinite(se) and se > a)
            if r["nonempty"].sum() > 0:
                rec[(k, a)] = r
    out = {"cells": len(rec)}

    def agg(cells, num, den):
        return ratio_ci(sum(rec[c][num] for c in cells),
                        sum(rec[c][den] for c in cells), rng)
    cells = list(rec)
    out["all_alphas"] = {
        "test_all_draws": agg(cells, "test_exc", "draws"),
        "test_nonempty": agg(cells, "test_exc", "nonempty"),
        "pool_all_draws": agg(cells, "pool_exc", "draws"),
        "pool_nonempty": agg(cells, "pool_exc", "nonempty"),
        "complement_nonempty": agg(cells, "comp_exc", "nonempty")}
    for a in ALPHAS:
        ca = [c for c in cells if c[1] == a]
        if ca:
            out[str(a)] = {"cells": len(ca),
                           "nonempty_share": agg(ca, "nonempty", "draws"),
                           "test_all_draws": agg(ca, "test_exc", "draws"),
                           "pool_all_draws": agg(ca, "pool_exc", "draws"),
                           "pool_nonempty": agg(ca, "pool_exc", "nonempty"),
                           "complement_nonempty": agg(ca, "comp_exc", "nonempty")}
    for name, v in out["all_alphas"].items():
        print(f"  self {name:20s} {v['rate']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] "
              f"({v['num']}/{v['den']})", flush=True)
    for a in ALPHAS:
        if str(a) in out:
            v = out[str(a)]
            print(f"   a={a}: cells {v['cells']} nonempty {v['nonempty_share']['rate']:.3f} "
                  f"test/all {v['test_all_draws']['rate']:.3f} pool/all "
                  f"{v['pool_all_draws']['rate']:.3f} pool/nonempty "
                  f"{v['pool_nonempty']['rate']:.3f} comp/nonempty "
                  f"{v['complement_nonempty']['rate']:.3f}")
    return out


def run_transfer(models, SP, rng, a=0.2):
    base = {k: float(np.nanmean(models[k]["err"][J])) for k in KEYS}
    hard = [k for k in KEYS if base[k] > a]
    S = len(SP)
    taus = {}
    for ka in KEYS:
        A = models[ka]
        ok = np.isfinite(A["err"][J])
        for sd, (cal, tst) in enumerate(SP):
            grid, _ = grid_split(tst, sd)
            c = cal[ok[cal]]
            taus[(ka, sd)] = certify(A["kle"][c], A["err"][J][c], a, A["kle"][grid], "asc20")

    def run(excl=()):
        r = {arm: {"ne": np.zeros(S), "exc": np.zeros(S), "excess": []}
             for arm in ("self", "lower", "higher")}
        for kb in hard:
            if kb in excl:
                continue
            Bm = models[kb]
            okb = np.isfinite(Bm["err"][J])
            for ka in KEYS:
                if ka in excl:
                    continue
                arm = ("self" if ka == kb else
                       "lower" if base[ka] < base[kb] else "higher")
                for sd, (cal, tst) in enumerate(SP):
                    tau = taus[(ka, sd)]
                    if tau is None:
                        continue
                    _, test = grid_split(tst, sd)
                    t = test[okb[test]]
                    _, se = apply_threshold(tau, Bm["kle"][t], Bm["err"][J][t])
                    if not np.isfinite(se):
                        continue
                    r[arm]["ne"][sd] += 1
                    r[arm]["exc"][sd] += int(se > a)
                    r[arm]["excess"].append(se - a)
        return {arm: dict(ratio_ci(d["exc"], d["ne"], rng),
                          mean_excess=float(np.mean(d["excess"])) if d["excess"] else None)
                for arm, d in r.items()}

    out = {"alpha": a, "targets": hard, "all": run(),
           "without_vinallama": run(("vinallama",))}
    for name in ("all", "without_vinallama"):
        for arm, v in out[name].items():
            print(f"  transfer {name:17s} {arm:6s} {v['rate']:.3f} "
                  f"[{v['lo']:.3f}, {v['hi']:.3f}] ({v['num']}/{v['den']}) "
                  f"excess {v['mean_excess']}", flush=True)
    return out


def run_per_task(models, SP, rng, a=0.2):
    S = len(SP)
    num = {t: np.zeros(S) for t in HARD_TASKS}
    den = {t: np.zeros(S) for t in HARD_TASKS}
    err = {t: [] for t in HARD_TASKS}
    for k in KEYS:
        M = models[k]
        e, s, task = M["err"][J], M["kle"], M["task"]
        ok = np.isfinite(e)
        for sd, (cal, tst) in enumerate(SP):
            grid, test = grid_split(tst, sd)
            c, t = cal[ok[cal]], test[ok[test]]
            tau = certify(s[c], e[c], a, s[grid], "asc20")
            if tau is None:
                continue
            for tk in HARD_TASKS:
                tt = t[task[t] == tk]
                _, se = apply_threshold(tau, s[tt], e[tt])
                if np.isfinite(se):
                    den[tk][sd] += 1
                    num[tk][sd] += int(se > a)
                    err[tk].append(se)
    out = {}
    for tk in HARD_TASKS:
        out[tk] = dict(ratio_ci(num[tk], den[tk], rng), sel_err=float(np.mean(err[tk])))
        v = out[tk]
        print(f"  per-task {tk:20s} sel err {v['sel_err']:.3f} exceed {v['rate']:.3f} "
              f"[{v['lo']:.3f}, {v['hi']:.3f}]", flush=True)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits(N_SPLITS)
    rng = np.random.default_rng(0)
    out = {"n_splits": N_SPLITS, "B": B,
           "self": run_self(models, SP, rng),
           "transfer": run_transfer(models, SP, rng),
           "per_task": run_per_task(models, SP, rng)}
    p = os.path.join(OUT_DIR, "split_robustness.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
