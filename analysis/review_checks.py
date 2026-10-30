"""Checks added after review (judge J1 unless stated).

  cells        number of errors in each (model, task) cell; within-task macro
               AUROC of every score under both judges, with all cells and with
               cells of fewer than 5 errors (or fewer than 5 correct answers)
               left out
  oracle       per-task feasibility with all 100 labels of a task, for KLE,
               MSP and MTE: selective error of the 20% most confident answers,
               the lowest selective error at any retention of 20% or more, and
               the number of models that reach alpha 0.2 at 20%
  retention    threshold transfer at alpha 0.2 onto the models with base
               error above alpha, 500 splits, non-empty certificates only:
               raw threshold vs. the certified retention level mapped through
               the target model's unlabelled grid questions
  mondrian     per-task certification at alpha 0.2 with designs that need
               fewer labels than Bonferroni over 20 levels: a fixed sequence
               per task from 50% retention, at delta and at delta/12, and the
               paper's fixed sequence per group of tasks (4 groups by answer
               type, fixed before running); coverage and the exceedance rate
               on NER, unanswerable and multi-hop, 500 splits

Writes results/review_checks.json.
Usage: python analysis/review_checks.py
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
from protocol import (COVERAGE_GRID, DELTA, OUT_DIR, JUDGES, _passes,  # noqa: E402
                      apply_threshold, auc, certify, grid_split, load_model)
from split_robustness import ratio_ci, splits  # noqa: E402

J = "luna"
A = 0.2
N_SPLITS = 500
HARD_TASKS = ("ner", "unanswerable", "multihop_reasoning")
MIN_CLASS = 5
GROUPS = {
    "labels": ("true_false", "multiple_choice", "sentiment_classification"),
    "spans": ("ner", "information_retrieval"),
    "short_open": ("factual_qa", "commonsense", "multihop_reasoning",
                   "math_reasoning", "unanswerable"),
    "long_open": ("summarization", "translation"),
}


def run_cells(models, scores):
    tasks = sorted(set(models[KEYS[0]]["task"]))
    counts = {}
    for k in KEYS:
        M = models[k]
        e = M["err"][J]
        counts[k] = {t: int(np.nansum(e[M["task"] == t])) for t in tasks}
    flat = [v for d in counts.values() for v in d.values()]
    out = {"errors_per_cell": counts, "min_errors": int(min(flat)),
           "cells_below_5": int(sum(v < MIN_CLASS for v in flat)),
           "cells": len(flat), "within": {}}
    for j in JUDGES:
        res = {}
        for m in METHODS:
            full, kept = [], []
            for k in KEYS:
                M = models[k]
                e, s = M["err"][j], scores[f"{k}/{m}"].astype(float)
                pf, pk = [], []
                for t in tasks:
                    i = (M["task"] == t) & np.isfinite(e)
                    a = auc(e[i], s[i])
                    if not np.isfinite(a):
                        continue
                    pf.append(a)
                    n1 = e[i].sum()
                    if n1 >= MIN_CLASS and i.sum() - n1 >= MIN_CLASS:
                        pk.append(a)
                full.append(np.mean(pf))
                kept.append(np.mean(pk))
            res[m] = {"all_cells": float(np.mean(full)), "min5": float(np.mean(kept))}
        out["within"][j] = res
        print(f"  within-task {j}: " + "  ".join(
            f"{m} {v['all_cells']:.3f}/{v['min5']:.3f}" for m, v in res.items()))
    print(f"  cells: min errors {out['min_errors']}, below 5: "
          f"{out['cells_below_5']}/{out['cells']}")
    return out


def run_oracle(models, scores):
    out = {}
    for m in ("KLE", "MSP", "MTE"):
        out[m] = {}
        for t in HARD_TASKS:
            at20, best = [], []
            for k in KEYS:
                M = models[k]
                i = (M["task"] == t) & np.isfinite(M["err"][J])
                e, s = M["err"][J][i], scores[f"{k}/{m}"].astype(float)[i]
                o = np.argsort(s, kind="stable")
                cum = np.cumsum(e[o]) / np.arange(1, len(o) + 1)
                n20 = int(np.ceil(0.2 * len(o)))
                at20.append(float(cum[n20 - 1]))
                best.append(float(cum[n20 - 1:].min()))
            out[m][t] = {"sel_err_at_20": float(np.mean(at20)),
                         "min_sel_err_ge20": float(np.mean(best)),
                         "models_reaching_alpha_at_20": int(sum(v <= A for v in at20))}
            v = out[m][t]
            print(f"  oracle {m:4s} {t:20s} at20 {v['sel_err_at_20']:.3f} "
                  f"best {v['min_sel_err_ge20']:.3f} models<=0.2 "
                  f"{v['models_reaching_alpha_at_20']}/8")
    return out


def run_retention(models, SP, rng):
    base = {k: float(np.nanmean(models[k]["err"][J])) for k in KEYS}
    hard = [k for k in KEYS if base[k] > A]
    S = len(SP)
    cert = {}
    for ka in KEYS:
        M = models[ka]
        ok = np.isfinite(M["err"][J])
        for sd, (cal, tst) in enumerate(SP):
            grid, _ = grid_split(tst, sd)
            c = cal[ok[cal]]
            tau = certify(M["kle"][c], M["err"][J][c], A, M["kle"][grid], "asc20")
            q = None if tau is None else float(np.mean(M["kle"][grid] <= tau))
            cert[(ka, sd)] = (tau, q)
    r = {(arm, kind): {"ne": np.zeros(S), "exc": np.zeros(S)}
         for arm in ("self", "lower", "higher") for kind in ("raw", "retention")}
    for kb in hard:
        Bm = models[kb]
        okb = np.isfinite(Bm["err"][J])
        for ka in KEYS:
            arm = ("self" if ka == kb else
                   "lower" if base[ka] < base[kb] else "higher")
            for sd, (cal, tst) in enumerate(SP):
                tau, q = cert[(ka, sd)]
                if tau is None:
                    continue
                grid, test = grid_split(tst, sd)
                t = test[okb[test]]
                for kind, th in (("raw", tau),
                                 ("retention", float(np.quantile(Bm["kle"][grid], q)))):
                    _, se = apply_threshold(th, Bm["kle"][t], Bm["err"][J][t])
                    if not np.isfinite(se):
                        continue
                    r[(arm, kind)]["ne"][sd] += 1
                    r[(arm, kind)]["exc"][sd] += int(se > A)
    out = {}
    for (arm, kind), d in r.items():
        out[f"{arm}_{kind}"] = ratio_ci(d["exc"], d["ne"], rng)
        v = out[f"{arm}_{kind}"]
        print(f"  transfer {arm:6s} {kind:9s} {v['rate']:.3f} "
              f"[{v['lo']:.3f}, {v['hi']:.3f}] ({v['num']}/{v['den']})")
    return out


def fixed_sequence(s, e, alpha, grid_scores, start, delta):
    levels = COVERAGE_GRID[COVERAGE_GRID >= start - 1e-9]
    chosen = None
    for tau in np.quantile(grid_scores, levels):
        if _passes(s, e, tau, alpha, delta):
            chosen = float(tau)
        else:
            break
    return chosen


def run_mondrian(models, SP, rng):
    tasks = sorted(set(models[KEYS[0]]["task"]))
    designs = {
        "bonf20_delta": ("task", lambda s, e, g: certify(s, e, A, g, "bonf", delta=DELTA)),
        "fs50_delta": ("task", lambda s, e, g: fixed_sequence(s, e, A, g, 0.5, DELTA)),
        "fs50_delta12": ("task", lambda s, e, g: fixed_sequence(s, e, A, g, 0.5,
                                                                DELTA / len(tasks))),
        "groups_fs20_delta4": ("group", lambda s, e, g: fixed_sequence(s, e, A, g, 0.2,
                                                                       DELTA / len(GROUPS))),
        "marginal_fs20": ("all", lambda s, e, g: fixed_sequence(s, e, A, g, 0.2, DELTA)),
    }
    S = len(SP)
    out = {}
    for name, (unit, fn) in designs.items():
        cov = []
        num = {t: np.zeros(S) for t in HARD_TASKS}
        den = {t: np.zeros(S) for t in HARD_TASKS}
        for k in KEYS:
            M = models[k]
            e, s, task = M["err"][J], M["kle"], M["task"]
            ok = np.isfinite(e)
            strata = ({t: (t,) for t in tasks} if unit == "task" else
                      GROUPS if unit == "group" else {"all": tuple(tasks)})
            for sd, (cal, tst) in enumerate(SP):
                grid, test = grid_split(tst, sd)
                c, t = cal[ok[cal]], test[ok[test]]
                kept = 0
                for members in strata.values():
                    cs = c[np.isin(task[c], members)]
                    ts = t[np.isin(task[t], members)]
                    gs = grid[np.isin(task[grid], members)]
                    tau = fn(s[cs], e[cs], s[gs])
                    if tau is None:
                        continue
                    r = s[ts] <= tau
                    kept += r.sum()
                    for tk in HARD_TASKS:
                        if tk not in members:
                            continue
                        m = r & (task[ts] == tk)
                        if m.any():
                            den[tk][sd] += 1
                            num[tk][sd] += int(e[ts][m].mean() > A)
                cov.append(kept / len(t))
        out[name] = {"coverage": float(np.mean(cov)),
                     "exceed": {tk: ratio_ci(num[tk], den[tk], rng) for tk in HARD_TASKS}}
        print(f"  mondrian {name:20s} coverage {out[name]['coverage']:.3f}  " + "  ".join(
            f"{tk[:5]} {v['rate']:.2f} ({v['den']})" for tk, v in out[name]["exceed"].items()))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits(N_SPLITS)
    rng = np.random.default_rng(0)
    out = {"n_splits": N_SPLITS,
           "cells": run_cells(models, scores),
           "oracle": run_oracle(models, scores),
           "retention": run_retention(models, SP, rng),
           "mondrian": run_mondrian(models, SP, rng)}
    p = os.path.join(OUT_DIR, "review_checks.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
