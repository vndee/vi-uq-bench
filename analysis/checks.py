"""Additional checks reported in Sec. 4.2-4.3 and 5.3 (judge J1).

  decomposition  for every score: within-task AUROC weighted by each task's
                 number of (wrong, correct) pairs, within-task macro without
                 summarization, cross-task AUROC (only pairs whose two
                 answers come from different tasks), and pooled AUROC after
                 standardizing the score within each task
  nonempty       share of draws with a non-empty KLE certificate, per alpha
  complement     self-calibrated certificate evaluated on all 700 questions
                 outside the calibration set (grid pool included), counted
                 over draws with a non-empty certificate
  transfer       threshold transfer at alpha 0.2 onto the models with base
                 error above alpha, counted over draws with a non-empty
                 certificate, split by whether the source model has a lower
                 or higher base error than the target; also without VinaLLaMA
  oracle         per-task feasibility with all 100 labels of a task: the
                 selective error of KLE's 20% most confident answers, and the
                 lowest selective error at any retention of 20% or more
  mondrian       per-task certificate at delta for each task (Bonferroni over
                 the 20 levels), and the smallest error-free retained set
                 that passes the bound under each design
  first_level    Gemma-3 vs. Qwen3 at alpha 0.15: error of the 20% most
                 confident calibration answers and how often the first test
                 of the sequence fails

Writes results/checks.json.
Usage: python analysis/checks.py
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
from protocol import (COVERAGE_GRID, DELTA, OUT_DIR, apply_threshold,  # noqa: E402
                      auc, certify, cp_ucb, grid_split, load_model, splits)

J = "luna"
HARD_TASKS = ("ner", "unanswerable", "multihop_reasoning")


def run_decomposition(models, scores):
    res = {m: {k: [] for k in ("pair_weighted", "no_summarization", "cross_task",
                               "standardized")} for m in METHODS}
    for key in KEYS:
        M = models[key]
        e, task = M["err"][J], M["task"]
        for m in METHODS:
            s = scores[f"{key}/{m}"].astype(float)
            num = den = 0.0
            per = []
            for t in np.unique(task):
                i = (task == t) & np.isfinite(e) & np.isfinite(s)
                n1 = e[i].sum()
                npair = n1 * (i.sum() - n1)
                if npair == 0:
                    continue
                a = auc(e[i], s[i])
                num += a * npair
                den += npair
                if t != "summarization":
                    per.append(a)
            res[m]["pair_weighted"].append(num / den)
            res[m]["no_summarization"].append(float(np.mean(per)))
            i = np.isfinite(e) & np.isfinite(s)
            ee, ss, tt = e[i], s[i], task[i]
            P, Q = np.where(ee == 1)[0], np.where(ee == 0)[0]
            cmp = (ss[P][:, None] > ss[Q][None, :]) + 0.5 * (ss[P][:, None] == ss[Q][None, :])
            res[m]["cross_task"].append(float(cmp[tt[P][:, None] != tt[Q][None, :]].mean()))
            z = np.zeros_like(s)
            for t in np.unique(task):
                w = task == t
                z[w] = (s[w] - np.nanmean(s[w])) / (np.nanstd(s[w]) + 1e-9)
            res[m]["standardized"].append(auc(e, z))
    out = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()}
    for m, v in out.items():
        print(f"  {m:11s} " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()))
    return out


def run_nonempty(models, SP):
    out = {}
    for a in (0.1, 0.15, 0.2, 0.3):
        ne = []
        for k in KEYS:
            M = models[k]
            e, s = M["err"][J], M["kle"]
            ok = np.isfinite(e)
            for sd, (cal, tst) in enumerate(SP):
                grid, _ = grid_split(tst, sd)
                c = cal[ok[cal]]
                ne.append(certify(s[c], e[c], a, s[grid], "asc20") is not None)
        out[str(a)] = float(np.mean(ne))
    print("  nonempty share", out)
    return out


def run_complement(models, SP):
    """Over the cells whose base error exceeds alpha."""
    out = {}
    for a in (0.1, 0.15, 0.2, 0.3):
        n = ne = v = 0
        for k in KEYS:
            M = models[k]
            e, s = M["err"][J], M["kle"]
            ok = np.isfinite(e)
            if np.nanmean(e) <= a:
                continue
            for sd, (cal, tst) in enumerate(SP):
                grid, _ = grid_split(tst, sd)
                cal, comp = cal[ok[cal]], tst[ok[tst]]
                tau = certify(s[cal], e[cal], a, s[grid], "asc20")
                _, se = apply_threshold(tau, s[comp], e[comp])
                n += 1
                if tau is not None:
                    ne += 1
                    v += int(np.isfinite(se) and se > a)
        out[str(a)] = {"draws": n, "nonempty": ne, "violations": v,
                       "rate_nonempty": v / max(ne, 1)}
        print(f"  complement a={a}: {out[str(a)]}")
    return out


def run_transfer(models, SP, a=0.2):
    base = {k: float(np.nanmean(models[k]["err"][J])) for k in KEYS}
    hard = [k for k in KEYS if base[k] > a]

    def run(excl=()):
        r = {arm: {"draws": 0, "nonempty": 0, "violations": 0, "excess": []}
             for arm in ("self", "lower", "higher")}
        for kb in hard:
            if kb in excl:
                continue
            for ka in KEYS:
                if ka in excl:
                    continue
                arm = ("self" if ka == kb else
                       "lower" if base[ka] < base[kb] else "higher")
                A, B = models[ka], models[kb]
                for sd, (cal, tst) in enumerate(SP):
                    grid, tst = grid_split(tst, sd)
                    c = cal[np.isfinite(A["err"][J][cal])]
                    t = tst[np.isfinite(B["err"][J][tst])]
                    tau = certify(A["kle"][c], A["err"][J][c], a, A["kle"][grid], "asc20")
                    _, se = apply_threshold(tau, B["kle"][t], B["err"][J][t])
                    r[arm]["draws"] += 1
                    if tau is not None and np.isfinite(se):
                        r[arm]["nonempty"] += 1
                        r[arm]["violations"] += int(se > a)
                        r[arm]["excess"].append(se - a)
        for d in r.values():
            d["rate_nonempty"] = d["violations"] / max(d["nonempty"], 1)
            d["mean_excess"] = float(np.mean(d.pop("excess"))) if d["nonempty"] else None
        return r

    out = {"alpha": a, "targets": hard, "all": run(), "without_vinallama": run(("vinallama",))}
    for name in ("all", "without_vinallama"):
        for arm, d in out[name].items():
            print(f"  transfer {name:17s} {arm:6s} {d}")
    return out


def run_oracle(models):
    out = {}
    for t in HARD_TASKS:
        at20, best = [], []
        for k in KEYS:
            M = models[k]
            m = (M["task"] == t) & np.isfinite(M["err"][J])
            e, s = M["err"][J][m], M["kle"][m]
            o = np.argsort(s, kind="stable")
            cum = np.cumsum(e[o]) / np.arange(1, len(o) + 1)
            n20 = int(np.ceil(0.2 * len(o)))
            at20.append(float(cum[n20 - 1]))
            best.append(float(cum[n20 - 1:].min()))
        out[t] = {"sel_err_at_20": float(np.mean(at20)),
                  "min_sel_err_ge20": float(np.mean(best)),
                  "models_reaching_0.2_at_20": int(sum(v <= 0.2 for v in at20)),
                  "per_model_at_20": dict(zip(KEYS, at20))}
        print(f"  oracle {t:20s} at20 {np.mean(at20):.3f} best>=20 {np.mean(best):.3f} "
              f"models<=0.2 {out[t]['models_reaching_0.2_at_20']}/8")
    return out


def run_mondrian(models, SP):
    tasks = sorted(set(models[KEYS[0]]["task"]))
    L = len(COVERAGE_GRID)
    out = {"min_n_zero_errors": {
        name: {str(a): int(next(n for n in range(1, 2000) if cp_ucb(0, n, d) <= a))
               for a in (0.2, 0.3)}
        for name, d in (("all_tasks_delta_over_12", DELTA / len(tasks) / L),
                        ("each_task_delta", DELTA / L))}}
    for a in (0.2, 0.3):
        tot, per = [], {t: [] for t in tasks}
        for k in KEYS:
            M = models[k]
            e, s, task = M["err"][J], M["kle"], M["task"]
            for sd, (cal, tst) in enumerate(SP):
                grid, tst = grid_split(tst, sd)
                c = cal[np.isfinite(e[cal])]
                t = tst[np.isfinite(e[tst])]
                kept = 0
                for tk in tasks:
                    ct, tt, gt = (c[task[c] == tk], t[task[t] == tk],
                                  grid[task[grid] == tk])
                    tau = certify(s[ct], e[ct], a, s[gt], "bonf", delta=DELTA)
                    cv, _ = apply_threshold(tau, s[tt], e[tt])
                    per[tk].append(cv)
                    kept += cv * len(tt)
                tot.append(kept / len(t))
        out[f"each_task_delta_a{a}"] = {
            "coverage": float(np.mean(tot)),
            "tasks": {tk: float(np.mean(per[tk])) for tk in tasks}}
        print(f"  mondrian per-task delta a={a}: coverage {np.mean(tot):.3f}")
    print("  min error-free set", out["min_n_zero_errors"])
    return out


def run_first_level(models, SP, a=0.15):
    out = {}
    for k in ("gemma3", "qwen3"):
        M = models[k]
        e, s = M["err"][J], M["kle"]
        ok = np.isfinite(e)
        err20, empty = [], 0
        for sd, (cal, tst) in enumerate(SP):
            grid, _ = grid_split(tst, sd)
            c = cal[ok[cal]]
            r = s[c] <= np.quantile(s[grid], 0.2)
            err20.append(float(e[c][r].mean()))
            empty += certify(s[c], e[c], a, s[grid], "asc20") is None
        out[k] = {"cal_err_top20": float(np.mean(err20)), "splits_failing": int(empty),
                  "splits": len(SP)}
        print(f"  first level {k}: {out[k]}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits()
    out = {"decomposition": run_decomposition(models, scores),
           "nonempty": run_nonempty(models, SP),
           "complement": run_complement(models, SP),
           "transfer": run_transfer(models, SP),
           "oracle": run_oracle(models),
           "mondrian": run_mondrian(models, SP),
           "first_level": run_first_level(models, SP)}
    p = os.path.join(OUT_DIR, "checks.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
