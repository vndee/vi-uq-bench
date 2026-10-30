"""Sensitivity analyses for the certificate (Sec. 5.2-5.3).

Uses the protocol.py splits and grid pools throughout.

  procedure   the KLE certificate under four FWER-valid procedures, both
              judges: fixed sequence from 5% or 20% retention, descending
              from 100%, and Bonferroni over the 20 levels.
  truncation  answers that hit the token budget, and AUROC on naturally
              ended answers only.

Writes results/cert_sensitivity.json.
Usage: python analysis/cert_sensitivity.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import GEN_DIR, SCORES  # noqa: E402
from benchmark import DISPLAY, KEYS, N  # noqa: E402
from protocol import (OUT_DIR, apply_threshold, auc, certify, grid_split,  # noqa: E402
                      load_model, splits)

ALPHAS = (0.1, 0.15, 0.2, 0.3)
JUDGES = ("luna", "gemini")


def run_procedures(models, SP):
    out = {}
    for j in JUDGES:
        out[j] = {}
        for proc in ("asc05", "asc20", "desc", "bonf"):
            per = {}
            for k in KEYS:
                M = models[k]
                e, s = M["err"][j], M["kle"]
                ok = np.isfinite(e)
                per[k] = {}
                for a in ALPHAS:
                    cov, viol, slack = [], [], []
                    for sd, (cal, tst) in enumerate(SP):
                        grid, tst = grid_split(tst, sd)
                        cal, tst = cal[ok[cal]], tst[ok[tst]]
                        tau = certify(s[cal], e[cal], a, s[grid], proc)
                        c, se = apply_threshold(tau, s[tst], e[tst])
                        cov.append(c)
                        viol.append(float(np.isfinite(se) and se > a))
                        if np.isfinite(se):
                            slack.append(a - se)
                    per[k][str(a)] = {"coverage": float(np.mean(cov)),
                                      "violation": float(np.mean(viol)),
                                      "slack_mean": float(np.mean(slack)) if slack else None,
                                      "n_certified": len(slack)}
            macro = {str(a): {
                "coverage": float(np.mean([per[k][str(a)]["coverage"] for k in KEYS])),
                "violation_nontrivial": _nontrivial_rate(per, models, j, a)}
                for a in ALPHAS}
            out[j][proc] = {"per_model": per, "macro": macro}
            print(f"[{j}] {proc:5s} " + "  ".join(
                f"a={a}: cov {macro[str(a)]['coverage']:.3f} "
                f"viol {macro[str(a)]['violation_nontrivial']:.3f}" for a in ALPHAS),
                flush=True)
        print(f"[{j}] per-model coverage at a=0.3 / 0.15:")
        for k in KEYS:
            print("   ", f"{DISPLAY[k]:10s}", "  ".join(
                f"{p}: {out[j][p]['per_model'][k]['0.3']['coverage']:.2f}/"
                f"{out[j][p]['per_model'][k]['0.15']['coverage']:.2f}"
                for p in ("asc05", "asc20", "desc", "bonf")),
                f"base {np.nanmean(models[k]['err'][j]):.3f}")
    return out


def _nontrivial_rate(per, models, j, a):
    """Violation rate over cells that certify something and whose base error
    exceeds alpha (the non-trivial cells)."""
    v = [per[k][str(a)]["violation"] for k in KEYS
         if per[k][str(a)]["coverage"] > 0 and np.nanmean(models[k]["err"][j]) > a]
    return float(np.mean(v)) if v else float("nan")


def run_truncation(models, scores, SP):
    out = {"per_model": {}, "per_task": {}}
    tasks = sorted(set(models[KEYS[0]]["task"]))
    task_g = {t: [] for t in tasks}
    task_s = {t: [] for t in tasks}
    for k in KEYS:
        rows = [json.loads(l) for l in open(os.path.join(GEN_DIR, f"gen_{k}.jsonl"))]
        rows.sort(key=lambda r: r["idx"])
        rows = rows[:N]
        tg = np.array([not r["greedy_hit_eos"] for r in rows])
        ts = np.array([np.mean([n >= r["max_new_tokens"] for n in r["sample_n_tokens"]])
                       for r in rows])
        M = models[k]
        e = M["err"]["luna"]
        ok = np.isfinite(e)
        nat = ok & ~tg
        aucs = {}
        for m in ("KLE", "MTE", "MeanLength", "DegMat"):
            s = scores[f"{k}/{m}"].astype(float)
            aucs[m] = {"all": auc(e[ok], s[ok]), "natural": auc(e[nat], s[nat])}
        out["per_model"][k] = {
            "greedy_trunc": float(tg.mean()), "sample_trunc": float(ts.mean()),
            "err_trunc": float(np.nanmean(e[tg])) if tg.any() else None,
            "err_natural": float(np.nanmean(e[~tg])), "auc": aucs}
        for t in tasks:
            m = M["task"] == t
            task_g[t].append(float(tg[m].mean()))
            task_s[t].append(float(ts[m].mean()))
        o = out["per_model"][k]
        print(f"[trunc] {DISPLAY[k]:10s} greedy {o['greedy_trunc']:.3f} samples "
              f"{o['sample_trunc']:.3f} err trunc {o['err_trunc']} natural "
              f"{o['err_natural']:.3f} | " + " ".join(
                  f"{m} {v['all']:.3f}->{v['natural']:.3f}" for m, v in aucs.items()),
              flush=True)
    for t in tasks:
        out["per_task"][t] = {"greedy_trunc_mean": float(np.mean(task_g[t])),
                              "greedy_trunc_max": float(np.max(task_g[t])),
                              "sample_trunc_mean": float(np.mean(task_s[t]))}
        v = out["per_task"][t]
        print(f"    {t:26s} greedy mean {v['greedy_trunc_mean']:.3f} max "
              f"{v['greedy_trunc_max']:.3f} samples {v['sample_trunc_mean']:.3f}")
    pm = out["per_model"]
    out["macro"] = {m: {w: float(np.mean([pm[k]["auc"][m][w] for k in KEYS]))
                        for w in ("all", "natural")}
                    for m in ("KLE", "MTE", "MeanLength", "DegMat")}
    out["greedy_trunc_overall"] = float(np.mean([pm[k]["greedy_trunc"] for k in KEYS]))
    print("[trunc] macro", out["macro"], "overall greedy",
          round(out["greedy_trunc_overall"], 4))
    return out


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits()
    out = {"truncation": run_truncation(models, scores, SP),
           "procedures": run_procedures(models, SP)}
    p = os.path.join(OUT_DIR, "cert_sensitivity.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
