"""Certified selective answering under the final procedure.

Fixed-sequence Learn-then-Test with Clopper-Pearson bounds, delta = 0.1,
starting at 20% retention (about 100 calibration points at the first test;
starting at 5% lets a single unlucky error end the sequence). Bonferroni
over the 20 levels is reported as a check.

Outputs (results/cert_final.json):
  table4     coverage per model and alpha, J1; macro for J1 and J2
  self       violations under per-model calibration
  transfer   threshold transfer across models at alpha 0.2
  per_task   per-task error of the marginal certificate at alpha 0.2 / 0.3,
             and per-task (Mondrian) coverage
  budget     copied from cert_sensitivity.json

Usage: python analysis/cert_final.py   (after cert_sensitivity.py)
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
from tiers import (DELTA, OUT_DIR, apply_threshold, grid_split,  # noqa: E402
                        load_model, splits)

PROC = "asc20"
ALPHAS = (0.1, 0.15, 0.2, 0.3)


def table4(models, SP, j, proc):
    per = {}
    for k in KEYS:
        M = models[k]
        e, s = M["err"][j], M["kle"]
        ok = np.isfinite(e)
        per[k] = {"base_err": float(np.nanmean(e))}
        for a in ALPHAS:
            cov, viol = [], []
            for sd, (cal, tst) in enumerate(SP):
                grid, tst = grid_split(tst, sd)
                cal, tst = cal[ok[cal]], tst[ok[tst]]
                tau = certify(s[cal], e[cal], a, s[grid], proc)
                c, se = apply_threshold(tau, s[tst], e[tst])
                cov.append(c)
                viol.append(int(np.isfinite(se) and se > a))
            per[k][str(a)] = {"coverage": float(np.mean(cov)),
                              "n_viol": int(sum(viol)), "n": len(viol)}
    return per


def self_violation(per):
    cells = [(k, a) for k in KEYS for a in ALPHAS
             if per[k][str(a)]["coverage"] > 0 and per[k]["base_err"] > a]
    v = sum(per[k][str(a)]["n_viol"] for k, a in cells)
    n = sum(per[k][str(a)]["n"] for k, a in cells)
    return {"cells": len(cells), "violations": v, "draws": n, "rate": v / n}


def transfer(models, SP, alpha=0.2):
    j = "luna"
    raw, ret, self_ = {}, {}, {}
    for ka in KEYS:
        A = models[ka]
        for kb in KEYS:
            Bm = models[kb]
            vr, vq = [], []
            for sd, (cal, tst) in enumerate(SP):
                grid, tst = grid_split(tst, sd)
                ea, sa, eb, sb = A["err"][j], A["kle"], Bm["err"][j], Bm["kle"]
                c = cal[np.isfinite(ea[cal])]
                t = tst[np.isfinite(eb[tst])]
                tau = certify(sa[c], ea[c], alpha, sa[grid], PROC)
                _, se = apply_threshold(tau, sb[t], eb[t])
                vr.append(float(np.isfinite(se) and se > alpha))
                if tau is None:
                    vq.append(0.0)
                else:
                    q = float(np.mean(sa[grid] <= tau))
                    _, se = apply_threshold(float(np.quantile(sb[grid], q)), sb[t], eb[t])
                    vq.append(float(np.isfinite(se) and se > alpha))
            if ka == kb:
                self_[ka] = float(np.mean(vr))
            else:
                raw[f"{ka}->{kb}"] = float(np.mean(vr))
                ret[f"{ka}->{kb}"] = float(np.mean(vq))
    hard = [k for k in KEYS if np.nanmean(models[k]["err"][j]) > alpha]
    tgt = lambda d: [v for p, v in d.items() if p.split("->")[1] in hard]  # noqa: E731
    return {"raw_all": float(np.mean(list(raw.values()))),
            "retention_all": float(np.mean(list(ret.values()))),
            "self_all": float(np.mean(list(self_.values()))),
            "hard_targets": hard,
            "raw_hard": float(np.mean(tgt(raw))),
            "retention_hard": float(np.mean(tgt(ret))),
            "self_hard": float(np.mean([self_[k] for k in hard])),
            "raw_pair_median": float(np.median(list(raw.values()))),
            "raw_pair_max": float(np.max(list(raw.values()))),
            "raw_pairs": raw, "retention_pairs": ret, "self": self_}


def per_task(models, SP):
    j = "luna"
    tasks = sorted(set(models[KEYS[0]]["task"]))
    out = {}
    for a in (0.2, 0.3):
        mg_err = {t: [] for t in tasks}
        mg_viol = {t: [] for t in tasks}
        mg_cov = {t: [] for t in tasks}
        md_cov = {t: [] for t in tasks}
        md_viol = {t: [] for t in tasks}
        tot_mg, tot_md = [], []
        for k in KEYS:
            M = models[k]
            e, s, task = M["err"][j], M["kle"], M["task"]
            for sd, (cal, tst) in enumerate(SP):
                grid, tst = grid_split(tst, sd)
                c = cal[np.isfinite(e[cal])]
                t = tst[np.isfinite(e[tst])]
                tau = certify(s[c], e[c], a, s[grid], PROC)
                tot_mg.append(apply_threshold(tau, s[t], e[t])[0])
                kept = 0
                for tk in tasks:
                    ct, tt, gt = (c[task[c] == tk], t[task[t] == tk],
                                  grid[task[grid] == tk])
                    cv, se = apply_threshold(tau, s[tt], e[tt])
                    mg_cov[tk].append(cv)
                    if np.isfinite(se):
                        mg_err[tk].append(se)
                        mg_viol[tk].append(float(se > a))
                    tau_t = certify(s[ct], e[ct], a, s[gt], "bonf",
                                    delta=DELTA / len(tasks))
                    cv, se = apply_threshold(tau_t, s[tt], e[tt])
                    md_cov[tk].append(cv)
                    kept += cv * len(tt)
                    if np.isfinite(se):
                        md_viol[tk].append(float(se > a))
                tot_md.append(kept / len(t))
        out[str(a)] = {
            "marginal_coverage": float(np.mean(tot_mg)),
            "mondrian_coverage": float(np.mean(tot_md)),
            "tasks": {tk: {
                "marg_cov": float(np.mean(mg_cov[tk])),
                "marg_err": float(np.mean(mg_err[tk])) if mg_err[tk] else None,
                "marg_viol": float(np.mean(mg_viol[tk])) if mg_viol[tk] else None,
                "mond_cov": float(np.mean(md_cov[tk])),
                "mond_viol": float(np.mean(md_viol[tk])) if md_viol[tk] else None}
                for tk in tasks}}
    return out


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits()
    out = {"proc": PROC}
    t4 = {j: table4(models, SP, j, PROC) for j in ("luna", "gemini")}
    bf = {j: table4(models, SP, j, "bonf") for j in ("luna", "gemini")}
    out["table4"] = t4["luna"]
    out["macro"] = {f"{j}_{p}": {str(a): float(np.mean([T[j][k][str(a)]["coverage"]
                                                         for k in KEYS]))
                                 for a in ALPHAS}
                    for p, T in (("asc20", t4), ("bonf", bf)) for j in T}
    out["self"] = {j: self_violation(t4[j]) for j in t4}
    out["self_bonf"] = {j: self_violation(bf[j]) for j in bf}
    out["transfer"] = transfer(models, SP)
    out["per_task"] = per_task(models, SP)
    sens = json.load(open(os.path.join(OUT_DIR, "cert_sensitivity.json")))
    out["budget"] = sens["budget_asc20"]
    out["budget_bonf"] = sens["budget_bonf"]

    print("Certified coverage (J1, asc20):")
    for k in sorted(KEYS, key=lambda k: t4["luna"][k]["base_err"]):
        r = t4["luna"][k]
        print(f"  {DISPLAY[k]:10s} {r['base_err']:.3f} " + " ".join(
            f"{r[str(a)]['coverage']:.2f}" for a in ALPHAS))
    for name, m in out["macro"].items():
        print(f"  macro {name:14s}", {a: round(v, 3) for a, v in m.items()})
    print("self", out["self"], "\nself_bonf", out["self_bonf"])
    tr = out["transfer"]
    print("transfer", {k: v for k, v in tr.items() if not k.endswith("pairs") and k != "self"})
    for a, v in out["per_task"].items():
        print(f"per-task a={a}: marginal cov {v['marginal_coverage']:.3f} "
              f"mondrian cov {v['mondrian_coverage']:.3f}")
        for tk, r in v["tasks"].items():
            print(f"   {tk:26s} marg cov {r['marg_cov']:.2f} err "
                  f"{r['marg_err'] if r['marg_err'] is None else round(r['marg_err'], 3)} "
                  f"viol {r['marg_viol'] if r['marg_viol'] is None else round(r['marg_viol'], 2)}"
                  f" | mond cov {r['mond_cov']:.2f} viol {r['mond_viol']}")
    p = os.path.join(OUT_DIR, "cert_final.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
