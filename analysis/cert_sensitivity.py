"""Sensitivity analyses for the certificate.

Uses the tiers.py splits and grid pools throughout.

  A. procedure   the KLE certificate under four FWER-valid procedures, both
                 judges: fixed sequence from 5% or 20% retention, descending
                 from 100%, and Bonferroni over the 20 levels.
  B. transfer    a threshold calibrated on one model applied to another,
                 either as a raw threshold or as a retention level mapped
                 through the target model's unlabelled pool.
  C. mondrian    per-task certificates at delta/12 vs. the per-task error of
                 the marginal certificate.
  D. budget      splitting a label budget between training a score and
                 calibrating it.
  E. truncation  answers that hit the token budget, and AUROC on naturally
                 ended answers only.

Writes results/cert_sensitivity.json (cert_final.py reads it).
Usage: python analysis/cert_sensitivity.py
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
from tiers import (COVERAGE_GRID, DELTA, N, GEN_DIR, OUT_DIR, PROBE_DEPTH,  # noqa: E402
                        apply_threshold, auc, cp_ucb, fit_predict, grid_split,
                        load_hidden, load_model, splits)

ALPHAS = (0.1, 0.15, 0.2, 0.3)
JUDGES = ("luna", "gemini")


def _passes(s, e, tau, alpha, delta):
    r = s <= tau
    return cp_ucb(int(e[r].sum()), int(r.sum()), delta) <= alpha


def certify(s, e, alpha, grid_scores, proc, delta=DELTA):
    """Largest certified threshold (None if nothing is certified)."""
    taus = np.quantile(grid_scores, COVERAGE_GRID)
    if proc in ("asc05", "asc20"):
        lo = 0.05 if proc == "asc05" else 0.20
        chosen = None
        for c, tau in zip(COVERAGE_GRID, taus):
            if c < lo - 1e-9:
                continue
            if _passes(s, e, tau, alpha, delta):
                chosen = float(tau)
            else:
                break
        return chosen
    if proc == "desc":
        return float(taus[-1]) if _passes(s, e, taus[-1], alpha, delta) else None
    if proc == "bonf":
        ok = [float(t) for t in taus if _passes(s, e, t, alpha, delta / len(taus))]
        return max(ok) if ok else None
    raise ValueError(proc)


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
        g = out[j]["asc05"]["per_model"]
        print(f"[{j}] per-model coverage at a=0.3 / 0.15:")
        for k in KEYS:
            print("   ", f"{DISPLAY[k]:10s}", "  ".join(
                f"{p}: {out[j][p]['per_model'][k]['0.3']['coverage']:.2f}/"
                f"{out[j][p]['per_model'][k]['0.15']['coverage']:.2f}"
                for p in ("asc05", "asc20", "desc", "bonf")),
                f"base {np.nanmean(models[k]['err'][j]):.3f}")
        _ = g
    return out


def _nontrivial_rate(per, models, j, a):
    """Violation rate over cells that certify something and whose base error
    exceeds alpha (the non-trivial cells)."""
    v = [per[k][str(a)]["violation"] for k in KEYS
         if per[k][str(a)]["coverage"] > 0 and np.nanmean(models[k]["err"][j]) > a]
    return float(np.mean(v)) if v else float("nan")


def run_transfer(models, SP, alpha=0.2):
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
                tau = certify(sa[c], ea[c], alpha, sa[grid], "asc05")
                _, se = apply_threshold(tau, sb[t], eb[t])
                vr.append(float(np.isfinite(se) and se > alpha))
                if tau is None:
                    vq.append(0.0)
                else:
                    q = float(np.mean(sa[grid] <= tau))
                    tau_b = float(np.quantile(sb[grid], q))
                    _, se = apply_threshold(tau_b, sb[t], eb[t])
                    vq.append(float(np.isfinite(se) and se > alpha))
            if ka == kb:
                self_[ka] = float(np.mean(vr))
            else:
                raw[f"{ka}->{kb}"] = float(np.mean(vr))
                ret[f"{ka}->{kb}"] = float(np.mean(vq))
    base = {k: float(np.nanmean(models[k]["err"][j])) for k in KEYS}
    hard = [k for k in KEYS if base[k] > alpha]
    sel = lambda d: [v for p, v in d.items() if p.split("->")[1] in hard]  # noqa: E731
    out = {"alpha": alpha, "raw_pairs": raw, "retention_pairs": ret, "self": self_,
           "raw_mean_all": float(np.mean(list(raw.values()))),
           "retention_mean_all": float(np.mean(list(ret.values()))),
           "self_mean_all": float(np.mean(list(self_.values()))),
           "target_base_gt_alpha": hard,
           "raw_mean_hard": float(np.mean(sel(raw))),
           "retention_mean_hard": float(np.mean(sel(ret))),
           "self_mean_hard": float(np.mean([self_[k] for k in hard])),
           "raw_pair_quantiles": [float(x) for x in np.quantile(list(raw.values()),
                                                                 [0, .25, .5, .75, 1])]}
    print(f"[transfer a={alpha}] all targets: raw {out['raw_mean_all']:.3f} "
          f"retention {out['retention_mean_all']:.3f} self {out['self_mean_all']:.3f}"
          f" | targets with base>{alpha} ({len(hard)}): raw {out['raw_mean_hard']:.3f}"
          f" retention {out['retention_mean_hard']:.3f} self {out['self_mean_hard']:.3f}"
          f" | raw pair quantiles {np.round(out['raw_pair_quantiles'], 2)}", flush=True)
    return out


def run_mondrian(models, SP, proc="bonf"):
    """Per-task LTT; `proc` for the per-task certificate. Ascending from 5%
    retention is hopeless at ~42 labels per task (2 points at the first level),
    so the fair comparison uses Bonferroni or the 20% start."""
    j = "luna"
    tasks = sorted(set(models[KEYS[0]]["task"]))
    out = {}
    for a in (0.2, 0.3):
        marg = {t: [] for t in tasks}
        mond_cov = {t: [] for t in tasks}
        mond_err = {t: [] for t in tasks}
        mond_viol = {t: [] for t in tasks}
        tot_marg, tot_mond = [], []
        for k in KEYS:
            M = models[k]
            e, s, task = M["err"][j], M["kle"], M["task"]
            for sd, (cal, tst) in enumerate(SP):
                grid, tst = grid_split(tst, sd)
                c = cal[np.isfinite(e[cal])]
                t = tst[np.isfinite(e[tst])]
                tau = certify(s[c], e[c], a, s[grid], "asc05")
                c_m, _ = apply_threshold(tau, s[t], e[t])
                tot_marg.append(c_m)
                kept = 0
                for tk in tasks:
                    ct, tt, gt = (c[task[c] == tk], t[task[t] == tk],
                                  grid[task[grid] == tk])
                    if tau is not None:
                        r = s[tt] <= tau
                        if r.any():
                            marg[tk].append(float(e[tt][r].mean()))
                    tau_t = certify(s[ct], e[ct], a, s[gt], proc,
                                    delta=DELTA / len(tasks))
                    cv, se = apply_threshold(tau_t, s[tt], e[tt])
                    mond_cov[tk].append(cv)
                    kept += cv * len(tt)
                    if np.isfinite(se):
                        mond_err[tk].append(se)
                        mond_viol[tk].append(float(se > a))
                tot_mond.append(kept / len(t))
        out[str(a)] = {
            "marginal_coverage": float(np.mean(tot_marg)),
            "mondrian_coverage": float(np.mean(tot_mond)),
            "per_task": {tk: {
                "marginal_sel_err": float(np.mean(marg[tk])) if marg[tk] else None,
                "mondrian_coverage": float(np.mean(mond_cov[tk])),
                "mondrian_sel_err": float(np.mean(mond_err[tk])) if mond_err[tk] else None,
                "mondrian_violation": float(np.mean(mond_viol[tk])) if mond_viol[tk] else None}
                for tk in tasks}}
        o = out[str(a)]
        print(f"[mondrian {proc} a={a}] coverage marginal {o['marginal_coverage']:.3f} "
              f"mondrian {o['mondrian_coverage']:.3f}", flush=True)
        for tk, v in o["per_task"].items():
            print(f"    {tk:26s} marg_err {v['marginal_sel_err'] or float('nan'):.3f}"
                  f"  mond_cov {v['mondrian_coverage']:.3f}"
                  f"  mond_err {v['mondrian_sel_err'] or float('nan'):.3f}"
                  f"  mond_viol {v['mondrian_violation'] if v['mondrian_violation'] is not None else float('nan'):.3f}")
    return out


def run_budget(models, SP, proc="asc05"):
    j = "luna"
    TRAIN = (100, 200, 300)
    res = {str(a): {str(n): {"kle": [], "probe": []} for n in TRAIN} for a in ALPHAS}
    per_model = {}
    for k in KEYS:
        M = models[k]
        e = M["err"][j]
        ok = np.isfinite(e)
        Hans, _ = load_hidden(k)
        L = int(round(PROBE_DEPTH * (Hans.shape[1] - 1)))
        H = Hans[:, L].astype(np.float32)
        del Hans
        per_model[k] = {}
        for sd, (cal, tst) in enumerate(SP):
            grid, tst = grid_split(tst, sd)
            cal, tst = cal[ok[cal]], tst[ok[tst]]
            perm = np.random.default_rng(200 + sd).permutation(cal)
            preds = {}
            for n in TRAIN:
                tr, cc = np.sort(perm[:n]), np.sort(perm[n:])
                preds[n] = (cc, fit_predict(H[tr], e[tr], H[cc]),
                            fit_predict(H[tr], e[tr], H[grid]),
                            fit_predict(H[tr], e[tr], H[tst]))
            for a in ALPHAS:
                tau = certify(M["kle"][cal], e[cal], a, M["kle"][grid], proc)
                ck = apply_threshold(tau, M["kle"][tst], e[tst])[0]
                for n in TRAIN:
                    cc, p_cc, p_gr, p_ts = preds[n]
                    tau = certify(p_cc, e[cc], a, p_gr, proc)
                    cp = apply_threshold(tau, p_ts, e[tst])[0]
                    res[str(a)][str(n)]["kle"].append((k, sd, ck))
                    res[str(a)][str(n)]["probe"].append((k, sd, cp))
        print(f"  budget: {DISPLAY[k]} done", flush=True)
    out = {}
    for a in ALPHAS:
        out[str(a)] = {}
        for n in TRAIN:
            r = res[str(a)][str(n)]
            K = np.array([[c for kk, s, c in r["kle"] if kk == k] for k in KEYS])
            P = np.array([[c for kk, s, c in r["probe"] if kk == k] for k in KEYS])
            d = (P - K).mean(0)
            out[str(a)][str(n)] = {
                "kle": float(K.mean()), "probe": float(P.mean()),
                "diff": float(d.mean()),
                "diff_lo": float(np.quantile(d, .025)),
                "diff_hi": float(np.quantile(d, .975)),
                "splits_probe_ahead": int((d > 0).sum()),
                "models_probe_ahead": int(((P - K).mean(1) > 0).sum())}
            o = out[str(a)][str(n)]
            print(f"[budget {proc} a={a} train={n}/cal={500 - n}] KLE {o['kle']:.3f} "
                  f"probe {o['probe']:.3f} diff {o['diff']:+.3f} "
                  f"[{o['diff_lo']:+.3f},{o['diff_hi']:+.3f}] splits+ "
                  f"{o['splits_probe_ahead']}/20 models+ {o['models_probe_ahead']}/8",
                  flush=True)
    return out


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
    which = sys.argv[1:] or ["trunc", "proc", "transfer", "mondrian", "budget",
                             "budget_asc20", "budget_bonf"]
    p = os.path.join(OUT_DIR, "cert_sensitivity.json")
    out = json.load(open(p)) if os.path.exists(p) else {}
    if "trunc" in which:
        out["truncation"] = run_truncation(models, scores, SP)
    if "proc" in which:
        out["procedures"] = run_procedures(models, SP)
    if "transfer" in which:
        out["transfer"] = run_transfer(models, SP)
    if "mondrian" in which:
        out["mondrian"] = {p_: run_mondrian(models, SP, p_)
                           for p_ in ("bonf", "asc20", "asc05")}
    if "budget" in which:
        out["budget"] = run_budget(models, SP)
    for p_ in ("asc20", "bonf"):
        if f"budget_{p_}" in which:
            out[f"budget_{p_}"] = run_budget(models, SP, p_)
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
