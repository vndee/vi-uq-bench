"""Shared protocol and the main certification / probe analyses.

Reads results/benchmark_scores.npz, the generation files (task types,
greedy log-probs and entropies, hidden states) and the judge labels.

Protocol, fixed before looking at test numbers:
  * 20 cal/test splits of the 1,200 questions (cal 500 / test 700). A split
    depends only on the seed, never on the model, so any cross-model
    transfer trains and tests on disjoint questions.
  * 200 of the 700 test questions per split form an unlabelled grid pool
    from which certification thresholds are taken.
  * Probes read the answer-averaged hidden state at 54% of the depth,
    fixed in advance; the layer sweep is reported but never used to choose.

The other analysis scripts import their splits, loaders and helpers from
here, so every number shares the same splits.

Writes results/tiers.json.
Usage: python analysis/tiers.py [--skip-probes]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import RESULTS, SCORES  # noqa: E402
from benchmark import DISPLAY, KEYS, N, GEN_DIR, labels  # noqa: E402

OUT_DIR = RESULTS
N_CAL = 500
N_SPLITS = 20
N_PROBE_SPLITS = 5
N_GRID = 200
ALPHAS = [0.1, 0.15, 0.2, 0.3]
DELTA = 0.1
COVERAGE_GRID = np.arange(0.05, 1.0001, 0.05)
PROBE_DEPTH = 0.54
B = 5000
JUDGES = ("luna", "gemini")
EPS = 1e-12


def auc(err, s):
    m = np.isfinite(s) & np.isfinite(err)
    if m.sum() < 10 or len(np.unique(err[m])) < 2:
        return float("nan")
    return float(roc_auc_score(err[m], s[m]))


def splits():
    """Question-level cal/test splits, identical for every model."""
    out = []
    for seed in range(N_SPLITS):
        perm = np.random.default_rng(seed).permutation(N)
        out.append((np.sort(perm[:N_CAL]), np.sort(perm[N_CAL:])))
    return out


def zfit(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)), float(np.nanstd(x) + 1e-9)


def zapply(x, mu_sd):
    z = (np.asarray(x, float) - mu_sd[0]) / mu_sd[1]
    return np.where(np.isfinite(z), z, 0.0)


def paired_bootstrap(err, a, b, rng):
    n = len(err)
    d = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        d[i] = auc(err[idx], a[idx]) - auc(err[idx], b[idx])
    d = d[np.isfinite(d)]
    return {"delta": 100 * (auc(err, a) - auc(err, b)),
            "lo": 100 * float(np.quantile(d, 0.025)),
            "hi": 100 * float(np.quantile(d, 0.975)),
            "p": float(np.mean(d <= 0))}


def load_model(key, scores):
    rows = [json.loads(l) for l in
            open(os.path.join(GEN_DIR, f"gen_{key}.jsonl"))]
    rows.sort(key=lambda r: r["idx"])
    task = np.array([r["task_type"] for r in rows[:N]])
    errs = {}
    for j in JUDGES:
        y = labels(key, j)
        errs[j] = np.where(np.isfinite(y), (1.0 - y >= 0.5).astype(float), np.nan)
    return {"task": task, "err": errs,
            "kle": scores[f"{key}/KLE"], "len": scores[f"{key}/MeanLength"]}


def _shape(x):
    dx = np.diff(x)
    return float(np.max(x) - np.min(x)), float(np.max(np.abs(dx)))


def _spec_entropy_centroid(x):
    if x.size < 4 or x.std() < EPS:
        return 0.0, 0.0
    xn = (x - x.mean()) / (x.std() + EPS)
    power = np.abs(np.fft.rfft(xn)) ** 2
    if power.size <= 1 or power[1:].sum() < EPS:
        return 0.0, 0.0
    power = power[1:] / (power[1:].sum() + EPS)
    freqs = np.arange(power.size) + 1
    return (float(-np.sum(power * np.log(power + EPS))),
            float(np.sum(freqs * power)))


def d_features(key):
    """(N, 4): four single-pass features of the greedy token-entropy curve
    (entropy complexity, and range / largest step / spectral centroid of
    the entropy minus min-entropy gap). NaN where the answer has < 3 tokens."""
    A = np.load(os.path.join(GEN_DIR, f"arrays_{key}.npz"), allow_pickle=True)
    F = np.full((N, 4), np.nan)
    for i, (lp, H) in enumerate(zip(A["greedy_logprobs"][:N],
                                    A["greedy_entropies"][:N])):
        lp = np.asarray(lp, float)
        H = np.asarray(H, float)
        if lp.size < 3 or H.size != lp.size:
            continue
        h_gap = H - (-lp)
        spec_ent, _ = _spec_entropy_centroid(H)
        _, hg_centroid = _spec_entropy_centroid(h_gap)
        hg_range, hg_maxdiff = _shape(h_gap)
        _, h_maxdiff = _shape(H)
        h_mad = float(np.mean(np.abs(np.diff(H))))
        F[i] = [H.mean() + 0.5 * spec_ent + 0.5 * h_mad,
                hg_maxdiff, hg_centroid, hg_range]
    return F


def d_score(F, err, cal):
    """Polarity + z from the cal split only; mean of oriented z-scores."""
    zs = []
    for j in range(F.shape[1]):
        f = F[:, j]
        a = auc(err[cal], f[cal])
        pol = -1.0 if np.isfinite(a) and a < 0.5 else 1.0
        zs.append(pol * zapply(f, zfit(f[cal])))
    return np.mean(zs, axis=0)


def fuse(a, b, cal):
    return 0.5 * (zapply(a, zfit(a[cal])) + zapply(b, zfit(b[cal])))


def cp_ucb(k, n, delta):
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def grid_split(tst, seed):
    """Carve N_GRID questions out of the test split as an UNLABELLED pool whose
    scores fix the LTT threshold grid. LTT's p-values need the hypothesis grid
    to be fixed before the calibration labels are seen; quantiles of the
    calibration scores themselves would make the grid a function of the
    calibration sample. The pool's labels are never read, and it is disjoint
    from the questions the certificate is evaluated on."""
    perm = np.random.default_rng(1000 + seed).permutation(tst)
    return np.sort(perm[:N_GRID]), np.sort(perm[N_GRID:])


def ltt_threshold(s, e, alpha, grid_scores):
    """Fixed-sequence Learn-then-Test with Clopper-Pearson. Candidate thresholds
    are quantiles of `grid_scores` (an unlabelled pool, see grid_split), tested
    in order of increasing retention; stop at the first failure."""
    chosen = None
    for tau in np.quantile(grid_scores, COVERAGE_GRID):
        r = s <= tau
        if cp_ucb(int(e[r].sum()), int(r.sum()), DELTA) <= alpha:
            chosen = float(tau)
        else:
            break
    return chosen


def apply_threshold(tau, s, e):
    if tau is None:
        return 0.0, float("nan")
    r = s <= tau
    return float(r.mean()), (float(e[r].mean()) if r.any() else float("nan"))


def tier1(models, SP, score_key="kle"):
    res = {}
    for k, M in models.items():
        e = M["err"]["luna"]
        s = M[score_key]
        ok = np.isfinite(e) & np.isfinite(s)
        per = {}
        for a in ALPHAS:
            cov, viol = [], []
            for sd, (cal, tst) in enumerate(SP):
                grid, tst = grid_split(tst, sd)
                cal, tst = cal[ok[cal]], tst[ok[tst]]
                tau = ltt_threshold(s[cal], e[cal], a, s[grid])
                c, se = apply_threshold(tau, s[tst], e[tst])
                cov.append(c)
                viol.append(float(np.isfinite(se) and se > a))
            per[str(a)] = {"coverage": float(np.mean(cov)),
                           "violation": float(np.mean(viol)),
                           "n_violations": int(np.sum(viol)),
                           "n_splits": len(viol)}
        res[k] = {"base_err": float(np.nanmean(e)), "alphas": per}
    return res


def tier1_transfer(models, SP, alpha=0.2):
    """Threshold calibrated on model A's cal questions, applied to model B's
    test questions (disjoint by construction)."""
    out = {}
    for ka, A in models.items():
        for kb, Bm in models.items():
            if ka == kb:
                continue
            v = []
            for sd, (cal, tst) in enumerate(SP):
                ea, sa = A["err"]["luna"], A["kle"]
                eb, sb = Bm["err"]["luna"], Bm["kle"]
                grid, tst = grid_split(tst, sd)
                c = cal[np.isfinite(ea[cal])]
                t = tst[np.isfinite(eb[tst])]
                tau = ltt_threshold(sa[c], ea[c], alpha, sa[grid])
                _, se = apply_threshold(tau, sb[t], eb[t])
                v.append(float(np.isfinite(se) and se > alpha))
            out[f"{ka}->{kb}"] = float(np.mean(v))
    return {"pairs": out, "mean_violation": float(np.mean(list(out.values())))}


def tier1_mondrian(models, SP, alpha=0.3):
    """Per-task selective error under ONE global (marginal) certificate."""
    tasks = sorted(set(next(iter(models.values()))["task"]))
    agg = {t: {"err": [], "viol": []} for t in tasks}
    for M in models.values():
        e, s = M["err"]["luna"], M["kle"]
        for sd, (cal, tst) in enumerate(SP):
            grid, tst = grid_split(tst, sd)
            c = cal[np.isfinite(e[cal])]
            t = tst[np.isfinite(e[tst])]
            tau = ltt_threshold(s[c], e[c], alpha, s[grid])
            if tau is None:
                continue
            for task in tasks:
                tt = t[M["task"][t] == task]
                r = s[tt] <= tau
                if r.sum() == 0:
                    continue
                se = float(e[tt][r].mean())
                agg[task]["err"].append(se)
                agg[task]["viol"].append(float(se > alpha))
    return {t: {"sel_err": float(np.mean(v["err"])) if v["err"] else None,
                "violation": float(np.mean(v["viol"])) if v["viol"] else None}
            for t, v in agg.items()}


def tier2(models, D, SP):
    rng = np.random.default_rng(0)
    full = np.arange(N)
    per, tasks_delta = {}, {}
    for k, M in models.items():
        row = {}
        for j in JUDGES:
            e = M["err"][j]
            ok = np.isfinite(e)
            idx = full[ok]
            d = d_score(D[k], e, idx)
            fused = fuse(M["kle"], d, idx)
            klelen = fuse(M["kle"], M["len"].astype(float), idx)
            r = {"kle": auc(e[idx], M["kle"][idx]), "d": auc(e[idx], d[idx]),
                 "fused": auc(e[idx], fused[idx]),
                 "rho_d_kle": float(spearmanr(d[idx], M["kle"][idx])[0]),
                 "vs_kle": paired_bootstrap(e[idx], fused[idx], M["kle"][idx], rng),
                 "vs_kle_len": paired_bootstrap(e[idx], fused[idx], klelen[idx], rng)}
            t_k, t_f, t_a, t_o, fuse_n = [], [], [], [], 0
            for cal, tst in SP:
                cal, tst = cal[ok[cal]], tst[ok[tst]]
                dd = d_score(D[k], e, cal)
                ff = fuse(M["kle"], dd, cal)
                go = auc(e[cal], ff[cal]) > auc(e[cal], M["kle"][cal])
                fuse_n += int(go)
                ak, af = auc(e[tst], M["kle"][tst]), auc(e[tst], ff[tst])
                t_k.append(ak)
                t_f.append(af)
                t_a.append(af if go else ak)
                t_o.append(max(ak, af))
            r["adaptive"] = {"always_kle": float(np.mean(t_k)),
                             "always_fuse": float(np.mean(t_f)),
                             "adaptive": float(np.mean(t_a)),
                             "oracle": float(np.mean(t_o)),
                             "fuse_rate": fuse_n / len(SP)}
            if j == "luna":
                for task in sorted(set(M["task"])):
                    m = ok & (M["task"] == task)
                    tasks_delta.setdefault(task, []).append(
                        100 * (auc(e[m], fused[m]) - auc(e[m], M["kle"][m])))
            row[j] = r
        per[k] = row
    macro = {}
    for j in JUDGES:
        macro[j] = {f: float(np.mean([per[k][j][f] for k in per]))
                    for f in ("kle", "d", "fused")}
        macro[j].update({f"adaptive_{f}": float(np.mean(
            [per[k][j]["adaptive"][f] for k in per]))
            for f in ("always_kle", "always_fuse", "adaptive", "oracle")})
    return {"per_model": per, "macro": macro,
            "per_task_delta_luna": {t: float(np.mean(v))
                                    for t, v in tasks_delta.items()}}


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.05, max_iter=2000)
    lr.fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1]


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def load_hidden(key):
    A = np.load(os.path.join(GEN_DIR, f"arrays_{key}.npz"), allow_pickle=True)
    return A["hidden_answer_mean"][:N], A["hidden_last_prompt"][:N]


def tier3_model(k, M, SP):
    """Everything that needs only this model's hidden states."""
    Hans, Hpr = load_hidden(k)
    n_layers = Hans.shape[1]
    L = int(round(PROBE_DEPTH * (n_layers - 1)))
    e = M["err"]["luna"]
    ok = np.isfinite(e)
    r = {"n_layers": n_layers, "layer": L, "dim": int(Hans.shape[2])}

    sweep = []
    for li in range(n_layers):
        a = []
        for cal, tst in SP[:N_PROBE_SPLITS]:
            cal, tst = cal[ok[cal]], tst[ok[tst]]
            a.append(auc(e[tst], fit_predict(Hans[cal, li], e[cal], Hans[tst, li])))
        sweep.append(float(np.mean(a)))
    r["sweep"] = sweep

    acc = {n: [] for n in ("sup", "sup_prompt", "kle_test", "labelfree",
                           "sup_plus_kle", "ece_sup")}
    for cal, tst in SP[:N_PROBE_SPLITS]:
        cal, tst = cal[ok[cal]], tst[ok[tst]]
        p = fit_predict(Hans[cal, L], e[cal], Hans[tst, L])
        acc["sup"].append(auc(e[tst], p))
        acc["ece_sup"].append(ece(p, e[tst]))
        acc["sup_prompt"].append(auc(e[tst], fit_predict(Hpr[cal, L], e[cal],
                                                         Hpr[tst, L])))
        acc["kle_test"].append(auc(e[tst], M["kle"][tst]))
        yk = (M["kle"] > np.median(M["kle"][cal])).astype(float)
        acc["labelfree"].append(auc(e[tst], fit_predict(Hans[cal, L], yk[cal],
                                                        Hans[tst, L])))
        pp = np.full(N, np.nan)
        pp[tst] = p
        f = fuse(pp, M["kle"], tst)
        acc["sup_plus_kle"].append(auc(e[tst], f[tst]))
    r.update({n: float(np.mean(v)) for n, v in acc.items()})

    budget = {}
    for nb in (50, 100, 200, 300, 500):
        a = []
        for s, (cal, tst) in enumerate(SP[:N_PROBE_SPLITS]):
            cal, tst = cal[ok[cal]], tst[ok[tst]]
            tr = np.random.default_rng(100 + s).choice(cal, min(nb, len(cal)),
                                                       replace=False)
            if len(np.unique(e[tr])) < 2:
                continue
            a.append(auc(e[tst], fit_predict(Hans[tr, L], e[tr], Hans[tst, L])))
        budget[str(nb)] = float(np.mean(a))
    r["budget"] = budget

    loto = {}
    for task in sorted(set(M["task"])):
        te = np.where(ok & (M["task"] == task))[0]
        tr = np.where(ok & (M["task"] != task))[0]
        loto[task] = {"probe": auc(e[te], fit_predict(Hans[tr, L], e[tr],
                                                      Hans[te, L])),
                      "kle": auc(e[te], M["kle"][te])}
    r["loto"] = loto
    r["loto_macro_probe"] = float(np.nanmean([v["probe"] for v in loto.values()]))
    r["loto_macro_kle"] = float(np.nanmean([v["kle"] for v in loto.values()]))

    cov_k, cov_p, auc_p = [], [], []
    for sd, (cal, tst) in enumerate(SP):
        grid, tst = grid_split(tst, sd)
        cal, tst = cal[ok[cal]], tst[ok[tst]]
        tau = ltt_threshold(M["kle"][cal], e[cal], 0.2, M["kle"][grid])
        cov_k.append(apply_threshold(tau, M["kle"][tst], e[tst])[0])
        perm = np.random.default_rng(200 + sd).permutation(cal)
        tr, cc = np.sort(perm[:200]), np.sort(perm[200:])
        p_cc = fit_predict(Hans[tr, L], e[tr], Hans[cc, L])
        p_gr = fit_predict(Hans[tr, L], e[tr], Hans[grid, L])
        p_ts = fit_predict(Hans[tr, L], e[tr], Hans[tst, L])
        tau = ltt_threshold(p_cc, e[cc], 0.2, p_gr)
        cov_p.append(apply_threshold(tau, p_ts, e[tst])[0])
        auc_p.append(auc(e[tst], p_ts))
    r["budget_tradeoff_a20"] = {"kle_500cal": float(np.mean(cov_k)),
                                "probe_200train_300cal": float(np.mean(cov_p)),
                                "probe_200train_auc": float(np.mean(auc_p))}
    return r, Hans[:, L]


def tier3_transfer(models, feats, SP):
    """Cross-model transfer at fixed depth, only where hidden sizes match.
    Train on A's cal questions, test on B's test questions."""
    out = {}
    for ka in models:
        for kb in models:
            if ka == kb or feats[ka].shape[1] != feats[kb].shape[1]:
                continue
            ea, eb = models[ka]["err"]["luna"], models[kb]["err"]["luna"]
            a = []
            for cal, tst in SP[:N_PROBE_SPLITS]:
                c = cal[np.isfinite(ea[cal])]
                t = tst[np.isfinite(eb[tst])]
                a.append(auc(eb[t], fit_predict(feats[ka][c], ea[c], feats[kb][t])))
            out[f"{ka}->{kb}"] = float(np.mean(a))
    return out


def per_task_benchmark(models, scores, methods=("KLE", "DegMat", "MTE", "MSP",
                                                "MeanLength")):
    """Macro-over-models AUROC per task (judge luna)."""
    tasks = sorted(set(next(iter(models.values()))["task"]))
    out = {}
    for t in tasks:
        out[t] = {"err_rate": float(np.mean([np.nanmean(M["err"]["luna"][M["task"] == t])
                                             for M in models.values()]))}
        for m in methods:
            v = []
            for k, M in models.items():
                sel = (M["task"] == t) & np.isfinite(M["err"]["luna"])
                v.append(auc(M["err"]["luna"][sel], scores[f"{k}/{m}"][sel]))
            out[t][m] = float(np.nanmean(v))
    return out


def judge_agreement(models):
    def kappa(a, b):
        po = np.mean(a == b)
        pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
        return float((po - pe) / (1 - pe))
    A, Bj, T = [], [], []
    per_model = {}
    for k, M in models.items():
        ok = np.isfinite(M["err"]["luna"]) & np.isfinite(M["err"]["gemini"])
        a, b = M["err"]["luna"][ok], M["err"]["gemini"][ok]
        per_model[k] = {"agree": float(np.mean(a == b)), "kappa": kappa(a, b),
                        "correct_luna": float(1 - a.mean()),
                        "correct_gemini": float(1 - b.mean())}
        A.append(a)
        Bj.append(b)
        T.append(M["task"][ok])
    A, Bj, T = map(np.concatenate, (A, Bj, T))
    per_task = {t: {"agree": float(np.mean(A[T == t] == Bj[T == t])),
                    "kappa": kappa(A[T == t], Bj[T == t]),
                    "gemini_lenient_share": float(np.mean(
                        (A[T == t] == 1)[A[T == t] != Bj[T == t]]))
                    if np.any(A[T == t] != Bj[T == t]) else None}
                for t in sorted(set(T))}
    return {"pooled": {"n": int(len(A)), "agree": float(np.mean(A == Bj)),
                       "kappa": kappa(A, Bj)},
            "per_model": per_model, "per_task": per_task}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-probes", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = splits()
    out = {"protocol": {"n_cal": N_CAL, "n_splits": N_SPLITS, "delta": DELTA,
                        "probe_depth": PROBE_DEPTH, "B": B,
                        "primary_judge": "luna"}}

    out["per_task_benchmark"] = per_task_benchmark(models, scores)
    out["judge_agreement"] = judge_agreement(models)
    print("judges:", json.dumps(out["judge_agreement"]["pooled"]))
    for t, v in out["per_task_benchmark"].items():
        print(f"  {t:26s} err {v['err_rate']:.2f} KLE {v['KLE']:.3f} "
              f"MTE {v['MTE']:.3f} len {v['MeanLength']:.3f}  "
              f"judge-agree {out['judge_agreement']['per_task'][t]['agree']:.3f}")

    t1 = tier1(models, SP)
    out["tier1_kle"] = t1
    print("\n=== TIER 1: certified coverage (KLE, LTT+CP, delta=0.1) ===")
    print(f"{'model':11s}{'err':>7s}" + "".join(f"{'a='+str(a):>12s}" for a in ALPHAS))
    for k, v in sorted(t1.items(), key=lambda kv: kv[1]["base_err"]):
        print(f"{DISPLAY[k]:11s}{v['base_err']:7.3f}" + "".join(
            f"{v['alphas'][str(a)]['coverage']:6.2f}/{v['alphas'][str(a)]['violation']:.2f}"
            .rjust(12) for a in ALPHAS))
    print("self-violation mean:", np.mean([v["alphas"][str(a)]["violation"]
                                         for v in t1.values() for a in ALPHAS]))
    nt = [(v["alphas"][str(a)]["n_violations"], v["alphas"][str(a)]["n_splits"])
          for v in t1.values() for a in ALPHAS
          if v["alphas"][str(a)]["coverage"] > 0 and v["base_err"] > a]
    kv, nv = sum(x for x, _ in nt), sum(n for _, n in nt)
    out["tier1_nontrivial"] = {"cells": len(nt), "violations": kv, "draws": nv,
                               "rate": kv / nv,
                               "cp95_upper": cp_ucb(kv, nv, 0.05)}
    print("non-trivial cells:", json.dumps(out["tier1_nontrivial"]))
    out["tier1_transfer"] = tier1_transfer(models, SP)
    print("cross-model violation @0.2:", out["tier1_transfer"]["mean_violation"])
    out["tier1_mondrian"] = tier1_mondrian(models, SP)
    print("per-task selective error @ global a=0.3:")
    for t, v in sorted(out["tier1_mondrian"].items(),
                       key=lambda kv: kv[1]["sel_err"] or 0):
        print(f"  {t:26s} {v['sel_err']:.3f}  viol {v['violation']:.2f}")

    D = {k: d_features(k) for k in KEYS}
    out["d_valid_frac"] = {k: float(np.isfinite(D[k][:, 0]).mean()) for k in KEYS}
    t2 = tier2(models, D, SP)
    out["tier2"] = t2
    print("\n=== TIER 2: D+KLE vs KLE (judge luna; full-set bootstrap) ===")
    for k in KEYS:
        r = t2["per_model"][k]["luna"]
        b = r["vs_kle"]
        print(f"{DISPLAY[k]:11s} KLE {r['kle']:.3f} D {r['d']:.3f} fused "
              f"{r['fused']:.3f}  d={b['delta']:+.1f} [{b['lo']:+.1f},{b['hi']:+.1f}]"
              f" p={b['p']:.3f} rho={r['rho_d_kle']:.2f} "
              f"fuse%={100*r['adaptive']['fuse_rate']:.0f}")
    print("macro:", json.dumps(t2["macro"], indent=1))

    if not args.skip_probes:
        t3, feats = {}, {}
        for k in KEYS:
            t3[k], feats[k] = tier3_model(k, models[k], SP)
            r = t3[k]
            print(f"  probe {DISPLAY[k]:10s} L{r['layer']}/{r['n_layers']} "
                  f"sup {r['sup']:.3f} prompt {r['sup_prompt']:.3f} "
                  f"KLE {r['kle_test']:.3f} labelfree {r['labelfree']:.3f} "
                  f"+KLE {r['sup_plus_kle']:.3f} LOTO {r['loto_macro_probe']:.3f}"
                  f"/{r['loto_macro_kle']:.3f} best-layer "
                  f"{int(np.argmax(r['sweep']))}", flush=True)
        out["tier3"] = t3
        out["tier3_transfer"] = tier3_transfer(models, feats, SP)
        for p, v in sorted(out["tier3_transfer"].items()):
            print(f"  transfer {p:22s} {v:.3f}")

    p = os.path.join(OUT_DIR, "tiers.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
