"""Second set of checks added after review (judge J1 unless stated).

  bonferroni    test exceedance of Bonferroni over the 20 levels vs. the
                fixed sequence, 500 splits, over the same (model, alpha)
                cells as split_robustness.py
  transfer_pool threshold transfer at alpha 0.2 (non-empty certificates),
                selective error measured on the target model's full pool
                instead of its test split
  two_level     intervals that include the sampling of the questions: an
                outer bootstrap over questions (shared across models, B =
                200) with 20 splits redrawn inside each replicate, for the
                self-calibrated pool exceedance, threshold transfer and
                per-task exceedance
  combine       scores that use the task: TaskPrior plus within-task
                standardized KLE (logistic regression fitted on the
                calibration split), and a per-task choice of score (the
                best of five scores on the calibration split); pooled and
                within-task AUROC on the test split, 20 splits, both judges
  other_scores  the marginal certificate at alpha 0.2 built on MSP or MTE
                instead of KLE: coverage and per-task exceedance on NER,
                unanswerable and multi-hop; and a per-task fixed sequence
                from 50% retention on NER with each score, 500 splits

Writes results/review_checks2.json.
Usage: python analysis/review_checks2.py
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import SCORES  # noqa: E402
from benchmark import KEYS, N  # noqa: E402
from protocol import (JUDGES, OUT_DIR, apply_threshold, auc, certify,  # noqa: E402
                      grid_split, load_model)
from protocol import splits as splits20  # noqa: E402
import split_robustness as srb  # noqa: E402
from review_checks import fixed_sequence  # noqa: E402

J = "luna"
A = 0.2
N_SPLITS = 500
B_OUTER = 200
HARD_TASKS = srb.HARD_TASKS
CHOICE = ("KLE", "DegMat", "SE", "MTE", "MSP")


def with_score(models, scores, m):
    """Copy of the models with the certified score replaced by `m`."""
    return {k: dict(M, kle=scores[f"{k}/{m}"].astype(float)) for k, M in models.items()}


def run_bonferroni(models, SP, rng):
    S = len(SP)
    out = {}
    for proc in ("asc20", "bonf"):
        num, den = np.zeros(S), np.zeros(S)
        for k in KEYS:
            M = models[k]
            e, s = M["err"][J], M["kle"]
            ok = np.isfinite(e)
            if np.nanmean(e) <= 0.1:
                continue
            for a in srb.ALPHAS:
                if np.nanmean(e) <= a:
                    continue
                n_ne, ex = 0, np.zeros(S)
                for sd, (cal, tst) in enumerate(SP):
                    grid, test = grid_split(tst, sd)
                    c, t = cal[ok[cal]], test[ok[test]]
                    if certify(s[c], e[c], a, s[grid], "asc20") is not None:
                        n_ne += 1
                    tau = certify(s[c], e[c], a, s[grid], proc)
                    if tau is None:
                        continue
                    _, se = apply_threshold(tau, s[t], e[t])
                    ex[sd] = float(np.isfinite(se) and se > a)
                if n_ne == 0:
                    continue
                num += ex
                den += 1
        out[proc] = srb.ratio_ci(num, den, rng)
        v = out[proc]
        print(f"  test exceedance {proc:5s} {v['rate']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] "
              f"({v['num']}/{v['den']})", flush=True)
    return out


def run_transfer_pool(models, SP, rng):
    base = {k: float(np.nanmean(models[k]["err"][J])) for k in KEYS}
    hard = [k for k in KEYS if base[k] > A]
    S = len(SP)
    taus = {}
    for ka in KEYS:
        M = models[ka]
        ok = np.isfinite(M["err"][J])
        for sd, (cal, tst) in enumerate(SP):
            grid, _ = grid_split(tst, sd)
            c = cal[ok[cal]]
            taus[(ka, sd)] = certify(M["kle"][c], M["err"][J][c], A, M["kle"][grid], "asc20")
    r = {arm: {"ne": np.zeros(S), "exc": np.zeros(S)} for arm in ("self", "lower", "higher")}
    for kb in hard:
        Bm = models[kb]
        okb = np.isfinite(Bm["err"][J])
        for ka in KEYS:
            arm = "self" if ka == kb else "lower" if base[ka] < base[kb] else "higher"
            for sd in range(S):
                tau = taus[(ka, sd)]
                if tau is None:
                    continue
                _, se = apply_threshold(tau, Bm["kle"][okb], Bm["err"][J][okb])
                if not np.isfinite(se):
                    continue
                r[arm]["ne"][sd] += 1
                r[arm]["exc"][sd] += int(se > A)
    out = {arm: srb.ratio_ci(d["exc"], d["ne"], rng) for arm, d in r.items()}
    for arm, v in out.items():
        print(f"  transfer on pool {arm:6s} {v['rate']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] "
              f"({v['num']}/{v['den']})", flush=True)
    return out


def resample(models, idx):
    return {k: {"task": M["task"][idx], "kle": M["kle"][idx], "len": M["len"][idx],
                "err": {j: M["err"][j][idx] for j in M["err"]}}
            for k, M in models.items()}


def run_two_level(models, rng):
    """Outer bootstrap over questions; the same 20 split seeds inside."""
    B_inner = srb.B
    srb.B = 1   # only the point estimate is needed inside a replicate
    SP = splits20()
    rows = []
    devnull = open(os.devnull, "w")
    for b in range(B_OUTER):
        idx = rng.integers(0, N, N)
        mb = resample(models, idx)
        stdout, sys.stdout = sys.stdout, devnull
        try:
            sf = srb.run_self(mb, SP, np.random.default_rng(b))
            tr = srb.run_transfer(mb, SP, np.random.default_rng(b))
            pt = srb.run_per_task(mb, SP, np.random.default_rng(b))
        finally:
            sys.stdout = stdout
        rows.append({"pool_all_draws": sf["all_alphas"]["pool_all_draws"]["rate"],
                     "transfer_self": tr["all"]["self"]["rate"],
                     "transfer_lower": tr["all"]["lower"]["rate"],
                     "transfer_higher": tr["all"]["higher"]["rate"],
                     "transfer_lower_no_vinallama": tr["without_vinallama"]["lower"]["rate"],
                     **{f"per_task_{t}": pt[t]["rate"] for t in HARD_TASKS}})
        if (b + 1) % 20 == 0:
            print(f"  two-level bootstrap {b + 1}/{B_OUTER}", flush=True)
    srb.B = B_inner
    out = {}
    for key in rows[0]:
        v = np.array([r[key] for r in rows], float)
        v = v[np.isfinite(v)]
        out[key] = {"lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975)),
                    "n": int(len(v))}
        print(f"  two-level {key:30s} [{out[key]['lo']:.3f}, {out[key]['hi']:.3f}]")
    return out


def zscore_within(s, task):
    z = np.zeros_like(s)
    for t in np.unique(task):
        w = task == t
        z[w] = (s[w] - np.nanmean(s[w])) / (np.nanstd(s[w]) + 1e-9)
    return z


def run_combine(models, scores):
    SP = splits20()
    tasks = sorted(set(models[KEYS[0]]["task"]))
    out = {}
    for j in JUDGES:
        res = {k: [] for k in ("TaskPrior", "KLE", "TaskPrior+KLE", "per_task_choice",
                               "TaskPrior+choice", "KLE_within", "choice_within")}
        for k in KEYS:
            M = models[k]
            e, task = M["err"][j], M["task"]
            Z = {m: zscore_within(scores[f"{k}/{m}"].astype(float), task) for m in CHOICE}
            acc = {key: [] for key in res}
            for cal, tst in SP:
                c, t = cal[np.isfinite(e[cal])], tst[np.isfinite(e[tst])]
                prior = {tt: float(e[c][task[c] == tt].mean()) for tt in tasks}
                tp = np.array([prior[tt] for tt in task])
                best = {}
                for tt in tasks:
                    cc = c[task[c] == tt]
                    a = {m: auc(e[cc], Z[m][cc]) for m in CHOICE}
                    a = {m: v for m, v in a.items() if np.isfinite(v)}
                    best[tt] = max(a, key=a.get) if a else "KLE"
                ch = np.array([Z[best[tt]][i] for i, tt in enumerate(task)])
                for name, feats in (("TaskPrior+KLE", np.c_[tp, Z["KLE"]]),
                                    ("TaskPrior+choice", np.c_[tp, ch])):
                    lr = LogisticRegression().fit(feats[c], e[c])
                    acc[name].append(auc(e[t], lr.decision_function(feats[t])))
                acc["TaskPrior"].append(auc(e[t], tp[t]))
                acc["KLE"].append(auc(e[t], scores[f"{k}/KLE"][t].astype(float)))
                acc["per_task_choice"].append(auc(e[t], ch[t]))
                for name, sc in (("KLE_within", Z["KLE"]), ("choice_within", ch)):
                    per = [auc(e[t][task[t] == tt], sc[t][task[t] == tt]) for tt in tasks]
                    acc[name].append(float(np.nanmean(per)))
            for key in res:
                res[key].append(float(np.mean(acc[key])))
        out[j] = {key: {"macro": float(np.mean(v)), "per_model": dict(zip(KEYS, v))}
                  for key, v in res.items()}
        print(f"  combine {j}: " + "  ".join(f"{key} {v['macro']:.3f}"
                                              for key, v in out[j].items()))
    return out


def run_other_scores(models, scores, SP, rng):
    out = {}
    for m in ("KLE", "MSP", "MTE"):
        mm = with_score(models, scores, m)
        S = len(SP)
        num = {t: np.zeros(S) for t in HARD_TASKS}
        den = {t: np.zeros(S) for t in HARD_TASKS}
        cov, ner_cert, ner_draws = [], 0, 0
        for k in KEYS:
            M = mm[k]
            e, s, task = M["err"][J], M["kle"], M["task"]
            ok = np.isfinite(e)
            for sd, (cal, tst) in enumerate(SP):
                grid, test = grid_split(tst, sd)
                c, t = cal[ok[cal]], test[ok[test]]
                tau = certify(s[c], e[c], A, s[grid], "asc20")
                cv, _ = apply_threshold(tau, s[t], e[t])
                cov.append(cv)
                if tau is not None:
                    for tk in HARD_TASKS:
                        tt = t[task[t] == tk]
                        _, se = apply_threshold(tau, s[tt], e[tt])
                        if np.isfinite(se):
                            den[tk][sd] += 1
                            num[tk][sd] += int(se > A)
                cn, gn = c[task[c] == "ner"], grid[task[grid] == "ner"]
                ner_draws += 1
                ner_cert += fixed_sequence(s[cn], e[cn], A, s[gn], 0.5, 0.1) is not None
        out[m] = {"coverage": float(np.mean(cov)),
                  "exceed": {tk: srb.ratio_ci(num[tk], den[tk], rng) for tk in HARD_TASKS},
                  "ner_per_task_fs50_certified": ner_cert, "draws": ner_draws}
        print(f"  marginal on {m}: coverage {out[m]['coverage']:.3f}  " + "  ".join(
            f"{tk[:5]} {v['rate']:.2f}" for tk, v in out[m]["exceed"].items())
            + f"  | NER per-task fs50 certified {ner_cert}/{ner_draws}", flush=True)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    SP = srb.splits(N_SPLITS)
    rng = np.random.default_rng(0)
    out = {"n_splits": N_SPLITS,
           "combine": run_combine(models, scores),
           "other_scores": run_other_scores(models, scores, SP, rng),
           "bonferroni": run_bonferroni(models, SP, rng),
           "transfer_pool": run_transfer_pool(models, SP, rng),
           "two_level": run_two_level(models, np.random.default_rng(1))}
    p = os.path.join(OUT_DIR, "review_checks2.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
