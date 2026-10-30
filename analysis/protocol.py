"""Shared protocol: splits, loaders, AUROC and the LTT certificate.

Fixed before looking at test numbers:
  * 20 cal/test splits of the 1,200 questions (cal 500 / test 700). A split
    depends only on the seed, never on the model, so any cross-model
    transfer calibrates and tests on disjoint questions.
  * 200 of the 700 test questions per split form an unlabelled grid pool
    from which certification thresholds are taken.

Every analysis script imports its splits, loaders and helpers from here, so
every number shares the same splits.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta, rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import GEN_DIR, RESULTS  # noqa: E402
from benchmark import N, labels  # noqa: E402

OUT_DIR = RESULTS
N_CAL = 500
N_SPLITS = 20
N_GRID = 200
DELTA = 0.1
COVERAGE_GRID = np.arange(0.05, 1.0001, 0.05)
JUDGES = ("luna", "gemini")


def auc(err, s):
    m = np.isfinite(s) & np.isfinite(err)
    if m.sum() < 10 or len(np.unique(err[m])) < 2:
        return float("nan")
    return float(roc_auc_score(err[m], s[m]))


def fauc(y, s):
    """Mann-Whitney AUROC without a minimum size; nan if a class is missing."""
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def splits():
    """Question-level cal/test splits, identical for every model."""
    out = []
    for seed in range(N_SPLITS):
        perm = np.random.default_rng(seed).permutation(N)
        out.append((np.sort(perm[:N_CAL]), np.sort(perm[N_CAL:])))
    return out


def grid_split(tst, seed):
    """Carve N_GRID questions out of the test split as an UNLABELLED pool whose
    scores fix the LTT threshold grid. LTT's p-values need the hypothesis grid
    to be fixed before the calibration labels are seen; quantiles of the
    calibration scores themselves would make the grid a function of the
    calibration sample. The pool's labels are never read, and it is disjoint
    from the questions the certificate is evaluated on."""
    perm = np.random.default_rng(1000 + seed).permutation(tst)
    return np.sort(perm[:N_GRID]), np.sort(perm[N_GRID:])


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


def cp_ucb(k, n, delta):
    """One-sided Clopper-Pearson upper bound on an error rate."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def _passes(s, e, tau, alpha, delta):
    r = s <= tau
    return cp_ucb(int(e[r].sum()), int(r.sum()), delta) <= alpha


def certify(s, e, alpha, grid_scores, proc, delta=DELTA):
    """Largest certified threshold (None if nothing is certified).

    Candidate thresholds are quantiles of `grid_scores` at the 20 retention
    levels of COVERAGE_GRID.
      asc20 / asc05  fixed sequence in order of increasing retention, from
                     20% (the paper's procedure) or 5%; stop at the first
                     failure
      desc           a single test at 100% retention
      bonf           Bonferroni over the 20 levels
    """
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


def apply_threshold(tau, s, e):
    """(coverage, selective error) of answering the questions with s <= tau."""
    if tau is None:
        return 0.0, float("nan")
    r = s <= tau
    return float(r.mean()), (float(e[r].mean()) if r.any() else float("nan"))
