"""Benchmark AUROC for the nine scores, both judges.

The semantic estimators (KLE, DegMat, NumSemSets, SE) are the lm-polygraph
classes in viuq/lmpg, called on a stats dict built from the stored
artifacts. MSP, PPL, MTE, MC-SE and MeanLength are read directly from the
generation arrays.

Conventions:
  * E and C are passed raw (not symmetrised). KLE symmetrises internally;
    NumSemSets uses them directionally.
  * semantic_matrix_classes is the per-pair argmax over [entail, neutral,
    contra], with neutral = 1 - E - C.
  * error = 1 - correctness and every score is "higher = more uncertain",
    so AUROC is roc_auc_score(error, score) with no sign flips.

Writes results/benchmark.json and results/benchmark_scores.npz (per-question
scores, read by every other analysis script). Run this first.

Usage: python analysis/benchmark.py
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import (DISPLAY, GEN_DIR, JUDGE_DIR, KEYS, N, NLI_DIR,  # noqa: E402
                         RESULTS, SCORES)

METHODS = ["KLE", "DegMat", "SE", "NumSets", "MTE", "PPL", "MC-SE", "MSP",
           "MeanLength"]


def labels(key, judge):
    """Correctness labels for one model, NaN where the judge gave none.

    Reads every judged_<judge>*.jsonl, so a judge run split by model
    (judged_gemini-phi4.jsonl, ...) is picked up as well as a single file.
    """
    y = np.full(N, np.nan)
    for p in sorted(glob.glob(os.path.join(JUDGE_DIR, f"judged_{judge}*.jsonl"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("set") == key and r.get("label") is not None:
                i = int(r["idx"])
                if 0 <= i < N:
                    y[i] = float(r["label"])
    return y


def build_stats(key, view="nli_base"):
    """The stats dict the lm-polygraph estimators consume, built from artifacts."""
    M = np.load(os.path.join(NLI_DIR, f"matrices_{key}.npz"))
    E = M[f"{view}_E"][:N].astype(np.float64)
    C = M[f"{view}_C"][:N].astype(np.float64)
    Nn = 1.0 - E - C
    classes = np.argmax(np.stack([E, Nn, C], axis=-1), axis=-1)

    rows = [json.loads(l) for l in
            open(os.path.join(GEN_DIR, f"gen_{key}.jsonl"))]
    rows.sort(key=lambda r: r["idx"])
    rows = rows[:N]
    A = np.load(os.path.join(GEN_DIR, f"arrays_{key}.npz"), allow_pickle=True)

    return {
        "semantic_matrix_entail": E,
        "semantic_matrix_contra": C,
        "semantic_matrix_classes": classes,
        "entailment_id": 0,
        "sample_texts": [r["samples"] for r in rows],
        "sample_log_probs": np.asarray(A["sample_seq_logprobs"][:N], float),
        "greedy_log_likelihoods": [list(x) for x in A["greedy_logprobs"][:N]],
        "greedy_entropies": [list(x) for x in A["greedy_entropies"][:N]],
        "n_tokens": np.array([r["greedy_n_tokens"] for r in rows], float),
    }


def main():
    from viuq.lmpg.kernel_language_entropy import KernelLanguageEntropy
    from viuq.lmpg.deg_mat import DegMat
    from viuq.lmpg.num_sem_sets import NumSemSets
    from viuq.lmpg.semantic_classes import SemanticClassesCalculator
    from viuq.lmpg.semantic_entropy import SemanticEntropy
    LIB = {"KLE": KernelLanguageEntropy(t=0.3),
           "DegMat": DegMat(similarity_score="NLI_score", affinity="entail"),
           "NumSets": NumSemSets()}
    CLASSES = SemanticClassesCalculator()
    SE = SemanticEntropy(class_probability_estimation="sum")

    per = {j: {} for j in ("luna", "gemini")}
    dump = {}
    for key in KEYS:
        st = build_stats(key)
        try:
            st.update(CLASSES(st, None, None))
        except Exception as e:                                    # noqa: BLE001
            print(f"  {key}: semantic_classes FAILED {str(e)[:80]}")
        scores = {}
        if "semantic_classes_entail" in st:
            try:
                scores["SE"] = np.asarray(SE(st), float)
            except Exception as e:                                # noqa: BLE001
                print(f"  {key} SE: FAILED {str(e)[:80]}")
        for name, est in LIB.items():
            try:
                scores[name] = np.asarray(est(st), float)
            except Exception as e:                                # noqa: BLE001
                print(f"  {key} {name}: FAILED {str(e)[:90]}")
        glp = st["greedy_log_likelihoods"]
        scores["MSP"] = np.array([-float(np.sum(x)) for x in glp])
        scores["PPL"] = np.array([-float(np.mean(x)) if len(x) else np.nan for x in glp])
        scores["MTE"] = np.array([float(np.mean(x)) if len(x) else np.nan
                                  for x in st["greedy_entropies"]])
        scores["MC-SE"] = -st["sample_log_probs"].mean(axis=1)
        scores["MeanLength"] = st["n_tokens"]
        for name, s in scores.items():
            dump[f"{key}/{name}"] = np.asarray(s, float)

        for j in ("luna", "gemini"):
            y = labels(key, j)
            for name, s in scores.items():
                ok = np.isfinite(s) & np.isfinite(y)
                if ok.sum() < 50:
                    continue
                err = (1.0 - y[ok] >= 0.5)
                if len(set(err)) < 2:
                    continue
                per[j].setdefault(name, {})[key] = float(roc_auc_score(err, s[ok]))
        print(f"  scored {key}", flush=True)

    out = {}
    for j in ("luna", "gemini"):
        print(f"\n=== BENCHMARK, judge = {j} (AUROC vs error) ===")
        hdr = f"{'method':11s}" + "".join(f"{DISPLAY[k][:6]:>8s}" for k in KEYS) \
              + f"{'MACRO':>8s}"
        print(hdr)
        print("-" * len(hdr))
        for m in METHODS:
            if m not in per[j]:
                continue
            v = [per[j][m].get(k, np.nan) for k in KEYS]
            macro = float(np.nanmean(v))
            out.setdefault(j, {})[m] = {"per_model": per[j][m], "macro": macro}
            print(f"{m:11s}" + "".join(f"{x:8.3f}" for x in v) + f"{macro:8.3f}")

    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "benchmark.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")
    ps = SCORES
    np.savez(ps, **dump)
    print(f"wrote {ps}")
    print("\nEstimators KLE / DegMat / NumSets are lm-polygraph classes called "
          "directly;\nMSP / PPL / MTE / MC-SE come from the stored arrays.")


if __name__ == "__main__":
    main()
