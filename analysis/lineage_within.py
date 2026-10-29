"""Cross-model probe transfer, scored within task.

A probe trained on model A's calibration questions is tested on model B's
held-out questions (54% depth, 5 splits, J1; only pairs with the same hidden
size). Pooled AUROC can be lifted by task recognition alone, so transfer is
reported both pooled and within task.

Writes results/lineage_within.json.
Usage: python analysis/lineage_within.py
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
from task_control import within_task  # noqa: E402
from tiers import (N_PROBE_SPLITS, OUT_DIR, PROBE_DEPTH, auc,  # noqa: E402
                        fit_predict, load_hidden, load_model, splits)


def main():
    scores = np.load(SCORES)
    models = {k: load_model(k, scores) for k in KEYS}
    feats = {}
    for k in KEYS:
        H, _ = load_hidden(k)
        feats[k] = H[:, int(round(PROBE_DEPTH * (H.shape[1] - 1)))].astype(np.float32)
        del H
    SP = splits()[:N_PROBE_SPLITS]
    out = {}
    for ka in KEYS:
        for kb in KEYS:
            if ka == kb or feats[ka].shape[1] != feats[kb].shape[1]:
                continue
            ea, eb, tb = models[ka]["err"]["luna"], models[kb]["err"]["luna"], models[kb]["task"]
            pa, wa = [], []
            for cal, tst in SP:
                c = cal[np.isfinite(ea[cal])]
                t = tst[np.isfinite(eb[tst])]
                p = fit_predict(feats[ka][c], ea[c], feats[kb][t])
                pa.append(auc(eb[t], p))
                wa.append(within_task(eb[t], p, tb[t]))
            out[f"{ka}->{kb}"] = {"pooled": float(np.mean(pa)), "within": float(np.nanmean(wa))}
            print(f"{DISPLAY[ka]:10s} -> {DISPLAY[kb]:10s} pooled {np.mean(pa):.3f} "
                  f"within {np.nanmean(wa):.3f}", flush=True)
    p = os.path.join(OUT_DIR, "lineage_within.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
