"""Pairwise NLI matrices over each question's sampled answers.

For every question, the K sampled answers are compared in both directions
with a multilingual NLI cross-encoder. The output keeps the raw entailment
and contradiction probabilities (E, C), not only the combined weight matrix,
because the semantic estimators each apply their own convention: KLE and
DegMat symmetrise them, NumSemSets uses them directionally.

  W = E_sym + 0.5 * (1 - E_sym - C_sym)

The diagonal is left as the model's own self-entailment probability; it is
not forced to 1.

Reads  artifacts/generations/gen_<key>.jsonl
Writes artifacts/nli/matrices_<key>.npz  (nli_base, nli_base_E, nli_base_C)

Usage:
  python pipeline/nli_matrices.py                 # every model with a gen file
  python pipeline/nli_matrices.py --only phi4,qwen3
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viuq.config import GEN_DIR, NLI_DIR  # noqa: E402

NLI_BASE = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MAX_LENGTH = 512
BATCH = int(os.environ.get("BATCH", "256"))


def dev():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def nli_view(model_id, sample_sets, device):
    """(W, E, C), each (n_questions, K, K) float32.

    Identical answers are scored once and the result is copied back, so
    duplicates keep their multiplicity in the K x K grid.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=torch.float32).to(device).eval()
    l2i = {k.lower(): v for k, v in mdl.config.label2id.items()}
    ent = next(v for k, v in l2i.items() if "entail" in k)
    con = next(v for k, v in l2i.items() if "contra" in k)

    pairs, spans, invs, counts = [], [], [], []
    for texts in sample_sets:
        uniq, inv = np.unique(texts, return_inverse=True)
        p = list(itertools.product(uniq, uniq))
        spans.append((len(pairs), len(pairs) + len(p)))
        pairs.extend(p)
        invs.append(inv)
        counts.append(len(uniq))

    probs = np.empty((len(pairs), mdl.config.num_labels), dtype=np.float32)
    for s in range(0, len(pairs), BATCH):
        chunk = pairs[s:s + BATCH]
        enc = tok.batch_encode_plus(chunk, padding=True, truncation=True,
                                    max_length=MAX_LENGTH,
                                    return_tensors="pt").to(device)
        probs[s:s + len(chunk)] = torch.softmax(
            mdl(**enc).logits, dim=1).float().cpu().numpy()
        if (s // BATCH) % 50 == 0:
            print(f"      {s}/{len(pairs)}", flush=True)

    Ws, Es, Cs = [], [], []
    for i, (a, b) in enumerate(spans):
        k = counts[i]
        pr = probs[a:b].reshape(k, k, -1)
        inv = invs[i]
        E = pr[:, :, ent][inv, :][:, inv]
        C = pr[:, :, con][inv, :][:, inv]
        Es.append(E.astype(np.float32))
        Cs.append(C.astype(np.float32))
        Esym = (E + E.T) / 2.0
        Csym = (C + C.T) / 2.0
        Ws.append((Esym + 0.5 * (1.0 - Esym - Csym)).astype(np.float32))
    del mdl
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.stack(Ws), np.stack(Es), np.stack(Cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated model keys")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N questions only (for a quick check)")
    ap.add_argument("--out-dir", default=NLI_DIR)
    args = ap.parse_args()
    keep = set(args.only.split(",")) if args.only else None
    os.makedirs(args.out_dir, exist_ok=True)
    device = dev()
    print(f"device={device}", flush=True)

    for path in sorted(glob.glob(os.path.join(GEN_DIR, "gen_*.jsonl"))):
        key = os.path.basename(path)[len("gen_"):-len(".jsonl")]
        if keep and key not in keep:
            continue
        rows = [json.loads(l) for l in open(path)]
        rows.sort(key=lambda r: r["idx"])
        if args.limit:
            rows = rows[:args.limit]
        texts = [r["samples"] for r in rows]
        print(f"\n=== {key}: {len(texts)} questions ===", flush=True)
        W, E, C = nli_view(NLI_BASE, texts, device)
        out = os.path.join(args.out_dir, f"matrices_{key}.npz")
        np.savez_compressed(out, nli_base=W, nli_base_E=E, nli_base_C=C)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
