"""AttentionScore from a finished phase-a generation.

Separate from generate.py because attention weights need eager attention.
This reloads the model in eager mode and runs one teacher-forced forward
over prompt + greedy answer per question, reusing the exact greedy token
ids from phase a.

Follows lm-polygraph's AttentionScore:
    layer = num_hidden_layers // 2
    ue    = (1/H) * sum_h sum_t log(A[layer][h][t][t] + 1e-12)
    score = -ue
summed over the full prompt + answer sequence (gen_only=False). Answer-only
and length-averaged variants are stored next to it.

Usage:
  python pipeline/attention_pass.py --model-id microsoft/phi-4 --key phi4 \
    --system-prompt "..." [--template-kwargs '{"enable_thinking":false}']
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import build_prompt_ids, attention_score  # noqa: E402
from viuq.config import GEN_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--gen", default=None, help="default: gen_<key>.jsonl in GEN_DIR")
    ap.add_argument("--out", default=None, help="default: attn_<key>.npz in GEN_DIR")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--template-kwargs", default=None)
    args = ap.parse_args()
    args.gen = args.gen or os.path.join(GEN_DIR, f"gen_{args.key}.jsonl")
    args.out = args.out or os.path.join(GEN_DIR, f"attn_{args.key}.npz")

    rows = [json.loads(l) for l in open(args.gen) if l.strip()]
    if not rows:
        raise SystemExit(f"{args.gen} is empty")
    if "greedy_ids" not in rows[0]:
        raise SystemExit(
            "phase-a output lacks `greedy_ids`; re-run generate.py --phase a "
            "(re-tokenising the text would not give the generated ids)")
    tkw = json.loads(args.template_kwargs) if args.template_kwargs else None

    print(f"loading {args.model_id} with EAGER attention", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="cuda",
        attn_implementation="eager", trust_remote_code=True).eval()

    done = {}
    if os.path.exists(args.out):
        with np.load(args.out) as z:
            done = dict(zip(z["idx"].tolist(), z["attn_score"].tolist()))
        print(f"resuming: {len(done)} already computed", flush=True)

    idxs, scores = list(done.keys()), list(done.values())
    extra = {k: [] for k in ("attn_gen_only", "attn_full_mean", "attn_gen_mean")}
    for n, r in enumerate(rows, 1):
        if r["idx"] in done:
            continue
        pid = build_prompt_ids(tok, r["question"], "cuda",
                               args.system_prompt, tkw)
        try:
            d = attention_score(model, tok, pid, r["greedy_ids"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on idx {r['idx']} (seq too long) -> nan", flush=True)
            d = {k: float("nan") for k in
                 ("attn_score", "attn_gen_only", "attn_full_mean", "attn_gen_mean")}
        idxs.append(r["idx"])
        scores.append(d["attn_score"])
        for k in extra:
            extra[k].append(d[k])
        if n % 50 == 0:
            print(f"{n}/{len(rows)}", flush=True)
            np.savez_compressed(args.out, idx=np.asarray(idxs, np.int32),
                                attn_score=np.asarray(scores, np.float32),
                                **{k: np.asarray(v, np.float32)
                                   for k, v in extra.items()})
    np.savez_compressed(args.out, idx=np.asarray(idxs, np.int32),
                        attn_score=np.asarray(scores, np.float32),
                        **{k: np.asarray(v, np.float32) for k, v in extra.items()})
    arr = np.asarray(scores, np.float32)
    print(f"DONE {len(idxs)} -> {args.out} | "
          f"nan {int(np.isnan(arr).sum())} | "
          f"mean {np.nanmean(arr):.2f} std {np.nanstd(arr):.2f}", flush=True)


if __name__ == "__main__":
    main()
