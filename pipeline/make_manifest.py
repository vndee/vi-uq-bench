"""Judging manifests: one row per (model, question) with the greedy answer.

Reads   artifacts/generations/gen_<key>.jsonl and the dataset CSV
Writes  artifacts/judge/manifests/manifest_<key>.jsonl

Each row carries `subtype`, which selects the judge's per-task criterion line.
It equals the task type, except that information_retrieval items are split
into ir_extractive and ir_reasoning (the latter graded on the derived value)
using the CSV's subtype column. Question and
reference are checked against the CSV, so an answer is never graded against
another item's reference.

Usage:
  python pipeline/make_manifest.py
  python pipeline/make_manifest.py --only phi4,gemma3
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viuq.config import DATASET, GEN_DIR, MANIFEST_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=MANIFEST_DIR)
    ap.add_argument("--gen-dir", default=GEN_DIR)
    ap.add_argument("--csv", default=DATASET)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_rows = {r["id"]: r for r in csv.DictReader(open(args.csv, newline="",
                                                         encoding="utf-8"))}
    keep = set(args.only.split(",")) if args.only else None
    total = 0
    for p in sorted(glob.glob(os.path.join(args.gen_dir, "gen_*.jsonl"))):
        key = os.path.basename(p)[len("gen_"):-len(".jsonl")]
        if keep and key not in keep:
            continue
        rows = [json.loads(l) for l in open(p)]
        if len(rows) != len(csv_rows):
            print(f"  {key}: skipped, {len(rows)} rows, expected {len(csv_rows)}")
            continue
        out = os.path.join(args.out_dir, f"manifest_{key}.jsonl")
        subs = Counter()
        with open(out, "w") as f:
            for r in rows:
                d = csv_rows.get(r["id"])
                if d is None or d["input"] != r["question"] or d["output"] != r["reference"]:
                    raise SystemExit(f"{key} idx={r['idx']} id={r['id']}: question or "
                                     f"reference does not match the dataset CSV")
                # only information_retrieval is split into subtypes for the
                # judge; false-premise items keep the unanswerable criterion
                t = r["task_type"]
                st = (d.get("subtype") or t) if t == "information_retrieval" else t
                subs[st] += 1
                f.write(json.dumps({"set": key, "idx": r["idx"], "id": r["id"],
                                    "question": r["question"],
                                    "reference": r["reference"],
                                    "answer": r["greedy"],
                                    "task_type": r["task_type"], "subtype": st},
                                   ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"  {key}: {len(rows)} rows -> {os.path.basename(out)}")

    n = 0
    for p in sorted(glob.glob(os.path.join(args.out_dir, "manifest_*.jsonl"))):
        rows = [json.loads(l) for l in open(p)]
        n += len(rows)
        assert len({r["idx"] for r in rows}) == len(rows), f"duplicate idx in {p}"
    print(f"wrote {total} rows; {n} rows across all manifests")


if __name__ == "__main__":
    main()
