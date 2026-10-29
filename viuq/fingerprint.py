"""Dataset fingerprint, stored next to the generations it produced.

All artifacts are positional (row i of every file is question i), so a
generation directory built from one version of the dataset must never be
scored against another. The generation script writes MANIFEST.json with a
hash of the dataset and refuses to add to a directory built from a
different one.

  python -m viuq.fingerprint                 # print the dataset's fingerprint
  python -m viuq.fingerprint --check DIR     # check DIR against the dataset
"""
import csv
import hashlib
import json
import os
import time

from viuq.config import DATASET

MANIFEST = "MANIFEST.json"


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fingerprint(path):
    """sha256 over (id, input, output) of every row, in order. -> (hex, n_rows)"""
    h = hashlib.sha256()
    n = 0
    for r in _rows(path):
        if not (r.get("id") or "").strip():
            continue
        n += 1
        for f in ("id", "input", "output"):
            v = (r.get(f) or "").replace("\r\n", "\n").replace("\r", "\n")
            h.update(v.encode("utf-8"))
            h.update(b"\x1f")
        h.update(b"\x1e")
    return h.hexdigest(), n


def write_manifest(run_dir, dataset_path):
    fp, n = fingerprint(dataset_path)
    os.makedirs(run_dir, exist_ok=True)
    m = {"dataset": os.path.basename(dataset_path),
         "dataset_fingerprint": fp,
         "dataset_rows": n,
         "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(run_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return m


def read_manifest(run_dir):
    p = os.path.join(run_dir, MANIFEST)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def assert_matches(run_dir, dataset_path):
    """Exit if run_dir was produced from a different dataset."""
    m = read_manifest(run_dir)
    if m is None:
        raise SystemExit(f"{run_dir} has no {MANIFEST}; cannot tell which "
                         f"dataset produced it")
    fp, n = fingerprint(dataset_path)
    if m["dataset_fingerprint"] != fp:
        raise SystemExit(
            f"{run_dir} was produced from a different dataset\n"
            f"  artifacts: {m['dataset']} rows={m['dataset_rows']} "
            f"{m['dataset_fingerprint'][:16]}\n"
            f"  dataset  : {os.path.basename(dataset_path)} rows={n} {fp[:16]}")
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATASET)
    ap.add_argument("--check", metavar="RUN_DIR")
    args = ap.parse_args()
    fp, n = fingerprint(args.data)
    print(f"{os.path.basename(args.data)}: {n} rows  fingerprint {fp}")
    if args.check:
        m = assert_matches(args.check, args.data)
        print(f"OK: {args.check} matches (written {m['written']})")


if __name__ == "__main__":
    main()
