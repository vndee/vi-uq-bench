"""Judge J1: OpenAI Batch API.

Re-running is safe: `build` skips every (set, idx) that already has a label in
artifacts/judge/judged_<tag>.jsonl, and `fetch` appends to that file. Failed
requests stay unlabelled and are picked up by the next `build`.

Needs OPENAI_API_KEY in the environment.

Usage:
  python pipeline/judge_openai.py build  --tag luna
  python pipeline/judge_openai.py submit --tag luna
  python pipeline/judge_openai.py status --tag luna
  python pipeline/judge_openai.py fetch  --tag luna     # once completed
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_common import (JUDGE_DIR, SCHEMA, SYSTEM, env_key, fill,  # noqa: E402
                          load_done, load_manifests)


def client():
    import openai
    return openai.OpenAI(api_key=env_key("OPENAI_API_KEY"))


def build(args):
    os.makedirs(JUDGE_DIR, exist_ok=True)
    rows = load_manifests(args.only)
    done = load_done(args.tag)
    todo = [r for r in rows if (r["set"], r["idx"]) not in done]
    path = os.path.join(JUDGE_DIR, f"batch_input_{args.tag}.jsonl")
    with open(path, "w") as f:
        for r in todo:
            body = {
                "model": args.model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": fill(r, args.language)}],
                "response_format": SCHEMA,
                "max_completion_tokens": 2048,
            }
            if args.effort:
                body["reasoning_effort"] = args.effort
            f.write(json.dumps({"custom_id": f"{r['set']}|{r['idx']}",
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": body}, ensure_ascii=False) + "\n")
    print(f"{len(todo)} requests -> {path} ({len(done)} already done)")


def submit(args):
    c = client()
    path = os.path.join(JUDGE_DIR, f"batch_input_{args.tag}.jsonl")
    up = c.files.create(file=open(path, "rb"), purpose="batch")
    b = c.batches.create(input_file_id=up.id,
                         endpoint="/v1/chat/completions",
                         completion_window="24h",
                         metadata={"tag": args.tag})
    json.dump({"batch_id": b.id},
              open(os.path.join(JUDGE_DIR, f"batch_meta_{args.tag}.json"), "w"))
    print(f"submitted batch {b.id} status={b.status}")


def status(args):
    c = client()
    meta = json.load(open(os.path.join(JUDGE_DIR, f"batch_meta_{args.tag}.json")))
    b = c.batches.retrieve(meta["batch_id"])
    print(f"{b.id}: {b.status} counts={b.request_counts}")


def fetch(args):
    c = client()
    meta = json.load(open(os.path.join(JUDGE_DIR, f"batch_meta_{args.tag}.json")))
    b = c.batches.retrieve(meta["batch_id"])
    if b.status != "completed":
        sys.exit(f"batch status={b.status}; not ready")
    outfile = os.path.join(JUDGE_DIR, f"judged_{args.tag}.jsonl")
    n_ok = n_err = 0
    with open(outfile, "a") as f:
        for fid in [b.output_file_id, b.error_file_id]:
            if not fid:
                continue
            for line in c.files.content(fid).text.splitlines():
                r = json.loads(line)
                sname, idx = r["custom_id"].rsplit("|", 1)
                rec = {"set": sname, "idx": int(idx), "model": args.tag}
                try:
                    content = r["response"]["body"]["choices"][0]["message"]["content"]
                    j = json.loads(content)
                    rec["label"] = 1.0 if j["is_correct"] else 0.0
                    rec["reasoning"] = j.get("reasoning", "")[:400]
                    n_ok += 1
                except Exception as e:
                    rec["label"] = None
                    rec["error"] = str(r.get("error") or e)[:200]
                    n_err += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"merged into {outfile}: ok={n_ok} err={n_err}"
          f"{' (build + submit again to retry)' if n_err else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "submit", "status", "fetch"])
    ap.add_argument("--tag", default="luna")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--language", default="vi", choices=["vi", "en"])
    ap.add_argument("--only", default=None,
                    help="comma-separated model keys to include")
    a = ap.parse_args()
    dict(build=build, submit=submit, status=status, fetch=fetch)[a.cmd](a)
