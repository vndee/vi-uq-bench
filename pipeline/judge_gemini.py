"""Judge J2: Gemini Batch Mode.

Same rubric as judge_openai.py, structured output through responseSchema.
Re-running is safe in the same way: `build` skips items already labelled in
artifacts/judge/judged_<tag>.jsonl and `fetch` appends to it.

Needs GEMINI_API_KEY in the environment. The key is sent as a header, never
in the URL, so it cannot end up in error messages or logs.

Usage (one batch per model keeps each under the per-batch quota):
  python pipeline/judge_gemini.py build  --tag gemini-phi4 --only phi4
  python pipeline/judge_gemini.py submit --tag gemini-phi4
  python pipeline/judge_gemini.py status --tag gemini-phi4
  python pipeline/judge_gemini.py fetch  --tag gemini-phi4
"""
import argparse
import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_common import (JUDGE_DIR, SYSTEM, env_key, fill, load_done,  # noqa: E402
                          load_manifests)

BASE = "https://generativelanguage.googleapis.com"
SCHEMA = {"type": "object",
          "properties": {"is_correct": {"type": "boolean"},
                         "reasoning": {"type": "string"}},
          "required": ["is_correct", "reasoning"]}


def _auth(headers=None):
    h = dict(headers or {})
    h["x-goog-api-key"] = env_key("GEMINI_API_KEY")
    return h


def _check(r):
    if not r.ok:
        raise SystemExit(f"HTTP {r.status_code}: {r.text[:400]}")
    return r


def build(args):
    os.makedirs(JUDGE_DIR, exist_ok=True)
    rows = load_manifests(args.only)
    done = load_done(args.tag)
    todo = [r for r in rows if (r["set"], r["idx"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    path = os.path.join(JUDGE_DIR, f"gbatch_input_{args.tag}.jsonl")
    with open(path, "w") as f:
        for r in todo:
            req = {
                "contents": [{"role": "user",
                              "parts": [{"text": fill(r, args.language)}]}],
                "systemInstruction": {"parts": [{"text": SYSTEM}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": SCHEMA,
                    # gemini-3.x rejects a budget of 0; 128 is the smallest
                    # accepted value and keeps thinking tokens low
                    "thinkingConfig": {"thinkingBudget": args.thinking_budget},
                },
            }
            f.write(json.dumps({"key": f"{r['set']}__{r['idx']}",
                                "request": req}, ensure_ascii=False) + "\n")
    # the model is fixed at build time; submit reads it from here
    json.dump({"model": args.model, "n": len(todo)},
              open(os.path.join(JUDGE_DIR, f"gbatch_build_{args.tag}.json"), "w"))
    print(f"{len(todo)} requests -> {path} ({len(done)} already done), "
          f"model={args.model}")


def _built_model(tag):
    p = os.path.join(JUDGE_DIR, f"gbatch_build_{tag}.json")
    if not os.path.exists(p):
        raise SystemExit(f"no build metadata for tag {tag}; run `build` first")
    return json.load(open(p))["model"]


def submit(args):
    path = os.path.join(JUDGE_DIR, f"gbatch_input_{args.tag}.jsonl")
    size = os.path.getsize(path)
    start = _check(requests.post(
        f"{BASE}/upload/v1beta/files",
        headers=_auth({"X-Goog-Upload-Protocol": "resumable",
                       "X-Goog-Upload-Command": "start",
                       "X-Goog-Upload-Header-Content-Length": str(size),
                       "X-Goog-Upload-Header-Content-Type": "application/jsonl",
                       "Content-Type": "application/json"}),
        json={"file": {"display_name": f"judge_{args.tag}"}}, timeout=120))
    up_url = start.headers["X-Goog-Upload-URL"]
    upl = _check(requests.post(
        up_url, headers={"X-Goog-Upload-Command": "upload, finalize",
                         "X-Goog-Upload-Offset": "0",
                         "Content-Length": str(size)},
        data=open(path, "rb"), timeout=600))
    fname = upl.json()["file"]["name"]
    b = _check(requests.post(
        f"{BASE}/v1beta/models/{_built_model(args.tag)}:batchGenerateContent",
        headers=_auth(),
        json={"batch": {"displayName": f"judge_{args.tag}",
                        "inputConfig": {"fileName": fname}}}, timeout=120)).json()
    json.dump({"op": b["name"]},
              open(os.path.join(JUDGE_DIR, f"gbatch_meta_{args.tag}.json"), "w"))
    print(f"submitted {b['name']}")


def _get(tag):
    meta = json.load(open(os.path.join(JUDGE_DIR, f"gbatch_meta_{tag}.json")))
    return _check(requests.get(f"{BASE}/v1beta/{meta['op']}",
                               headers=_auth(), timeout=60)).json()


def status(args):
    b = _get(args.tag)
    print(f"{b['name']}: state={b.get('metadata', {}).get('state')} "
          f"done={b.get('done')}")


def fetch(args):
    b = _get(args.tag)
    if not b.get("done"):
        sys.exit(f"not done: state={b.get('metadata', {}).get('state')}")
    if "error" in b:
        sys.exit(f"batch error: {b['error']}")
    resp = b["response"]
    fname = (resp.get("responsesFile")
             or resp.get("output", {}).get("responsesFile"))
    inline = (resp.get("inlinedResponses") or {}).get("inlinedResponses")
    lines = []
    if fname:
        dl = _check(requests.get(f"{BASE}/download/v1beta/{fname}:download",
                                 params={"alt": "media"}, headers=_auth(),
                                 timeout=600))
        lines = dl.text.splitlines()
    outfile = os.path.join(JUDGE_DIR, f"judged_{args.tag}.jsonl")
    n_ok = n_err = 0
    with open(outfile, "a") as f:
        for entry in (lines or inline or []):
            r = json.loads(entry) if isinstance(entry, str) else entry
            k = r.get("key") or r.get("metadata", {}).get("key")
            sname, idx = k.rsplit("__", 1)
            rec = {"set": sname, "idx": int(idx), "model": args.tag}
            try:
                txt = r["response"]["candidates"][0]["content"]["parts"][0]["text"]
                j = json.loads(txt)
                rec["label"] = 1.0 if j["is_correct"] else 0.0
                rec["reasoning"] = str(j.get("reasoning", ""))[:400]
                n_ok += 1
            except Exception as e:
                rec["label"] = None
                rec["error"] = str(r.get("error") or e)[:200]
                n_err += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"merged into {outfile}: ok={n_ok} err={n_err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "submit", "status", "fetch"])
    ap.add_argument("--tag", default="gemini")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--language", default="vi", choices=["vi", "en"])
    ap.add_argument("--thinking-budget", type=int, default=128)
    ap.add_argument("--only", default=None,
                    help="comma-separated model keys to include")
    a = ap.parse_args()
    dict(build=build, submit=submit, status=status, fetch=fetch)[a.cmd](a)
