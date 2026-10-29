"""Pieces shared by the two judge clients (OpenAI batch, Gemini batch)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubric import build_prompt  # noqa: E402
from viuq.config import JUDGE_DIR, MANIFEST_DIR  # noqa: E402

SYSTEM = "You are an expert evaluator."

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "CorrectnessJudgment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"is_correct": {"type": "boolean"},
                           "reasoning": {"type": "string"}},
            "required": ["is_correct", "reasoning"],
            "additionalProperties": False,
        },
    },
}


def prompt_for(language, task=None):
    """Rubric for one item, with its task criterion line if it has one."""
    return build_prompt(task, language)


def fill(row, language):
    """The user message for one manifest row. The subtype decides the
    criterion line, falling back to the task type."""
    return prompt_for(language, row.get("subtype") or row.get("task_type")).format(
        input=row["question"], target=row["reference"], generation=row["answer"])


def env_key(name):
    k = os.environ.get(name, "").strip()
    if not k:
        sys.exit(f"set {name} in the environment")
    return k


def load_manifests(only=None, manifest_dir=MANIFEST_DIR):
    found = sorted(glob.glob(os.path.join(manifest_dir, "*.jsonl")))
    if not found:
        raise SystemExit(f"no manifests in {manifest_dir}; run make_manifest.py")
    rows = []
    for f in found:
        rows += [json.loads(l) for l in open(f)]
    if only:
        keep = set(only.split(","))
        rows = [r for r in rows if r["set"] in keep]
    return rows


def load_done(tag):
    """(set, idx) pairs that already have a label under this tag."""
    done = set()
    f = os.path.join(JUDGE_DIR, f"judged_{tag}.jsonl")
    if os.path.exists(f):
        for l in open(f):
            try:
                r = json.loads(l)
                if r.get("label") is not None:
                    done.add((r["set"], r["idx"]))
            except json.JSONDecodeError:
                pass
    return done
