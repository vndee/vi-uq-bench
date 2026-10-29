"""Generation: one GPU pass per model, capturing everything later stages need.

Only work that needs the model weights happens here. NLI graphs, scoring,
judging, certification and probes all run later from the saved files.

Per question it stores:
  greedy text and token ids             judging
  per-token log-probs of the greedy     MSP, perplexity
  per-step predictive entropy           mean token entropy
  K sampled answers + log-probs         SE, MC-SE, DegMat, KLE, NumSemSets
  mean-pooled hidden states, all        probes (taken from the generated
    layers, and the last prompt token   sequence itself)
  token count and an end-of-turn flag   truncation, MeanLength

Two phases, because attention weights need eager attention, which is about
twice as slow:
  phase a (sdpa)   greedy answer + K samples, log-probs, entropies, hidden
                   states -> gen_<key>.jsonl, arrays_<key>.npz
  phase b (eager)  one teacher-forced forward per question for
                   AttentionScore (see attention_pass.py)

Decoding uses skip_special_tokens=True. Whether an answer ended naturally is
decided by length first (generated < budget), because chat models stop on a
turn-end token that is not tok.eos_token_id.

The run directory is stamped with the dataset fingerprint (MANIFEST.json),
and the script refuses to add to a directory built from another dataset.

Usage (settings per model are in configs/models.tsv):
  python pipeline/generate.py --model-id google/gemma-3-12b-it --key gemma3 \
    --system-prompt "..." --n-samples 10 --max-new-tokens 256 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --seed 1 --phase a
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viuq.config import DATASET, GEN_DIR  # noqa: E402
from viuq.fingerprint import assert_matches, read_manifest, write_manifest  # noqa: E402


def build_prompt_ids(tok, question, device, system_prompt=None,
                     template_kwargs=None):
    """Reconstruct the generation-time prompt (lm-polygraph tokenize())."""
    if tok.chat_template is None:
        return tok(question, return_tensors="pt").input_ids[0].to(device)
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    kwargs.update(template_kwargs or {})
    msgs = [{"role": "user", "content": question}]
    if system_prompt:
        try:
            text = tok.apply_chat_template(
                [{"role": "system", "content": system_prompt}] + msgs, **kwargs)
        except Exception:
            text = tok.apply_chat_template(
                [{"role": "user",
                  "content": system_prompt + "\n\n" + question}], **kwargs)
    else:
        text = tok.apply_chat_template(msgs, **kwargs)
    return tok(text, return_tensors="pt",
               add_special_tokens=False).input_ids[0].to(device)


def stop_ids(tok, model, extra):
    s = set(extra or [])
    if tok.eos_token_id is not None:
        s.add(tok.eos_token_id)
    cfg = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(cfg, (list, tuple)):
        s.update(x for x in cfg if x is not None)
    elif cfg is not None:
        s.add(cfg)
    return s


@torch.no_grad()
def greedy_capture(model, tok, prompt_ids, args, stops, budget=None):
    """Greedy answer + per-token logprob + per-step entropy + hidden states.

    `budget` overrides args.max_new_tokens for this item (see --task-budget)
    and also drives the truncation flag.
    """
    budget = budget or args.max_new_tokens
    out = model.generate(
        prompt_ids.unsqueeze(0), do_sample=False,
        max_new_tokens=budget,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=list(stops) or tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        output_scores=True, return_dict_in_generate=True,
        output_hidden_states=True,
    )
    gen = out.sequences[0, prompt_ids.shape[0]:]
    n_gen = int(gen.shape[0])

    lps, ents = [], []
    for t, step in enumerate(out.scores):
        logp = torch.log_softmax(step[0].float(), dim=-1)
        if t < n_gen:
            lps.append(float(logp[gen[t]]))
        p = logp.exp()
        ents.append(float(-(p * logp).sum()))

    n_layers = len(out.hidden_states[0])
    dim = out.hidden_states[0][0].shape[-1]
    hs_answer = np.zeros((n_layers, dim), dtype=np.float32)
    hs_prompt = np.zeros((n_layers, dim), dtype=np.float32)
    ans_steps = out.hidden_states[1:] or out.hidden_states
    for li in range(n_layers):
        hs_prompt[li] = out.hidden_states[0][li][0, -1, :].float().cpu().numpy()
        hs_answer[li] = np.mean([st[li][0, -1, :].float().cpu().numpy()
                                 for st in ans_steps], axis=0)

    text = tok.decode(gen, skip_special_tokens=True).strip()
    hit_eos = (n_gen < budget) or bool(set(gen.tolist()) & stops)
    return dict(text=text, ids=gen.tolist(), logprobs=lps, entropies=ents,
                n_tokens=n_gen, hit_eos=hit_eos,
                hidden_answer=hs_answer, hidden_prompt=hs_prompt)


@torch.no_grad()
def sample_capture(model, tok, prompt_ids, args, stops, budget=None):
    """K sampled answers + per-token logprobs (drives every multi-sample UE)."""
    budget = budget or args.max_new_tokens
    out = model.generate(
        prompt_ids.unsqueeze(0), do_sample=True,
        num_return_sequences=args.n_samples,
        max_new_tokens=budget,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=list(stops) or tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        output_scores=True, return_dict_in_generate=True,
    )
    plen = prompt_ids.shape[0]
    gen = out.sequences[:, plen:]
    CHUNK = 2
    parts = []
    for i in range(0, out.sequences.shape[0], CHUNK):
        sl = slice(i, i + CHUNK)
        parts.append(model.compute_transition_scores(
            out.sequences[sl], tuple(sc[sl] for sc in out.scores),
            normalize_logits=True))
    trans = torch.cat(parts, dim=0)
    del parts, out.scores
    pad = tok.pad_token_id or tok.eos_token_id
    texts, seq_lp, tok_lp, ntoks = [], [], [], []
    for i in range(gen.shape[0]):
        n_real = int((gen[i] != pad).sum())
        texts.append(tok.decode(gen[i], skip_special_tokens=True).strip())
        row = trans[i, :n_real].float().cpu().numpy()
        seq_lp.append(float(row.sum()))
        tok_lp.append(row.tolist())
        ntoks.append(n_real)
    return dict(texts=texts, seq_logprobs=seq_lp, token_logprobs=tok_lp,
                n_tokens=ntoks)


@torch.no_grad()
def attention_score(model, tok, prompt_ids, answer_ids):
    """AttentionScore, matching lm_polygraph/estimators/attention_score.py.

    Verified against the published implementation, which is:
        layer = num_hidden_layers // 2      (config default, gen_only=False)
        ue_i  = (1/H) * sum_h sum_t log( A[layer][h][t][t] + 1e-12 )
        score = -ue_i
    i.e. the SELF-attention diagonal, summed over ALL positions of the full
    prompt+answer sequence (not just generated ones, since gen_only
    defaults to False and the configs pass no arguments), averaged over
    heads, then negated.

    The sum over positions makes the score scale with sequence length;
    that is a property of the published estimator and is kept as is. The
    answer-only and length-averaged variants are returned alongside it.

    One teacher-forced forward, no generation loop, so this costs a small
    fraction of phase A. Requires eager attention.
    """
    ids = torch.cat([prompt_ids,
                     torch.tensor(answer_ids, device=prompt_ids.device,
                                  dtype=prompt_ids.dtype)])
    out = model(ids.unsqueeze(0), output_attentions=True)
    cfg = getattr(model.config, "text_config", model.config)
    layer = cfg.num_hidden_layers // 2
    A = out.attentions[layer][0].float()
    if A.shape[-1] != A.shape[-2]:
        return float("nan")
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    n_prompt = int(prompt_ids.shape[0])
    full = torch.log(diag + 1e-12).sum(dim=-1).mean()
    gen = torch.log(diag[:, n_prompt:] + 1e-12).sum(dim=-1).mean() \
        if diag.shape[-1] > n_prompt else torch.tensor(float("nan"))
    full_mean = torch.log(diag + 1e-12).mean(dim=-1).mean()
    gen_mean = torch.log(diag[:, n_prompt:] + 1e-12).mean(dim=-1).mean() \
        if diag.shape[-1] > n_prompt else torch.tensor(float("nan"))
    return {"attn_score": float(-full),
            "attn_gen_only": float(-gen),
            "attn_full_mean": float(-full_mean),
            "attn_gen_mean": float(-gen_mean)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--key", required=True, help="short name, e.g. gemma3")
    ap.add_argument("--csv", default=DATASET)
    ap.add_argument("--text-column", default="input")
    ap.add_argument("--out-dir", default=GEN_DIR)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--template-kwargs", default=None)
    ap.add_argument("--eos-ids", default=None)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--task-budget",
                    default="math_reasoning=640,summarization=384,unanswerable=768",
                    help="per-task max_new_tokens, e.g. "
                         "'math_reasoning=384,summarization=320'. "
                         "Pass '' to use --max-new-tokens everywhere.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-idx", default=None)
    ap.add_argument("--phase", choices=["a", "b", "both"], default="both")
    ap.add_argument("--no-hidden", action="store_true",
                    help="skip hidden states (saves ~0.4 GB/model; probes need them)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if read_manifest(args.out_dir) is None:
        m = write_manifest(args.out_dir, args.csv)
        print(f"[manifest] {args.out_dir} <- {m['dataset']} "
              f"rows={m['dataset_rows']} fp={m['dataset_fingerprint'][:16]}…")
    else:
        assert_matches(args.out_dir, args.csv)
        print(f"[manifest] {args.out_dir} dataset OK")

    tkw = json.loads(args.template_kwargs) if args.template_kwargs else None
    extra_eos = ([int(x) for x in args.eos_ids.split(",")]
                 if args.eos_ids else None)
    df = pd.read_csv(args.csv)
    questions = df[args.text_column].astype(str).tolist()
    refs = df["output"].astype(str).tolist() if "output" in df else [""] * len(questions)
    ids = df["id"].astype(str).tolist() if "id" in df else list(map(str, range(len(questions))))
    task_types = (df["task_type"].astype(str).tolist() if "task_type" in df
                  else [""] * len(questions))
    if args.limit:
        questions, refs, ids = questions[:args.limit], refs[:args.limit], ids[:args.limit]
        task_types = task_types[:args.limit]

    budgets = {}
    for part in (args.task_budget or "").split(","):
        if part.strip():
            t, v = part.split("=")
            budgets[t.strip()] = int(v)
    if budgets:
        if "task_type" not in df:
            raise SystemExit("--task-budget needs a task_type column in --csv")
        n_hit = sum(1 for t in task_types if t in budgets)
        print(f"per-task budgets {budgets} -> applies to {n_hit} items "
              f"(default {args.max_new_tokens})")

    jl = os.path.join(args.out_dir, f"gen_{args.key}.jsonl")
    npz = os.path.join(args.out_dir, f"arrays_{args.key}.npz")

    if args.only_idx:
        todo = ([int(l) for l in open(args.only_idx)] if os.path.exists(args.only_idx)
                else [int(x) for x in args.only_idx.split(",")])
    else:
        have = set()
        if os.path.exists(jl):
            for l in open(jl):
                try:
                    have.add(json.loads(l)["idx"])
                except Exception:
                    pass
        todo = [i for i in range(len(questions)) if i not in have]
    print(f"{len(todo)} questions to process -> {jl}", flush=True)

    if args.phase == "both":
        print("NOTE --phase both runs generation under EAGER attention "
              "(~2x slower). Prefer '--phase a' then '--phase b'.", flush=True)
    attn_impl = "eager" if args.phase in ("b", "both") else "sdpa"
    print(f"loading {args.model_id} (attn={attn_impl})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="cuda",
        attn_implementation=attn_impl, trust_remote_code=True).eval()
    stops = stop_ids(tok, model, extra_eos)
    print(f"stop ids: {sorted(stops)}", flush=True)

    acc = {k: [] for k in ("idx", "hid_ans", "hid_pr", "attn", "g_lp",
                           "g_ent", "s_lp")}
    if not args.only_idx and have:
        npz_idx = set()
        if os.path.exists(npz):
            try:
                with np.load(npz, allow_pickle=True) as z:
                    npz_idx = set(int(i) for i in z["idx"].tolist())
            except Exception as e:
                print(f"npz unreadable ({type(e).__name__}) — treating as empty",
                      flush=True)
        orphans = have - npz_idx
        if orphans:
            print(f"dropping {len(orphans)} jsonl row(s) with no npz entry so they "
                  f"are regenerated (jsonl and npz must describe the same set)",
                  flush=True)
            keep_lines = []
            for l in open(jl):
                try:
                    if json.loads(l)["idx"] in npz_idx:
                        keep_lines.append(l)
                except Exception:
                    pass
            os.replace(jl, jl + ".pre-trunc")
            with open(jl, "w") as fh:
                fh.writelines(keep_lines)
            have = npz_idx & have
            todo = [i for i in range(len(questions)) if i not in have]
            print(f"  jsonl truncated to {len(keep_lines)} rows; "
                  f"{len(todo)} to process", flush=True)

    if os.path.exists(npz) and not args.only_idx:
        try:
            with np.load(npz, allow_pickle=True) as z:
                prev = z["idx"].tolist()
                keep = [k for k, i in enumerate(prev) if i in have]
                a_attn = z["attn_score"]
                a_slp = z["sample_seq_logprobs"]
                a_glp = z["greedy_logprobs"]
                a_gent = z["greedy_entropies"]
                has_hid = "hidden_answer_mean" in z.files
                a_ha = z["hidden_answer_mean"] if has_hid else None
                a_hp = z["hidden_last_prompt"] if has_hid else None
                for k in keep:
                    acc["idx"].append(int(prev[k]))
                    acc["attn"].append(float(a_attn[k]))
                    acc["s_lp"].append(a_slp[k])
                    acc["g_lp"].append(a_glp[k])
                    acc["g_ent"].append(a_gent[k])
                    if has_hid:
                        acc["hid_ans"].append(a_ha[k])
                        acc["hid_pr"].append(a_hp[k])
            print(f"npz resume: carried {len(acc['idx'])} of {len(prev)} existing "
                  f"rows forward (jsonl has {len(have)})", flush=True)
            assert len(acc["idx"]) == len(have), (
                f"npz carried {len(acc['idx'])} rows but jsonl has {len(have)} — "
                f"the truncation step failed to reconcile them")
        except Exception as e:
            print(f"npz resume FAILED ({type(e).__name__}: {e}) — refusing to "
                  f"overwrite a partial npz", flush=True)
            raise
    with open(jl, "a") as f:
        for n, i in enumerate(todo, 1):
            torch.manual_seed(args.seed + i)
            pid = build_prompt_ids(tok, questions[i], "cuda",
                                   args.system_prompt, tkw)
            bud = budgets.get(task_types[i]) or args.max_new_tokens
            g = greedy_capture(model, tok, pid, args, stops, budget=bud)
            torch.manual_seed(args.seed + i)
            s = sample_capture(model, tok, pid, args, stops, budget=bud)
            a = (attention_score(model, tok, pid, g["ids"])["attn_score"]
                 if args.phase in ("b", "both") else float("nan"))

            f.write(json.dumps({
                "idx": i, "id": ids[i], "question": questions[i],
                "reference": refs[i], "task_type": task_types[i],
                "max_new_tokens": bud,
                "greedy": g["text"], "greedy_ids": g["ids"],
                "greedy_n_tokens": g["n_tokens"],
                "greedy_hit_eos": g["hit_eos"],
                "greedy_words": len(g["text"].split()),
                "samples": s["texts"], "sample_n_tokens": s["n_tokens"],
                "sample_seq_logprobs": s["seq_logprobs"],
            }, ensure_ascii=False) + "\n")
            f.flush()

            acc["idx"].append(i)
            acc["g_lp"].append(np.asarray(g["logprobs"], np.float32))
            acc["g_ent"].append(np.asarray(g["entropies"], np.float32))
            acc["s_lp"].append(np.asarray(s["seq_logprobs"], np.float32))
            acc["attn"].append(a)
            if not args.no_hidden:
                acc["hid_ans"].append(g["hidden_answer"].astype(np.float32))
                acc["hid_pr"].append(g["hidden_prompt"].astype(np.float32))

            if n % 25 == 0:
                print(f"{n}/{len(todo)} (idx {i})", flush=True)
                _save(npz, acc, args)
    _save(npz, acc, args)
    _sort_jsonl(jl)
    print(f"DONE {len(todo)} -> {jl} + {npz}", flush=True)


def _sort_jsonl(jl):
    """Rewrite `jl` in ascending idx order, atomically. Idempotent.

    Rows regenerated with --only-idx are appended at the end, so the file
    is sorted once the run finishes and row i always holds question i.
    """
    if not os.path.exists(jl):
        return
    rows = []
    for l in open(jl):
        try:
            rows.append((json.loads(l)["idx"], l))
        except Exception:
            pass
    if all(rows[i][0] > rows[i - 1][0] for i in range(1, len(rows))):
        return
    rows.sort(key=lambda t: t[0])
    tmp = jl + ".sorting"
    with open(tmp, "w") as f:
        f.writelines(l for _i, l in rows)
    os.replace(tmp, jl)
    print(f"sorted {os.path.basename(jl)} into ascending idx order "
          f"({len(rows)} rows)", flush=True)


def _save(npz, acc, args):
    if not acc["idx"]:
        return
    order = sorted(range(len(acc["idx"])), key=lambda k: acc["idx"][k])
    if order != list(range(len(order))):
        for k in acc:
            if acc[k]:
                acc[k] = [acc[k][i] for i in order]
    out = {"idx": np.asarray(acc["idx"], np.int32),
           "attn_score": np.asarray(acc["attn"], np.float32),
           "sample_seq_logprobs": np.stack(acc["s_lp"]),
           "greedy_logprobs": np.array(acc["g_lp"], dtype=object),
           "greedy_entropies": np.array(acc["g_ent"], dtype=object)}
    if acc["hid_ans"]:
        out["hidden_answer_mean"] = np.stack(acc["hid_ans"])
        out["hidden_last_prompt"] = np.stack(acc["hid_pr"])
    np.savez(npz, **out)


if __name__ == "__main__":
    main()
