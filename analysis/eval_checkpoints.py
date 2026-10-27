#!/usr/bin/env python3
"""One evaluator for every run (treetune, verl, OpenRLHF checkpoints in HF format).

Greedy Pass@1 at a 2,048-token context (paper protocol) and optional avg@k
sampling at temperature 0.6, with problem-level rows so paired statistics are
possible.  Grading uses vdra_core.verifier (treetune's grader when the igpo repo
is on PYTHONPATH).

  python analysis/eval_checkpoints.py --run_id R2_s42 --ckpt_glob 'runs/R2_s42/checkpoints/*/hf_pretrained' \
      --dataset math500=$IGPO/data/math:test aime24=$IGPO/data/aime24 ... \
      --gpu_hours_json results/gpu_hours/R2_s42.json --out_dir results/eval

One file per (run, step, benchmark, mode): <out_dir>/<run>.step<S>.<bench>.<mode>.jsonl, written
atomically (temp file + rename).  A result is reused only when its sidecar matches the
checkpoint, exact dataset rows, tokenizer and verifier fingerprints, evaluation settings,
row count, and result-byte digest; legacy filename-only outputs are reevaluated.  In-training
evaluation (eval_pending.py) and post-hoc evaluation (50_eval.sh) share these checks.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets_util import load_problems  # noqa: E402
from vdra_core import verifier  # noqa: E402

TEMPLATE = (
    "<｜begin▁of▁sentence｜><｜User｜>Solve the following math problem efficiently and clearly.  "
    "The last line of your response should be of the following format: 'Therefore, the final answer "
    "is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final "
    "number or expression that solves the problem. Think step by step before answering.\n\n"
    "{query}<｜Assistant｜><think>\n"
)


EVAL_CACHE_SCHEMA = 1


def _stable_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_fingerprint(checkpoint_path: str) -> str:
    """Fingerprint the checkpoint path and file inventory without reading weights."""
    path = os.fspath(checkpoint_path)
    if not os.path.exists(path):
        identity = {"path": path, "exists": False}
    elif os.path.isfile(path):
        stat = os.stat(path)
        identity = {
            "path": os.path.realpath(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= 1024 * 1024:
            with open(path, "rb") as checkpoint_file:
                identity["sha256"] = hashlib.sha256(checkpoint_file.read()).hexdigest()
    else:
        files = []
        for root, dirs, names in os.walk(path, followlinks=True):
            dirs.sort()
            for name in sorted(names):
                full_path = os.path.join(root, name)
                try:
                    stat = os.stat(full_path)
                except OSError:
                    files.append([os.path.relpath(full_path, path), "unreadable"])
                else:
                    entry = [
                        os.path.relpath(full_path, path).replace(os.sep, "/"),
                        stat.st_size,
                        stat.st_mtime_ns,
                    ]
                    if stat.st_size <= 1024 * 1024:
                        try:
                            with open(full_path, "rb") as checkpoint_file:
                                entry.append(hashlib.sha256(checkpoint_file.read()).hexdigest())
                        except OSError:
                            entry.append("unreadable")
                    files.append(entry)
        identity = {"path": os.path.realpath(path), "files": files}
    return _stable_sha256(identity)


def tokenizer_fingerprint(tokenizer_path: str | None) -> str:
    """Fingerprint tokenizer vocabulary/config assets without hashing model weights."""
    if tokenizer_path is None:
        return _stable_sha256({"path": None})
    path = os.fspath(tokenizer_path)
    if not os.path.exists(path):
        return _stable_sha256({"path": path, "exists": False})

    token_files = {
        "added_tokens.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    }
    if os.path.isfile(path):
        candidates = [path]
        root = os.path.dirname(path)
        names = {os.path.basename(path)}
    else:
        root = path
        candidates = []
        names = set()
        for base, dirs, filenames in os.walk(path):
            dirs.sort()
            for filename in sorted(filenames):
                if filename in token_files or filename.endswith((".model", ".tiktoken")):
                    candidates.append(os.path.join(base, filename))

    assets = []
    for candidate in sorted(candidates):
        name = os.path.relpath(candidate, root).replace(os.sep, "/")
        try:
            with open(candidate, "rb") as token_file:
                digest = hashlib.sha256(token_file.read()).hexdigest()
        except OSError:
            digest = "unreadable"
        assets.append([name, digest])
    return _stable_sha256({"path": os.path.realpath(path), "assets": assets})


def evaluation_signature(
    *,
    run_id: str,
    step: int,
    checkpoint_path: str,
    benchmark: str,
    dataset_spec: str,
    problems,
    mode: str,
    max_model_len: int,
    tokenizer: str | None,
    seed: int,
    temperature: float,
    sample_k: int,
    verifier_backend: str,
    verifier_fingerprint: str,
    gpu_hours: float | None,
    limit: int | None,
) -> str:
    """Return a cache key bound to checkpoint, exact data, verifier, and eval settings."""
    dataset_rows = [
        {"id": p["id"], "problem": p["problem"], "answer": p["answer"]}
        for p in problems
    ]
    payload = {
        "schema": EVAL_CACHE_SCHEMA,
        "run_id": run_id,
        "step": int(step),
        "checkpoint": checkpoint_fingerprint(checkpoint_path),
        "benchmark": benchmark,
        "dataset_spec": dataset_spec,
        "dataset_sha256": _stable_sha256(dataset_rows),
        "problem_count": len(dataset_rows),
        "limit": limit,
        "mode": mode,
        "max_model_len": int(max_model_len),
        "tokenizer": tokenizer,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "seed": int(seed),
        "temperature": float(temperature),
        "sample_k": int(sample_k),
        "verifier_backend": verifier_backend,
        "verifier_fingerprint": verifier_fingerprint,
        "prompt_template_sha256": hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest(),
    }
    return _stable_sha256(payload)


def result_metadata_path(out_path: str) -> str:
    return f"{out_path}.meta.json"


def result_is_current(out_path: str, signature: str, *, expected_rows: int) -> bool:
    """Accept cached output only if its signature, row count, and bytes all match."""
    try:
        with open(out_path, "rb") as result_file:
            result_bytes = result_file.read()
        with open(result_metadata_path(out_path), encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("schema") != EVAL_CACHE_SCHEMA:
            return False
        if metadata.get("signature") != signature:
            return False
        if metadata.get("row_count") != expected_rows:
            return False
        if metadata.get("result_sha256") != hashlib.sha256(result_bytes).hexdigest():
            return False
        rows = result_bytes.splitlines()
        if len(rows) != expected_rows:
            return False
        return all(isinstance(json.loads(row), dict) for row in rows)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def write_result_metadata(out_path: str, signature: str, *, row_count: int) -> None:
    """Write a cache sidecar after the result file has been atomically installed."""
    with open(out_path, "rb") as result_file:
        result_bytes = result_file.read()
    metadata = {
        "schema": EVAL_CACHE_SCHEMA,
        "signature": signature,
        "row_count": row_count,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    metadata_path = result_metadata_path(out_path)
    temp_path = f"{metadata_path}.tmp{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, sort_keys=True)
        metadata_file.write("\n")
    os.replace(temp_path, metadata_path)


def refresh_cached_gpu_hours(
    out_path: str,
    signature: str,
    *,
    expected_rows: int,
    gpu_hours: float | None,
) -> bool:
    """Refresh timing annotations on a valid cached evaluation without rerunning it."""
    if not result_is_current(out_path, signature, expected_rows=expected_rows):
        return False
    with open(out_path, encoding="utf-8") as result_file:
        rows = [json.loads(line) for line in result_file if line.strip()]
    for row in rows:
        row["gpu_hours"] = gpu_hours
    temp_path = f"{out_path}.tmp{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as result_file:
        for row in rows:
            result_file.write(json.dumps(row) + "\n")
    os.replace(temp_path, out_path)
    write_result_metadata(out_path, signature, row_count=len(rows))
    return True


def result_path(out_dir: str, run_id: str, step: int, bench: str, mode: str) -> str:
    return os.path.join(out_dir, f"{run_id}.step{int(step)}.{bench}.{mode.replace('@', 'at')}.jsonl")


def step_of(path: str) -> int:
    m = re.findall(r"(?:iter|step|global_step|ckpt)[_\-]*0*(\d+)", path)
    return int(m[-1]) if m else -1


def limit_problems(problems, limit):
    """Keep at most the first N problems from each benchmark."""
    if limit is None:
        return list(problems)
    if limit < 1:
        raise ValueError("problem limit must be positive")
    return list(problems)[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--ckpt_glob", required=True)
    ap.add_argument("--dataset", nargs="+", required=True, help="name=spec")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--step", type=int, default=None, help="force the step label (e.g. 0 for the base model)")
    ap.add_argument("--steps", default="", help="comma list; empty = all checkpoints found")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N problems per benchmark")
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--sample_k", type=int, default=0, help="avg@k sampled eval (0 = off)")
    ap.add_argument("--sample_datasets", default="aime24,aime25,amc23")
    ap.add_argument("--sample_min_step", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_hours_json", default=None, help="step -> cumulative GPU-hours, from gpu_hours.py")
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    if a.limit is not None and a.limit < 1:
        ap.error("problem limit must be positive")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ckpts = sorted(glob.glob(a.ckpt_glob), key=step_of)
    wanted = {int(s) for s in a.steps.split(",") if s.strip()}
    if wanted:
        ckpts = [c for c in ckpts if step_of(c) in wanted]
    gh = json.load(open(a.gpu_hours_json)) if a.gpu_hours_json and os.path.exists(a.gpu_hours_json) else {}
    dataset_specs = dict(d.split("=", 1) for d in a.dataset)
    data = {
        name: limit_problems(load_problems(spec), a.limit)
        for name, spec in dataset_specs.items()
    }
    os.makedirs(a.out_dir, exist_ok=True)
    verifier_backend = verifier.backend_name()
    verifier_fingerprint = verifier.backend_fingerprint()
    print(f"[eval] verifier={verifier_backend} checkpoints={len(ckpts)}")
    for ck in ckpts:
        step = a.step if a.step is not None else step_of(ck)
        if step < 0:
            raise ValueError(f"cannot determine checkpoint step from {ck!r}")
        tok = None
        llm = None
        for name, probs in data.items():
            if not probs:
                raise ValueError(f"benchmark {name!r} contains no evaluation problems")
            modes = [("greedy", 1, 0.0)]
            if a.sample_k > 0 and name in a.sample_datasets.split(",") and step >= a.sample_min_step:
                modes.append((f"avg@{a.sample_k}", a.sample_k, a.temperature))
            for mode, n, temp in modes:
                out_path = result_path(a.out_dir, a.run_id, step, name, mode)
                signature = evaluation_signature(
                    run_id=a.run_id,
                    step=step,
                    checkpoint_path=ck,
                    benchmark=name,
                    dataset_spec=dataset_specs[name],
                    problems=probs,
                    mode=mode,
                    max_model_len=a.max_model_len,
                    tokenizer=a.tokenizer or ck,
                    seed=a.seed,
                    temperature=temp,
                    sample_k=n,
                    verifier_backend=verifier_backend,
                    verifier_fingerprint=verifier_fingerprint,
                    gpu_hours=gh.get(str(step)),
                    limit=a.limit,
                )
                if result_is_current(out_path, signature, expected_rows=len(probs)):
                    refresh_cached_gpu_hours(
                        out_path, signature, expected_rows=len(probs), gpu_hours=gh.get(str(step))
                    )
                    continue
                if tok is None:
                    tok = AutoTokenizer.from_pretrained(a.tokenizer or ck)
                if llm is None:
                    llm = LLM(model=ck, tokenizer=a.tokenizer or ck, max_model_len=a.max_model_len, seed=a.seed,
                              enable_prefix_caching=True, gpu_memory_utilization=0.85)
                prompts = [tok(TEMPLATE.format(query=p["problem"]), add_special_tokens=False).input_ids for p in probs]
                sp = SamplingParams(n=n, temperature=temp, top_p=1.0, seed=a.seed,
                                    max_tokens=a.max_model_len - max(len(x) for x in prompts) - 1)
                outs = llm.generate(prompts=[{"prompt_token_ids": x} for x in prompts], sampling_params=sp, use_tqdm=False)
                tmp = out_path + f".tmp{os.getpid()}"
                row_count = 0
                with open(tmp, "w") as f:
                    for p, o in zip(probs, outs):
                        scores = [verifier.score(c.text, p["answer"], p["problem"]) for c in o.outputs]
                        f.write(json.dumps({
                            "run_id": a.run_id, "step": step, "gpu_hours": gh.get(str(step)),
                            "benchmark": name, "mode": mode, "problem_id": p["id"],
                            "verifier_backend": verifier_backend,
                            "verifier_fingerprint": verifier_fingerprint,
                            "correct": float(scores[0]) if n == 1 else None,
                            "k_correct": int(sum(scores)), "k": n,
                            "truncated": any(c.finish_reason == "length" for c in o.outputs),
                            "resp_tokens": int(sum(len(c.token_ids) for c in o.outputs) / n),
                        }) + "\n")
                        row_count += 1
                if row_count != len(probs):
                    raise RuntimeError(
                        f"evaluator returned {row_count} rows for {len(probs)} problems in {name}"
                    )
                os.replace(tmp, out_path)
                write_result_metadata(out_path, signature, row_count=row_count)
                print(f"[eval] {a.run_id} step={step} {name} {mode} done")
        del llm


if __name__ == "__main__":
    main()
