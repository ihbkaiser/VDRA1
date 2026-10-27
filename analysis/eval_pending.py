#!/usr/bin/env python3
"""Evaluate every not-yet-evaluated checkpoint of one run on a set of GPUs.

Used between training chunks by scripts/20_train.sh (EVAL_EVERY_N) and by
scripts/50_eval.sh.  For each pending checkpoint (oldest first), its pending
benchmarks are split across the GPUs (largest first, onto the least-loaded
GPU) and evaluated by one eval_checkpoints.py process per GPU in parallel.
A benchmark is pending when its cached result or metadata is missing, stale, corrupt,
or tied to a different checkpoint, dataset, tokenizer, verifier, or evaluation setup.

  python analysis/eval_pending.py --run_dir runs/R2_s42 --run_id R2_s42 --gpus 0,1 \
      --dataset math500=... aime24=... --out_dir results/eval --base_model <path> [--include_step0]

Evaluation events go to run_dir/run_meta.jsonl as eval_start / eval_end, which
gpu_hours.py ignores, so evaluation never counts as training compute.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from datasets_util import load_problems  # noqa: E402
from eval_checkpoints import (  # noqa: E402
    evaluation_signature,
    limit_problems,
    refresh_cached_gpu_hours,
    result_is_current,
    result_path,
)
from vdra_core import verifier  # noqa: E402


def list_ckpts(run_dir: str) -> List[Tuple[int, str]]:
    out = subprocess.run([sys.executable, os.path.join(HERE, "list_ckpts.py"), run_dir],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        if "\t" in line:
            s, p = line.split("\t", 1)
            rows.append((int(s), p))
    return rows


def modes_for(bench: str, step: int, a) -> List[str]:
    modes = ["greedy"]
    if a.sample_k > 0 and bench in a.sample_datasets.split(",") and step >= a.sample_min_step:
        modes.append(f"avg@{a.sample_k}")
    return modes


def pending(
    run_id: str,
    step: int,
    benches: Dict[str, str],
    a,
    *,
    checkpoint_path: str | None = None,
    problems_by_bench: Dict[str, list] | None = None,
    gpu_hours_by_step: Dict[str, float] | None = None,
) -> List[str]:
    """Return benchmarks lacking complete output for this exact evaluation spec."""
    if checkpoint_path is None:
        # Filename-only caches cannot prove which checkpoint or protocol produced them.
        return list(benches)
    problem_sets = problems_by_bench or {}
    gpu_hours_by_step = gpu_hours_by_step or {}
    verifier_backend = verifier.backend_name()
    verifier_fingerprint = verifier.backend_fingerprint()
    pending_benches = []
    for bench, spec in benches.items():
        problems = problem_sets.get(bench)
        if problems is None:
            problems = limit_problems(load_problems(spec), getattr(a, "limit", None))
        if not problems:
            raise ValueError(f"benchmark {bench!r} contains no evaluation problems")
        for mode in modes_for(bench, step, a):
            if mode == "greedy":
                sample_k, temperature = 1, 0.0
            else:
                sample_k = int(mode.split("@", 1)[1])
                temperature = float(getattr(a, "temperature", 0.6))
            signature = evaluation_signature(
                run_id=run_id,
                step=step,
                checkpoint_path=checkpoint_path,
                benchmark=bench,
                dataset_spec=spec,
                problems=problems,
                mode=mode,
                max_model_len=int(getattr(a, "max_model_len", 2048)),
                tokenizer=(
                    getattr(a, "tokenizer", None)
                    or getattr(a, "base_model", None)
                    or checkpoint_path
                ),
                seed=int(getattr(a, "seed", 42)),
                temperature=temperature,
                sample_k=sample_k,
                verifier_backend=verifier_backend,
                verifier_fingerprint=verifier_fingerprint,
                gpu_hours=gpu_hours_by_step.get(str(step)),
                limit=getattr(a, "limit", None),
            )
            out_path = result_path(a.out_dir, run_id, step, bench, mode)
            current = result_is_current(out_path, signature, expected_rows=len(problems))
            if current:
                refresh_cached_gpu_hours(
                    out_path,
                    signature,
                    expected_rows=len(problems),
                    gpu_hours=gpu_hours_by_step.get(str(step)),
                )
            else:
                pending_benches.append(bench)
                break
    return pending_benches


def add_eval_limit(command: List[str], limit):
    """Pass the per-benchmark problem limit to eval_checkpoints.py."""
    if limit is None:
        return list(command)
    if limit < 1:
        raise ValueError("problem limit must be positive")
    return list(command) + ["--limit", str(limit)]


def split_over_gpus(benches: List[str], sizes: Dict[str, int], gpus: List[str]) -> Dict[str, List[str]]:
    """Longest-processing-time-first assignment of benchmarks to GPUs."""
    load = {g: 0 for g in gpus}
    plan: Dict[str, List[str]] = {g: [] for g in gpus}
    for b in sorted(benches, key=lambda x: -sizes.get(x, 1)):
        g = min(gpus, key=lambda x: (load[x], gpus.index(x)))
        plan[g].append(b)
        load[g] += sizes.get(b, 1)
    return {g: bs for g, bs in plan.items() if bs}


def wait_for_free_gpus(gpus: List[str], max_used_mb: int, timeout_s: int) -> None:
    """Training processes (vLLM, Ray workers) can take a moment to release memory."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception:
            return
        used = {l.split(",")[0].strip(): int(l.split(",")[1]) for l in out.strip().splitlines() if "," in l}
        if all(used.get(g, 0) <= max_used_mb for g in gpus):
            return
        time.sleep(10)
    print(f"[eval_pending] GPUs {gpus} still busy after {timeout_s}s; evaluating anyway", flush=True)


def meta(run_dir: str, row: Dict) -> None:
    row = dict(row, time=time.time())
    with open(os.path.join(run_dir, "run_meta.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--gpus", required=True, help="comma list of GPU ids for this run")
    ap.add_argument("--dataset", nargs="+", required=True, help="name=spec")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu_hours_json", default=None)
    ap.add_argument("--base_model", default=None, help="evaluated as step 0 with --include_step0")
    ap.add_argument("--tokenizer", default=None, help="tokenizer path for saved checkpoints")
    ap.add_argument("--include_step0", action="store_true")
    ap.add_argument("--steps", default="", help="only these steps (comma list)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N problems per benchmark")
    ap.add_argument("--sample_k", type=int, default=0)
    ap.add_argument("--sample_datasets", default="aime24,aime25,amc23")
    ap.add_argument("--sample_min_step", type=int, default=100)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evaluator", default=f"{sys.executable} {os.path.join(HERE, 'eval_checkpoints.py')}",
                    help="command prefix (tests substitute a fake)")
    ap.add_argument("--free_mb", type=int, default=3000)
    ap.add_argument("--free_timeout_s", type=int, default=300)
    a = ap.parse_args()
    if a.limit is not None and a.limit < 1:
        ap.error("problem limit must be positive")

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    benches = dict(d.split("=", 1) for d in a.dataset)
    source_problems = {b: load_problems(spec) for b, spec in benches.items()}
    sizes = {b: len(rows) for b, rows in source_problems.items()}
    problems_by_bench = {
        b: limit_problems(rows, a.limit) for b, rows in source_problems.items()
    }
    ckpts = list_ckpts(a.run_dir)
    if a.include_step0 and a.base_model:
        ckpts = [(0, a.base_model)] + [c for c in ckpts if c[0] != 0]
    want = {int(s) for s in a.steps.split(",") if s.strip()}
    if want:
        ckpts = [c for c in ckpts if c[0] in want]
    gpu_hours_by_step = (
        json.load(open(a.gpu_hours_json, encoding="utf-8"))
        if a.gpu_hours_json and os.path.exists(a.gpu_hours_json)
        else {}
    )
    todo = [
        (
            s,
            p,
            pending(
                a.run_id,
                s,
                benches,
                a,
                checkpoint_path=p,
                problems_by_bench=problems_by_bench,
                gpu_hours_by_step=gpu_hours_by_step,
            ),
        )
        for s, p in sorted(ckpts)
    ]
    todo = [t for t in todo if t[2]]
    if not todo:
        print(f"[eval_pending] {a.run_id}: nothing pending", flush=True)
        return 0
    if os.environ.get("DRY_RUN") == "1":
        for step, path, bs in todo:
            print(f"[eval_pending] (dry run) {a.run_id} step {step}: {split_over_gpus(bs, sizes, gpus)}", flush=True)
        return 0
    wait_for_free_gpus(gpus, a.free_mb, a.free_timeout_s)
    rc_all = 0
    for step, path, bs in todo:
        plan = split_over_gpus(bs, sizes, gpus)
        t0 = time.time()
        meta(a.run_dir, {"event": "eval_start", "step": step, "gpus": a.gpus, "benchmarks": bs})
        procs = []
        for g, gbs in plan.items():
            cmd = shlex.split(a.evaluator) + [
                "--run_id", a.run_id, "--ckpt_glob", path, "--step", str(step),
                "--dataset", *[f"{b}={benches[b]}" for b in gbs],
                "--out_dir", a.out_dir, "--max_model_len", str(a.max_model_len),
                "--sample_k", str(a.sample_k), "--sample_datasets", a.sample_datasets,
                "--sample_min_step", str(a.sample_min_step),
                "--temperature", str(a.temperature), "--seed", str(a.seed),
            ]
            tokenizer = a.tokenizer or a.base_model
            if tokenizer:
                cmd += ["--tokenizer", tokenizer]
            cmd = add_eval_limit(cmd, a.limit)
            if a.gpu_hours_json:
                cmd += ["--gpu_hours_json", a.gpu_hours_json]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
            procs.append((g, gbs, subprocess.Popen(cmd, env=env)))
        rcs = {g: p.wait() for g, _, p in procs}
        bad = {g: rc for g, rc in rcs.items() if rc != 0}
        rc_all |= int(bool(bad))
        meta(a.run_dir, {"event": "eval_end", "step": step, "seconds": time.time() - t0,
                         "gpu_seconds": (time.time() - t0) * len(plan), "failed_gpus": bad})
        print(f"[eval_pending] {a.run_id} step {step}: {bs} on {list(plan)} in {time.time() - t0:.0f}s"
              + (f"; failures {bad}" if bad else ""), flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
