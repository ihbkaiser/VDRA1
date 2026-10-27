#!/usr/bin/env python3
"""Compare TreePO's current tree-advantage path with one-shot CPU staging.

The candidate is generated from the pinned source function and differs only by
copying the per-trajectory score vector to CPU once before tree-group math and
copying the final vector back before masked whitening. This avoids per-scalar
GPU/CPU conversions while preserving the existing aggregation code.
"""

from __future__ import annotations

import ast
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TREEPO = ROOT / "work" / "TreePO"
SOURCE_RELATIVE = Path("verl") / "trainer" / "ppo" / "core_algos.py"
SOURCE_COMMIT = "f05552394e734ea2838cf9e3b8191f5093319dcf"
RESULT = ROOT / "codex" / "phase_b_results" / "local_treepo_advantage_cpu_staging_20261026.json"


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-8)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    mean = masked_mean(values, mask)
    variance = masked_mean((values - mean) ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        variance = variance * (mask_sum / (mask_sum - 1))
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    mean, variance = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(variance + 1e-8)
    return whitened if shift_mean else whitened + mean


def load_functions():
    source_text = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{SOURCE_RELATIVE.as_posix()}"],
        cwd=TREEPO,
        text=True,
        encoding="utf-8",
    )
    tree = ast.parse(source_text)
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "compute_tree_reinforce_plus_plus_baseline_outcome_advantage"
    )
    legacy_text = ast.get_source_segment(source_text, node)
    if not legacy_text:
        raise RuntimeError("could not extract the pinned TreePO estimator")

    candidate_text = legacy_text
    edits = (
        (
            "    scores = token_level_rewards.sum(dim=-1)\n",
            "    scores = token_level_rewards.sum(dim=-1)\n    score_device = scores.device\n",
        ),
        (
            "    with torch.no_grad():\n",
            "    with torch.no_grad():\n        scores = scores.cpu()\n",
        ),
        (
            "        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask\n",
            "        scores = scores.to(score_device)\n"
            "        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask\n",
        ),
    )
    for old, new in edits:
        if candidate_text.count(old) != 1:
            raise RuntimeError(f"expected exactly one source anchor: {old!r}")
        candidate_text = candidate_text.replace(old, new, 1)

    globals_for_function = {
        "Dict": Dict,
        "List": List,
        "defaultdict": defaultdict,
        "np": np,
        "torch": torch,
        "verl_F": SimpleNamespace(masked_whiten=masked_whiten),
    }
    exec(compile(legacy_text, f"{SOURCE_COMMIT}:{SOURCE_RELATIVE.as_posix()}", "exec"), globals_for_function)
    legacy = globals_for_function[node.name]
    exec(compile(candidate_text, f"{SOURCE_COMMIT}:{SOURCE_RELATIVE.as_posix()}", "exec"), globals_for_function)
    candidate = globals_for_function[node.name]
    return legacy, candidate


def make_batch(batch_size: int, response_length: int, device: torch.device):
    prompts_per_group = 8
    if batch_size % prompts_per_group:
        raise ValueError("batch size must be divisible by 8")
    torch.manual_seed(20261026 + batch_size)
    token_rewards = torch.randn(batch_size, response_length, device=device, dtype=torch.float32)
    response_mask = (torch.rand(batch_size, response_length, device=device) > 0.08).to(torch.float32)
    index = []
    for row in range(batch_size):
        prompt = row // prompts_per_group
        response = row % prompts_per_group
        index.append(
            f"prompt-{prompt}#branch-{response // 4}#sub-{response // 2}#leaf-{response}"
        )
    return token_rewards, response_mask, np.asarray(index), 1e-6, False, "average"


def timed(fn, args, repeats: int) -> list[float]:
    for _ in range(2):
        fn(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this GPU synchronization benchmark")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    legacy, candidate = load_functions()
    results = []
    for batch_size in (64, 256):
        args = make_batch(batch_size, 256, device)
        old_output = legacy(*args)
        new_output = candidate(*args)
        torch.cuda.synchronize()
        max_abs_diff = max(
            float((old - new).abs().max().cpu()) for old, new in zip(old_output, new_output)
        )
        parity = all(torch.allclose(old, new, rtol=1e-6, atol=1e-7) for old, new in zip(old_output, new_output))

        old_ms = timed(legacy, args, repeats=9)
        new_ms = timed(candidate, args, repeats=9)
        old_median = statistics.median(old_ms)
        new_median = statistics.median(new_ms)
        results.append(
            {
                "batch_size": batch_size,
                "response_length": 256,
                "strategy": "average",
                "remove_no_std_subgroup": False,
                "legacy_median_ms": old_median,
                "candidate_median_ms": new_median,
                "speedup": old_median / new_median,
                "legacy_samples_ms": old_ms,
                "candidate_samples_ms": new_ms,
                "max_abs_output_delta": max_abs_diff,
                "allclose_rtol_1e-6_atol_1e-7": parity,
            }
        )

    report = {
        "benchmark": "treepo_tree_rfpp_advantage_cpu_staging",
        "source_commit": SOURCE_COMMIT,
        "source_file": f"{TREEPO.name}/{SOURCE_RELATIVE.as_posix()}",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "method": "one score-vector GPU-to-CPU copy, existing group aggregation on CPU, one CPU-to-GPU copy before masked whitening",
        "results": results,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
