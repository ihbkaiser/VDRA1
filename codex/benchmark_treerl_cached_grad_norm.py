#!/usr/bin/env python3
"""Measure the redundant per-minibatch TreeRL gradient-norm scan."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def manual_grad_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    parts = [grad.norm(2).to(dtype=torch.float64) for grad in grads]
    return torch.linalg.vector_norm(torch.stack(parts)).detach()


def run_manual(grads: list[torch.Tensor], minibatches: int, device: torch.device) -> tuple[float, list[float]]:
    values = []
    started = time.perf_counter()
    for _ in range(minibatches):
        values.append(manual_grad_norm(grads))
    host_values = torch.stack(values).cpu().tolist()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0, host_values


def run_cached(
    cached_norm: torch.Tensor,
    minibatches: int,
    accumulation_steps: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    values = []
    started = time.perf_counter()
    for step in range(minibatches):
        if (step + 1) % accumulation_steps == 0:
            # The optimizer already calculated this value at the update boundary.
            values.append(cached_norm)
    host_values = torch.stack(values).cpu().tolist()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0, host_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parameter-tensors", type=int, default=160)
    parser.add_argument("--elements-per-tensor", type=int, default=8192)
    parser.add_argument("--minibatches", type=int, default=48)
    parser.add_argument("--accumulation-steps", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.parameter_tensors, args.elements_per_tensor, args.minibatches, args.accumulation_steps, args.repeats) <= 0:
        raise SystemExit("tensor dimensions, minibatches, accumulation, and repeats must be positive")
    if args.minibatches % args.accumulation_steps:
        raise SystemExit("minibatches must be divisible by accumulation_steps")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda", torch.cuda.current_device())
    grads = [
        torch.randn(args.elements_per_tensor, dtype=torch.float32, device=device)
        for _ in range(args.parameter_tensors)
    ]
    cached_norm = manual_grad_norm(grads)
    torch.cuda.synchronize(device)

    for _ in range(args.warmups):
        run_manual(grads, args.minibatches, device)
        run_cached(cached_norm, args.minibatches, args.accumulation_steps, device)

    timings = {"per_minibatch_full_scan": [], "cached_at_update_boundary": []}
    manual_values = []
    cached_values = []
    for _ in range(args.repeats):
        elapsed, manual_values = run_manual(grads, args.minibatches, device)
        timings["per_minibatch_full_scan"].append(elapsed)
        elapsed, cached_values = run_cached(
            cached_norm, args.minibatches, args.accumulation_steps, device
        )
        timings["cached_at_update_boundary"].append(elapsed)

    medians = {name: statistics.median(values) for name, values in timings.items()}
    report = {
        "benchmark": "treerl_cached_grad_norm",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "parameter_tensors": args.parameter_tensors,
            "elements_per_tensor": args.elements_per_tensor,
            "minibatches_per_update": args.minibatches,
            "optimizer_accumulation_steps": args.accumulation_steps,
            "optimizer_updates": args.minibatches // args.accumulation_steps,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "median_ms": medians,
        "speedup_ratio_full_scan_over_cached": (
            medians["per_minibatch_full_scan"] / medians["cached_at_update_boundary"]
            if medians["cached_at_update_boundary"] > 0
            else None
        ),
        "full_norm_scans_per_update": {
            "per_minibatch_full_scan": args.minibatches,
            "cached_at_update_boundary": 0,
        },
        "reported_update_norms": len(cached_values),
        "cached_norm_reference": (
            "The same synthetic gradient norm was computed once before timing to model "
            "the value published by the optimizer; this is not a DeepSpeed parity test."
        ),
        "measurement": (
            "Local single-GPU microbenchmark of the redundant Trainer-side norm scan. "
            "The cached path reads the norm already computed inside DeepSpeed's optimizer step; "
            "that optimizer work, distributed safe_get_full_grad all-gathers, model training, "
            "and end-to-end B200 time are excluded. Only optimizer-boundary norms are reported."
        ),
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
