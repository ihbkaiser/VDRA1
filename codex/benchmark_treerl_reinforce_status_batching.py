#!/usr/bin/env python3
"""Benchmark TreeRL status reduction once per replay epoch vs per minibatch."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


STATUS_KEYS = (
    "policy_loss",
    "grad_norm",
    "kl",
    "reward",
    "reward_normalized",
    "response_length",
    "total_length",
    "response_overlong_ratio",
    "pass_rate",
    "response_entropy",
)


def make_status(step: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Match the active status mix: two CUDA scalars and eight CPU scalars."""
    cuda_values = torch.tensor(
        [0.25 + step * 0.001, 0.5 + step * 0.002],
        dtype=torch.float32,
        device=device,
    )
    cpu_values = torch.tensor(
        [
            0.01 + step * 0.0001,
            1.0 + step * 0.01,
            0.75 + step * 0.003,
            128.0 + (step % 16),
            256.0 + (step % 32),
            0.05 + step * 0.0002,
            0.25 + (step % 2),
            0.5 + step * 0.0003,
        ],
        dtype=torch.float32,
    )
    return {
        "policy_loss": cuda_values[0],
        "grad_norm": cuda_values[1],
        **dict(zip(STATUS_KEYS[2:], cpu_values)),
    }


def prepare_status_row(status: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Mirror ReinforceTrainer's scalar staging and KL weighting."""
    kl_value = status["kl"]
    response_length_value = status["response_length"]
    if (
        isinstance(kl_value, torch.Tensor)
        and isinstance(response_length_value, torch.Tensor)
        and kl_value.device == response_length_value.device
    ):
        status = dict(status)
        status["kl"] = kl_value.double() * response_length_value.double()
    else:
        status = dict(status)
        kl_scalar = kl_value.item() if isinstance(kl_value, torch.Tensor) else float(kl_value)
        length_scalar = (
            response_length_value.item()
            if isinstance(response_length_value, torch.Tensor)
            else float(response_length_value)
        )
        status["kl"] = float(kl_scalar) * float(length_scalar)

    cpu_values = []
    device_positions = []
    device_values = []
    for index, key in enumerate(STATUS_KEYS):
        value = status[key]
        if isinstance(value, torch.Tensor) and value.is_cuda:
            cpu_values.append(0.0)
            device_positions.append(index)
            device_values.append(value.detach().float().reshape(()))
        else:
            if isinstance(value, torch.Tensor):
                value = value.item()
            cpu_values.append(float(value))

    row = torch.tensor(cpu_values, dtype=torch.float32, device=device)
    if device_values:
        row.index_copy_(
            0,
            torch.tensor(device_positions, dtype=torch.long, device=device),
            torch.stack(device_values),
        )
    return row


def decode_status_rows(host_rows: list[list[float]]) -> list[dict[str, float]]:
    statuses = []
    for values in host_rows:
        status = dict(zip(STATUS_KEYS, values))
        status["kl"] /= status["response_length"]
        statuses.append(status)
    return statuses


def run_per_minibatch(updates: int, device: torch.device) -> tuple[float, list[dict[str, float]]]:
    statuses = []
    started = time.perf_counter()
    for step in range(updates):
        row = prepare_status_row(make_status(step, device), device)
        # The world-size-one reducer is an identity, as in the previous local
        # benchmark; the .cpu().tolist() call still forces each per-row sync.
        statuses.extend(decode_status_rows([row.cpu().tolist()]))
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0, statuses


def run_batched(
    updates: int, sync_batch_size: int, device: torch.device
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    statuses = []
    started = time.perf_counter()
    for step in range(updates):
        rows.append(prepare_status_row(make_status(step, device), device))
        if len(rows) >= sync_batch_size or step + 1 == updates:
            # The world-size-one reducer is an identity; this models one batched
            # all-reduce and one host synchronization per bounded group.
            host_rows = torch.stack(rows).cpu().tolist()
            statuses.extend(decode_status_rows(host_rows))
            rows.clear()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0, statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=48)
    parser.add_argument("--sync-batch-size", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.updates <= 0 or args.sync_batch_size <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise SystemExit("updates, sync-batch-size, and repeats must be positive; warmups may be zero")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda", torch.cuda.current_device())
    for _ in range(args.warmups):
        run_per_minibatch(args.updates, device)
        run_batched(args.updates, args.sync_batch_size, device)

    timings = {"per_minibatch": [], "batched_chunks": []}
    outputs = {}
    for _ in range(args.repeats):
        elapsed, outputs["per_minibatch"] = run_per_minibatch(args.updates, device)
        timings["per_minibatch"].append(elapsed)
        elapsed, outputs["batched_chunks"] = run_batched(
            args.updates, args.sync_batch_size, device
        )
        timings["batched_chunks"].append(elapsed)

    max_abs_delta = max(
        abs(outputs["per_minibatch"][row][key] - outputs["batched_chunks"][row][key])
        for row in range(args.updates)
        for key in STATUS_KEYS
    )
    medians = {name: statistics.median(values) for name, values in timings.items()}
    report = {
        "benchmark": "treerl_reinforce_status_batching",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "minibatches_per_epoch": args.updates,
            "metrics_per_minibatch": len(STATUS_KEYS),
            "cuda_scalar_metrics": 2,
            "cpu_tensor_metrics": 8,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "world_size": 1,
        },
        "median_ms": medians,
        "speedup_ratio_per_minibatch_over_batched_chunks": (
            medians["per_minibatch"] / medians["batched_chunks"]
        ),
        "host_materializations_per_epoch": {
            "per_minibatch": args.updates,
            "batched_chunks": (args.updates + args.sync_batch_size - 1) // args.sync_batch_size,
        },
        "max_abs_metric_delta": max_abs_delta,
        "measurement": (
            "Synthetic single-GPU active TreeRL ReinforceTrainer status path. "
            "The baseline stages and materializes each minibatch separately; the candidate "
            "stages the same rows, then stacks and materializes groups of at most sync_batch_size rows. "
            "World-size-one all_reduce is modeled as an identity, so distributed collective "
            "latency and model training are excluded."
        ),
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
