#!/usr/bin/env python3
"""Benchmark the active TreeRL ReinforceTrainer status materialization path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def reduce_old(metrics, device):
    """Mirror per-field .item(), GPU copy, and single-rank all-reduce behavior."""
    status = {
        key: value.item() if isinstance(value, torch.Tensor) else float(value)
        for key, value in metrics.items()
    }
    status["kl"] *= status["response_length"]
    reduced = {}
    for key, value in status.items():
        tensor = torch.Tensor([value]).to(device)
        tensor /= 1
        reduced[key] = tensor.cpu().item()
    reduced["kl"] /= reduced["response_length"]
    return reduced


def reduce_packed(metrics, device):
    """Mirror the proposed vectorized all-reduce and one D2H materialization."""
    status = dict(metrics)
    kl_value = status["kl"]
    response_length_value = status["response_length"]
    if (
        isinstance(kl_value, torch.Tensor)
        and isinstance(response_length_value, torch.Tensor)
        and kl_value.device == response_length_value.device
    ):
        status["kl"] = kl_value.double() * response_length_value.double()
    else:
        kl_scalar = kl_value.item() if isinstance(kl_value, torch.Tensor) else float(kl_value)
        length_scalar = (
            response_length_value.item()
            if isinstance(response_length_value, torch.Tensor)
            else float(response_length_value)
        )
        status["kl"] = kl_scalar * length_scalar

    keys = list(status)
    cpu_values = []
    device_positions = []
    device_values = []
    for index, key in enumerate(keys):
        value = status[key]
        if isinstance(value, torch.Tensor) and value.is_cuda:
            cpu_values.append(0.0)
            device_positions.append(index)
            device_values.append(value.detach().float().reshape(()))
        else:
            if isinstance(value, torch.Tensor):
                value = value.item()
            cpu_values.append(float(value))

    values = torch.tensor(cpu_values, dtype=torch.float32, device=device)
    if device_values:
        values.index_copy_(
            0,
            torch.tensor(device_positions, dtype=torch.long, device=device),
            torch.stack(device_values),
        )
    values /= 1
    reduced = dict(zip(keys, values.cpu().tolist()))
    reduced["kl"] /= reduced["response_length"]
    return reduced


def make_metrics(step: int, device: torch.device):
    gpu_values = torch.tensor(
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
    metrics = {"policy_loss": gpu_values[0], "grad_norm": gpu_values[1]}
    metrics.update(
        zip(
            (
                "kl",
                "reward",
                "reward_normalized",
                "response_length",
                "total_length",
                "response_overlong_ratio",
                "pass_rate",
                "response_entropy",
            ),
            cpu_values,
        )
    )
    return metrics


def run_case(reducer, updates: int, device: torch.device):
    rows = []
    started = time.perf_counter()
    for step in range(updates):
        rows.append(reducer(make_metrics(step, device), device))
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda", torch.cuda.current_device())
    updates, warmups, repeats = 48, 3, 9
    for _ in range(warmups):
        run_case(reduce_old, updates, device)
        run_case(reduce_packed, updates, device)

    timings = {"per_metric": [], "packed_vector": []}
    outputs = {}
    for _ in range(repeats):
        elapsed, outputs["per_metric"] = run_case(reduce_old, updates, device)
        timings["per_metric"].append(elapsed)
        elapsed, outputs["packed_vector"] = run_case(reduce_packed, updates, device)
        timings["packed_vector"].append(elapsed)

    old_rows = outputs["per_metric"]
    packed_rows = outputs["packed_vector"]
    max_abs_delta = max(
        abs(old_rows[row][key] - packed_rows[row][key])
        for row in range(updates)
        for key in old_rows[row]
    )
    medians = {name: statistics.median(values) for name, values in timings.items()}
    report = {
        "benchmark": "treerl_reinforce_status_sync",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "updates": updates,
            "metrics_per_update": 10,
            "cuda_scalar_metrics": 2,
            "cpu_tensor_metrics": 8,
            "warmups": warmups,
            "repeats": repeats,
        },
        "median_ms": medians,
        "speedup_ratio_old_over_packed": medians["per_metric"] / medians["packed_vector"],
        "max_abs_metric_delta": max_abs_delta,
        "measurement": (
            "Synthetic single-GPU TreeRL ReinforceTrainer status path with two CUDA loss metrics and eight CPU replay-info metrics. "
            "The legacy path materializes tensors then copies/reduces ten fields independently. The packed path reduces one FP32 vector and copies it to CPU once. "
            "World-size-one all_reduce is an identity; model work and multi-rank collective latency are excluded."
        ),
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
