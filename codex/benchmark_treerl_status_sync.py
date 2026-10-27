"""Benchmark TreeRL PPO's per-metric versus packed status synchronization."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def reduce_old(gpu_metrics, cpu_metrics, device):
    """Mirror the old .item() calls and one single-rank all-reduce per field."""
    status = dict(cpu_metrics)
    status.update({key: value.item() for key, value in gpu_metrics.items()})
    status["kl"] *= status["response_length"]

    reduced = {}
    for key, value in status.items():
        tensor = torch.Tensor([value]).to(device)
        # The Phase B single-rank compatibility shim makes this collective an
        # identity operation. Keep the world-size-one mean scaling here.
        tensor /= 1
        reduced[key] = tensor.cpu().item()
    reduced["kl"] /= reduced["response_length"]
    return reduced


def reduce_packed(gpu_metrics, cpu_metrics, device):
    """Mirror the packed PPO status path and its one final D2H synchronization."""
    status = dict(cpu_metrics)
    status.update(gpu_metrics)
    status["kl"] *= status["response_length"]

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

    vector = torch.tensor(cpu_values, dtype=torch.float32, device=device)
    if device_values:
        vector.index_copy_(
            0,
            torch.tensor(device_positions, dtype=torch.long, device=device),
            torch.stack(device_values),
        )
    # The same single-rank identity all-reduce as the old path.
    vector /= 1
    reduced = dict(zip(keys, vector.cpu().tolist()))
    reduced["kl"] /= reduced["response_length"]
    return reduced


def make_inputs(step, device):
    # Model the three CUDA scalar metrics (policy loss, critic loss, value mean)
    # while keeping the five experience-info metrics as Python scalars.
    metrics = torch.tensor(
        [0.25 + step * 0.001, 0.5 + step * 0.002, 0.75 + step * 0.003],
        dtype=torch.float32,
        device=device,
    )
    gpu_metrics = {
        "policy_loss": metrics[0],
        "critic_loss": metrics[1],
        "values": metrics[2],
    }
    cpu_metrics = {
        "kl": 0.01 + step * 0.0001,
        "reward": 1.0 + step * 0.01,
        "return": 2.0 + step * 0.02,
        "response_length": 128.0 + (step % 16),
        "total_length": 256.0 + (step % 32),
    }
    return gpu_metrics, cpu_metrics


def run_case(reducer, updates, device):
    outputs = []
    started = time.perf_counter()
    for step in range(updates):
        gpu_metrics, cpu_metrics = make_inputs(step, device)
        outputs.append(reducer(gpu_metrics, cpu_metrics, device))
    torch.cuda.synchronize()
    return time.perf_counter() - started, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    device = torch.device("cuda", torch.cuda.current_device())
    updates, warmups, repeats = 48, 3, 7
    torch.cuda.init()
    for _ in range(warmups):
        run_case(reduce_old, updates, device)
        run_case(reduce_packed, updates, device)

    timings = {"old_per_metric": [], "packed_vector": []}
    outputs = {}
    for _ in range(repeats):
        elapsed, outputs["old_per_metric"] = run_case(reduce_old, updates, device)
        timings["old_per_metric"].append(elapsed * 1000)
        elapsed, outputs["packed_vector"] = run_case(reduce_packed, updates, device)
        timings["packed_vector"].append(elapsed * 1000)

    old_rows = outputs["old_per_metric"]
    packed_rows = outputs["packed_vector"]
    max_abs_delta = max(
        abs(old_rows[row][key] - packed_rows[row][key])
        for row in range(updates)
        for key in old_rows[row]
    )
    medians = {key: statistics.median(value) for key, value in timings.items()}
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "updates": updates,
            "metrics_per_update": 8,
            "cuda_scalar_metrics": 3,
            "python_scalar_metrics": 5,
            "warmups": warmups,
            "repeats": repeats,
        },
        "median_ms": medians,
        "speedup_ratio_old_over_packed": medians["old_per_metric"] / medians["packed_vector"],
        "max_abs_metric_delta": max_abs_delta,
        "measurement": (
            "Synthetic single-GPU TreeRL PPO status loop. The old path materializes three CUDA scalar metrics with .item() and "
            "copies/reduces/copies each of eight fields independently. The packed path inserts three detached CUDA scalars into "
            "one FP32 vector and copies its result to CPU once. World-size-one all_reduce is an identity, matching the Phase B shim; "
            "transformer work and multi-rank collective latency are excluded."
        ),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
