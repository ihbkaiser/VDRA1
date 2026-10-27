"""Benchmark IGPO's per-metric logging reductions against dtype-packed reductions.

The single-rank identity collective isolates host synchronization and Python/
tensor overhead. It does not model NCCL latency or transformer training.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


WEIGHTED = {
    "advantages/mean",
    "advantages/std",
    "rewards/mean",
    "returns",
    "non_score_rewards",
    "actor/loss",
    "actor/logit_entropy",
    "actor/approx_kl",
    "actor/policy_kl",
    "actor/clip_frac",
    "ratio",
    "critic/loss",
    "critic/value",
    "critic/mse",
    "critic/clip_frac",
}


def make_metrics(device: torch.device) -> dict[str, torch.Tensor | float]:
    names = [
        "advantages/mean",
        "advantages/std",
        "advantages/close_to_zero_perc",
        "rewards/mean",
        "num_tokens",
        "_num_participating_tokens",
        "actor/loss",
        "actor/logit_entropy",
        "actor/approx_kl",
        "actor/policy_kl",
        "actor/clip_frac",
        "critic/loss",
        "critic/value",
        "critic/mse",
        "critic/clip_frac",
        "returns",
        "non_score_rewards",
    ]
    metrics: dict[str, torch.Tensor | float] = {
        name: torch.tensor((index + 1) / 17, dtype=torch.float32, device=device)
        for index, name in enumerate(names)
    }
    metrics["num_tokens"] = torch.tensor(4096, dtype=torch.int64, device=device)
    metrics["_num_participating_tokens"] = torch.tensor(
        3584, dtype=torch.int64, device=device
    )
    metrics["kl_coef"] = 0.02
    return metrics


def reduce_old(metrics, device, world_size: int, steps: int, count_collective) -> dict[str, float]:
    if "_num_participating_tokens" in metrics:
        token_count = metrics["_num_participating_tokens"].clone()
        count_collective(token_count)
        token_count = token_count.item()
    else:
        token_count = 1

    logs = {}
    for name, value in metrics.items():
        if name.startswith("_") or value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            count_collective(value)
            value = value.item()
            divisor = world_size
        else:
            divisor = 1
        if name in WEIGHTED:
            value /= token_count
        else:
            value /= divisor * steps
        logs[name] = round(value, 8)
    return logs


def reduce_packed(metrics, device, world_size: int, steps: int, count_collective) -> dict[str, float]:
    grouped = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            grouped.setdefault(value.dtype, []).append((name, value.reshape(())))

    reduced = {}
    for dtype in sorted(grouped, key=str):
        values = grouped[dtype]
        packed = torch.stack([value for _, value in values])
        count_collective(packed)
        host_values = packed.cpu().tolist()
        reduced.update((name, value) for (name, _), value in zip(values, host_values))

    token_count = reduced.get("_num_participating_tokens", 1)
    logs = {}
    for name, value in metrics.items():
        if name.startswith("_") or value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = reduced[name]
            divisor = world_size
        else:
            divisor = 1
        if name in WEIGHTED:
            value /= token_count
        else:
            value /= divisor * steps
        logs[name] = round(value, 8)
    return logs


def time_case(reducer, metrics, device, iterations: int, counter: dict[str, int]) -> float:
    started = time.perf_counter()
    def count_collective(_tensor):
        counter["calls"] += 1

    for _ in range(iterations):
        reducer(metrics, device, world_size=1, steps=4, count_collective=count_collective)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000


def main(output: str | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device")
    device = torch.device("cuda")
    metrics = make_metrics(device)
    iterations, warmups, repeats = 48, 3, 9

    old_times = []
    packed_times = []
    old_collectives = 0
    packed_collectives = 0
    for _ in range(warmups):
        time_case(reduce_old, metrics, device, 4, {"calls": 0})
        time_case(reduce_packed, metrics, device, 4, {"calls": 0})
    for _ in range(repeats):
        old_counter = {"calls": 0}
        packed_counter = {"calls": 0}
        old_times.append(time_case(reduce_old, metrics, device, iterations, old_counter))
        packed_times.append(time_case(reduce_packed, metrics, device, iterations, packed_counter))
        old_collectives = old_counter["calls"] // iterations
        packed_collectives = packed_counter["calls"] // iterations

    old_logs = reduce_old(metrics, device, 1, 4, lambda tensor: None)
    packed_logs = reduce_packed(metrics, device, 1, 4, lambda tensor: None)
    if old_logs.keys() != packed_logs.keys():
        raise AssertionError("old and packed reducers emitted different metric keys")
    max_delta = max(abs(old_logs[key] - packed_logs[key]) for key in old_logs)

    result = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "logged_tensor_metrics": sum(isinstance(value, torch.Tensor) for value in metrics.values()),
            "dtype_groups": len({value.dtype for value in metrics.values() if isinstance(value, torch.Tensor)}),
            "synthetic_logging_intervals": iterations,
            "updates_per_interval": 4,
        },
        "collectives_per_interval": {"old": old_collectives, "packed": packed_collectives},
        "median_ms": {"old_per_metric": statistics.median(old_times), "packed_by_dtype": statistics.median(packed_times)},
        "speedup_ratio_old_over_new": statistics.median(old_times) / statistics.median(packed_times),
        "max_abs_metric_delta": max_delta,
        "measurement": "Synthetic IGPO metric logging with single-rank identity collectives; captures tensor synchronization and Python overhead, not NCCL latency or transformer compute.",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    main(parser.parse_args().output)
