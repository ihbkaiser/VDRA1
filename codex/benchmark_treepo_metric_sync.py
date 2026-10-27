"""Microbenchmark TreePO's old per-microbatch metric sync vs packed transfer."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch


def make_templates(
    microbatches: int, updates: int, include_kl: bool
) -> list[dict[str, list[torch.Tensor]]]:
    keys = ["actor/pg_loss", "actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower"]
    if include_kl:
        keys.insert(0, "actor/kl_loss")
    keys.append("actor/grad_norm")
    values = torch.arange(updates * microbatches * len(keys), device="cuda", dtype=torch.float32).unbind()
    cursor = 0
    batches = []
    for _ in range(updates):
        metrics = {}
        for key in keys:
            metrics[key] = list(values[cursor : cursor + microbatches])
            cursor += microbatches
        batches.append(metrics)
    return batches


def old_materialize(metrics: dict[str, list[torch.Tensor]]) -> dict[str, list[float]]:
    return {key: [value.item() for value in values] for key, values in metrics.items()}


def packed_cast_each_materialize(
    metrics: dict[str, list[torch.Tensor]],
) -> dict[str, list[float]]:
    tensor_values = []
    slots = []
    for key, values in metrics.items():
        for index, value in enumerate(values):
            if torch.is_tensor(value):
                slots.append((key, index))
                tensor_values.append(value.detach().double().reshape(()))

    result = {key: list(values) for key, values in metrics.items()}
    if tensor_values:
        materialized = torch.stack(tensor_values).cpu().tolist()
        for (key, index), value in zip(slots, materialized):
            result[key][index] = value
    return result


def packed_stack_first_materialize(
    metrics: dict[str, list[torch.Tensor]],
) -> dict[str, list[float]]:
    tensor_values = []
    slots = []
    for key, values in metrics.items():
        for index, value in enumerate(values):
            if torch.is_tensor(value):
                slots.append((key, index))
                tensor_values.append(value.detach().reshape(()))

    result = {key: list(values) for key, values in metrics.items()}
    if tensor_values:
        # Stack native dtypes first; torch.stack promotes mixed floating dtypes,
        # avoiding one device cast kernel per scalar before the single D2H copy.
        materialized = torch.stack(tensor_values).cpu().tolist()
        for (key, index), value in zip(slots, materialized):
            result[key][index] = value
    return result


def measure(fn, templates: list[dict[str, list[torch.Tensor]]], repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = [fn(metrics) for metrics in templates]
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
        if not result:
            raise RuntimeError("empty benchmark result")
    return samples


def benchmark(microbatches: int, updates: int, repeats: int, include_kl: bool) -> dict[str, object]:
    templates = make_templates(microbatches, updates, include_kl)
    for metrics in templates:
        expected = old_materialize(metrics)
        cast_each = packed_cast_each_materialize(metrics)
        stack_first = packed_stack_first_materialize(metrics)
        if expected != cast_each or expected != stack_first:
            raise RuntimeError("metric materialization paths returned different values")

    measure(old_materialize, templates, repeats=3)
    measure(packed_cast_each_materialize, templates, repeats=3)
    measure(packed_stack_first_materialize, templates, repeats=3)
    old_ms = measure(old_materialize, templates, repeats)
    packed_cast_each_ms = measure(packed_cast_each_materialize, templates, repeats)
    packed_stack_first_ms = measure(packed_stack_first_materialize, templates, repeats)
    old_median = statistics.median(old_ms)
    packed_cast_each_median = statistics.median(packed_cast_each_ms)
    packed_stack_first_median = statistics.median(packed_stack_first_ms)
    return {
        "microbatches_per_update": microbatches,
        "updates_per_sample": updates,
        "repeats": repeats,
        "kl_enabled": include_kl,
        "scalar_syncs_old_per_sample": updates * microbatches * (5 if include_kl else 4) + updates * microbatches,
        "scalar_syncs_packed_per_sample": updates,
        "old_median_ms": old_median,
        "packed_cast_each_median_ms": packed_cast_each_median,
        "packed_cast_each_speedup": old_median / packed_cast_each_median,
        "packed_stack_first_median_ms": packed_stack_first_median,
        "packed_stack_first_speedup": old_median / packed_stack_first_median,
        "old_samples_ms": old_ms,
        "packed_cast_each_samples_ms": packed_cast_each_ms,
        "packed_stack_first_samples_ms": packed_stack_first_ms,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    torch.cuda.init()
    rows = [
        benchmark(microbatches, updates=48, repeats=20, include_kl=include_kl)
        for microbatches in (8, 32)
        for include_kl in (False, True)
    ]
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "metric_path": "treepo actor metric synchronization only; no model compute",
        "results": rows,
    }
    output = Path("codex/phase_b_results/local_treepo_metric_sync_20261026.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={output.as_posix()}")


if __name__ == "__main__":
    main()
