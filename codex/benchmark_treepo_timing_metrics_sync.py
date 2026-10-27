#!/usr/bin/env python3
"""Compare TreePO's old two-item timing reduction with one packed host copy."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "codex/phase_b_results/local_treepo_timing_metrics_sync_20261026.json"


def separate_item_syncs(prompt_length: torch.Tensor, response_length: torch.Tensor) -> tuple[float, float]:
    return torch.sum(prompt_length).item(), torch.sum(response_length).item()


def packed_cpu_copy(prompt_length: torch.Tensor, response_length: torch.Tensor) -> tuple[float, float]:
    values = torch.stack((torch.sum(prompt_length), torch.sum(response_length))).cpu().tolist()
    return values[0], values[1]


def timed(fn, prompt_length, response_length, updates: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(updates):
        fn(prompt_length, response_length)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this synchronization benchmark")

    batch_size = 32
    updates = 48
    repeats = 11
    generator = torch.Generator(device="cuda").manual_seed(42)
    prompt_length = torch.randint(64, 513, (batch_size,), generator=generator, device="cuda").float()
    response_length = torch.randint(64, 2049, (batch_size,), generator=generator, device="cuda").float()

    expected = separate_item_syncs(prompt_length, response_length)
    actual = packed_cpu_copy(prompt_length, response_length)
    if expected != actual:
        raise AssertionError(f"token-count mismatch: {expected!r} != {actual!r}")

    for _ in range(5):
        separate_item_syncs(prompt_length, response_length)
        packed_cpu_copy(prompt_length, response_length)

    separate_ms = []
    packed_ms = []
    for repeat in range(repeats):
        order = (
            ((separate_item_syncs, separate_ms), (packed_cpu_copy, packed_ms))
            if repeat % 2 == 0
            else ((packed_cpu_copy, packed_ms), (separate_item_syncs, separate_ms))
        )
        for fn, samples in order:
            samples.append(timed(fn, prompt_length, response_length, updates))

    separate_median = statistics.median(separate_ms)
    packed_median = statistics.median(packed_ms)
    result = {
        "benchmark": "TreePO timing metric token-count host synchronization",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shape": {"batch_size": batch_size, "prompt_length_vector": [batch_size], "response_length_vector": [batch_size]},
        "updates_per_repeat": updates,
        "repeats": repeats,
        "warmups": 5,
        "parity": "exact",
        "separate_item_syncs_ms": separate_ms,
        "packed_cpu_copy_ms": packed_ms,
        "median_separate_item_syncs_ms": separate_median,
        "median_packed_cpu_copy_ms": packed_median,
        "speedup": separate_median / packed_median if packed_median else None,
        "saved_ms_per_48_updates": separate_median - packed_median,
        "measurement_scope": "GPU reductions plus host synchronization for two batch token totals; excludes model compute and all training work.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_ms")}, indent=2))
    print(f"result={OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
