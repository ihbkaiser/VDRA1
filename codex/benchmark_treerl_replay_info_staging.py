#!/usr/bin/env python3
"""Benchmark packing TreeRL replay-buffer info before Python scalar conversion."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


INFO_KEYS = (
    "kl",
    "reward",
    "reward_normalized",
    "response_length",
    "total_length",
    "response_overlong_ratio",
    "pass_rate",
    "response_entropy",
)


def split_legacy(info):
    batch_size = next(iter(info.values())).shape[0]
    rows = [{} for _ in range(batch_size)]
    for key, value in info.items():
        for index, item in enumerate(torch.unbind(value)):
            if key in {"pass_rate", "pass_at_1"} and item.dim() != 0:
                rows[index][key] = item[0].item()
            else:
                rows[index][key] = item.item()
    return rows


def split_packed(info):
    keys = list(info)
    batch_size = next(iter(info.values())).shape[0]
    packed = torch.stack([info[key].detach().reshape(batch_size) for key in keys], dim=0)
    host_values = packed.cpu().tolist()
    return [
        {key: host_values[key_index][row_index] for key_index, key in enumerate(keys)}
        for row_index in range(batch_size)
    ]


def make_info(batch_size: int, device: torch.device):
    torch.manual_seed(20261026 + batch_size)
    return {
        key: torch.randn(batch_size, dtype=torch.float32, device=device)
        for key in INFO_KEYS
    }


def timed(fn, info, repeats: int):
    for _ in range(2):
        fn(info)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(info)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this replay-buffer staging benchmark")

    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    results = []
    for batch_size in (8, 32, 64):
        info = make_info(batch_size, device)
        legacy_rows = split_legacy(info)
        packed_rows = split_packed(info)
        parity = legacy_rows == packed_rows
        legacy_ms = timed(split_legacy, info, repeats=11)
        packed_ms = timed(split_packed, info, repeats=11)
        legacy_median = statistics.median(legacy_ms)
        packed_median = statistics.median(packed_ms)
        results.append(
            {
                "batch_size": batch_size,
                "info_fields": len(INFO_KEYS),
                "legacy_median_ms": legacy_median,
                "packed_median_ms": packed_median,
                "speedup": legacy_median / packed_median,
                "legacy_samples_ms": legacy_ms,
                "packed_samples_ms": packed_ms,
                "exact_python_value_parity": parity,
            }
        )

    report = {
        "benchmark": "treerl_replay_buffer_info_staging",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "method": "stack same-shaped per-sample experience metrics, perform one GPU-to-CPU copy, then build replay-buffer Python rows",
        "results": results,
        "measurement": "Isolated replay-buffer metadata staging only; model work and experience generation are excluded.",
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
