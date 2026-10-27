"""Benchmark ragged IGPO log-probability collection strategies on CUDA."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def materialize_selected(log_chunks, length_chunks, batch_sizes, widths):
    flat_values = torch.cat(log_chunks).cpu().tolist()
    flat_lengths = torch.cat(length_chunks).cpu().tolist()
    output = []
    value_offset = 0
    length_offset = 0
    for batch_size, width in zip(batch_sizes, widths):
        for _ in range(batch_size):
            length = int(flat_lengths[length_offset])
            output.append(flat_values[value_offset : value_offset + length])
            value_offset += length
            length_offset += 1
    assert length_offset == len(flat_lengths)
    assert value_offset == len(flat_values)
    return output


def run_boolean_index(batches):
    logs, lengths, sizes, widths = [], [], [], []
    started = time.perf_counter()
    for values, seq_lengths in batches:
        valid = torch.arange(values.shape[1], device=values.device)[None, :] < seq_lengths[:, None]
        logs.append(values[valid])
        lengths.append(seq_lengths)
        sizes.append(values.shape[0])
        widths.append(values.shape[1])
    output = materialize_selected(logs, lengths, sizes, widths)
    torch.cuda.synchronize()
    return time.perf_counter() - started, output


def run_padded_gpu(batches):
    logs, lengths, sizes, widths = [], [], [], []
    started = time.perf_counter()
    max_width = max(values.shape[1] for values, _ in batches)
    for values, seq_lengths in batches:
        logs.append(F.pad(values, (0, max_width - values.shape[1])))
        lengths.append(seq_lengths)
        sizes.append(values.shape[0])
        widths.append(values.shape[1])
    flat_values = torch.cat(logs, dim=0).reshape(-1).cpu().tolist()
    flat_lengths = torch.cat(lengths).cpu().tolist()
    output, row_offset = [], 0
    for batch_size in sizes:
        for _ in range(batch_size):
            length = int(flat_lengths[len(output)])
            start = row_offset * max_width
            output.append(flat_values[start : start + length])
            row_offset += 1
    assert row_offset == len(flat_lengths)
    torch.cuda.synchronize()
    return time.perf_counter() - started, output


def run_pinned_staging(batches):
    logs, lengths, sizes, widths = [], [], [], []
    started = time.perf_counter()
    for values, seq_lengths in batches:
        host_values = torch.empty(values.shape, dtype=values.dtype, device="cpu", pin_memory=True)
        host_lengths = torch.empty(seq_lengths.shape, dtype=seq_lengths.dtype, device="cpu", pin_memory=True)
        host_values.copy_(values, non_blocking=True)
        host_lengths.copy_(seq_lengths, non_blocking=True)
        logs.append(host_values)
        lengths.append(host_lengths)
        sizes.append(values.shape[0])
        widths.append(values.shape[1])
    torch.cuda.synchronize()
    output = []
    for host_values, host_lengths in zip(logs, lengths):
        for row, length in enumerate(host_lengths.tolist()):
            output.append(host_values[row, : int(length)].tolist())
    return time.perf_counter() - started, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--batches", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--widths", default="64,128,256,512,384,96,256,128")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    widths = [int(value) for value in args.widths.split(",")]
    batches = []
    for index in range(args.batches):
        width = widths[index % len(widths)]
        values = torch.randn(args.batch_size, width, device="cuda")
        lengths = torch.randint(1, width + 1, (args.batch_size,), device="cuda")
        batches.append((values, lengths))
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()

    methods = {
        "cuda_boolean_index": run_boolean_index,
        "pad_and_concat_gpu": run_padded_gpu,
        "pinned_host_staging": run_pinned_staging,
    }
    timings, peaks, outputs = {name: [] for name in methods}, {}, {}
    for name, fn in methods.items():
        for iteration in range(args.warmup + args.repeats):
            torch.cuda.reset_peak_memory_stats()
            elapsed, output = fn(batches)
            if iteration >= args.warmup:
                timings[name].append(elapsed * 1000)
                peaks[name] = max(
                    peaks.get(name, 0),
                    torch.cuda.max_memory_allocated() - base_allocated,
                )
            outputs[name] = output
            del output
        torch.cuda.empty_cache()

    baseline = outputs["cuda_boolean_index"]
    deltas = {}
    for name, output in outputs.items():
        if len(output) != len(baseline):
            raise SystemExit(f"row count mismatch for {name}")
        deltas[name] = max(
            abs(a - b)
            for left, right in zip(baseline, output)
            for a, b in zip(left, right)
        )

    medians = {name: statistics.median(values) for name, values in timings.items()}
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "batches": args.batches,
            "batch_size": args.batch_size,
            "padded_widths_cycle": widths,
        },
        "median_ms": medians,
        "peak_extra_gpu_bytes": peaks,
        "max_abs_value_delta_vs_boolean_index": deltas,
        "measurement": "synthetic ragged [batch, sequence] CUDA log-probs; includes collection, one final synchronization/transfer, and Python-list reconstruction",
    }
    result["speedup_vs_boolean_index"] = {
        name: medians["cuda_boolean_index"] / value for name, value in medians.items()
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
