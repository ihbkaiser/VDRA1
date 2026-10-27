"""Collect CPU/CUDA traces for the TreeRL PPO status-reduction microbench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

CODEX_DIR = Path(__file__).resolve().parent
ROOT = CODEX_DIR.parent
sys.path.insert(0, str(CODEX_DIR))

from benchmark_treerl_status_sync import make_inputs, reduce_old, reduce_packed  # noqa: E402


def collect(name, reducer, output_dir: Path, device: torch.device) -> dict:
    updates = 8
    # Warm up CUDA and allocator paths outside the trace window.
    for step in range(2):
        reducer(*make_inputs(step, device), device)
    torch.cuda.synchronize()

    trace_path = output_dir / f"{name}.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function(f"{name}_8_updates"):
            for step in range(updates):
                reducer(*make_inputs(step, device), device)
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))

    event_counts = {event.key: event.count for event in prof.key_averages()}
    focus_ops = (
        "aten::item",
        "aten::_local_scalar_dense",
        "aten::to",
        "aten::_to_copy",
        "aten::stack",
        "aten::index_copy_",
        "aten::copy_",
    )
    return {
        "trace": str(trace_path.relative_to(ROOT)).replace("\\", "/"),
        "updates": updates,
        "counts": {key: event_counts.get(key, 0) for key in focus_ops},
        "self_cpu_time_us": {
            event.key: round(event.self_cpu_time_total, 3)
            for event in prof.key_averages()
            if event.key in focus_ops
        },
        "self_cuda_time_us": {
            event.key: round(event.self_cuda_time_total, 3)
            for event in prof.key_averages()
            if event.key in focus_ops
        },
        "cpu_summary": prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires CUDA.")
    output_dir = ROOT / "profile_output" / "treerl_status_sync_20261026"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", torch.cuda.current_device())

    results = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "old_per_metric": collect("old_per_metric", reduce_old, output_dir, device),
        "packed_vector": collect("packed_vector", reduce_packed, output_dir, device),
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
