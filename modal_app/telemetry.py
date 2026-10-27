"""Container-start provenance and the required fixed GPU micro-benchmark."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


def _command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _usable_cpu_label(value: str | None) -> str | None:
    if not value:
        return None
    label = value.strip()
    if label.lower() in {"unknown", "none", "n/a", "not available"}:
        return None
    return label


def _visible_cpu_count() -> int | None:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
            if count > 0:
                return count
        except (OSError, TypeError, ValueError):
            pass
    count = os.cpu_count()
    return count if count and count > 0 else None


def host_cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                model = _usable_cpu_label(line.split(":", 1)[1])
                if model:
                    return model
    except OSError:
        pass
    lscpu = _command_output(["lscpu"])
    if lscpu:
        for line in lscpu.splitlines():
            if line.lower().startswith("model name:"):
                model = _usable_cpu_label(line.split(":", 1)[1])
                if model:
                    return model

    processor = _usable_cpu_label(platform.processor())
    if processor:
        return processor

    architecture = _usable_cpu_label(platform.machine())
    count = _visible_cpu_count()
    if architecture and count:
        return f"{architecture} ({count} visible logical CPUs; model unavailable)"
    if architecture:
        return f"{architecture} (model unavailable)"
    if count:
        return f"{count} visible logical CPUs (model unavailable)"
    return "unknown (CPU model and count unavailable)"


def _validate_compute_capability(
    capability: tuple[int, int], expected: tuple[int, int] | None = (10, 0)
) -> None:
    if expected is not None and capability != expected:
        wanted = ".".join(map(str, expected))
        found = ".".join(map(str, capability))
        raise RuntimeError(f"Expected compute capability {wanted}, saw {found}")


def run_start_benchmark(
    expected_capability: tuple[int, int] | None = (10, 0),
) -> dict[str, Any]:
    """Record environment and a 60-second BF16 matmul benchmark to run_meta."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The container has no visible CUDA device")

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    capability_tuple = tuple(torch.cuda.get_device_capability(0))
    _validate_compute_capability(capability_tuple, expected=expected_capability)
    capability = list(capability_tuple)

    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    for _ in range(8):
        torch.mm(a, b)
    torch.cuda.synchronize()

    started = time.monotonic()
    deadline = started + 60.0
    iterations = 0
    while time.monotonic() < deadline:
        torch.mm(a, b)
        torch.cuda.synchronize()
        iterations += 1
    elapsed = time.monotonic() - started
    del a, b
    torch.cuda.empty_cache()

    driver = _command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {
        "event": "container_start",
        "time": time.time(),
        "host": platform.node(),
        "gpu": gpu_name,
        "gpu_compute_capability": ".".join(map(str, capability)),
        "gpu_memory_bytes": int(props.total_memory),
        "driver": driver,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "host_cpu": host_cpu(),
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "benchmark": "bf16_matmul_4096_fixed_60s",
        "benchmark_seconds": elapsed,
        "benchmark_iterations": iterations,
        "benchmark_iterations_per_second": iterations / elapsed,
    }


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
