"""Measure VDRA's pure-Python queue allocator overhead on representative queue sizes."""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vdra_core.allocation import QueueItem, allocate_queue


def benchmark(queue_size: int, calls: int = 10_000, repeats: int = 5) -> dict[str, object]:
    items = [
        QueueItem(
            node_id=f"node-{index}",
            coef=0.02 + (index % 7) * 0.013,
            k_v=6,
            nominal=6,
            capped=False,
        )
        for index in range(queue_size)
    ]
    budget = 6 * queue_size

    for _ in range(200):
        allocate_queue(items, budget=budget)

    samples = []
    allocation = None
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(calls):
            allocation = allocate_queue(items, budget=budget)
        samples.append((time.perf_counter() - started) / calls * 1_000_000)

    return {
        "queue_size": queue_size,
        "budget": budget,
        "calls_per_repeat": calls,
        "repeats": repeats,
        "median_microseconds_per_queue": statistics.median(samples),
        "samples_microseconds_per_queue": samples,
        "last_allocation": allocation,
    }


def main() -> None:
    result = {
        "device": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "measurement": "allocate_queue only; no model, proxy, or framework work",
        "results": [benchmark(queue_size) for queue_size in (1, 4, 16)],
    }
    output = Path(__file__).resolve().parent / "phase_b_results" / "local_allocation_cpu_20261026.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={output.as_posix()}")


if __name__ == "__main__":
    main()
