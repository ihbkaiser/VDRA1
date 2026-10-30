"""Allocation distribution summaries for stored reasoning trees."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .alloc_replay import allocate


def gini(values: Sequence[float]) -> float:
    xs = [max(float(v), 0.0) for v in values]
    if not xs or sum(xs) == 0:
        return 0.0
    ordered = sorted(xs)
    n = len(ordered)
    weighted = sum((i + 1) * value for i, value in enumerate(ordered))
    return (2 * weighted) / (n * sum(ordered)) - (n + 1) / n


def hhi(values: Sequence[float]) -> float:
    xs = [max(float(v), 0.0) for v in values]
    total = sum(xs)
    return sum((x / total) ** 2 for x in xs) if total else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(float(x) for x in values)
    position = (len(xs) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return xs[low]
    return xs[low] + (xs[high] - xs[low]) * (position - low)


def summarize_allocations(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    allocations = [int(row.get("allocated_branch_factor", 0) or 0) for row in records]
    requested = [row.get("requested_branch_factor") for row in records]
    children = [int(row.get("n_children", 0) or 0) for row in records]
    requested_total = sum(int(x) for x in requested if x is not None)
    allocated_total = sum(allocations)
    built_total = sum(children)
    n = len(allocations)
    return {
        "n_nodes": n,
        "n0_fraction": sum(x == 0 for x in allocations) / n if n else float("nan"),
        "n1_fraction": sum(x == 1 for x in allocations) / n if n else float("nan"),
        "over6_fraction": sum(x > 6 for x in allocations) / n if n else float("nan"),
        "n_min": min(allocations) if n else None,
        "n_p25": _quantile(allocations, 0.25),
        "n_median": _quantile(allocations, 0.50),
        "n_p75": _quantile(allocations, 0.75),
        "n_p90": _quantile(allocations, 0.90),
        "n_max": max(allocations) if n else None,
        "gini": gini(allocations),
        "hhi": hhi(allocations),
        "allocated": allocated_total,
        "built": built_total,
        "requested": requested_total if requested else None,
        "allocation_utilization": allocated_total / requested_total if requested_total else float("nan"),
        "built_utilization": built_total / requested_total if requested_total else float("nan"),
        "mean_floor_loss": mean(
            max(int(row.get("requested_branch_factor", row.get("allocated_branch_factor", 0)) or 0)
                - int(row.get("allocated_branch_factor", 0) or 0), 0)
            for row in records
        ) if records else float("nan"),
    }


def queue_counterfactual(
    records: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    alternative: str,
    upper: int | Sequence[int] | None = None,
) -> dict[str, Any]:
    coeffs = [float(row.get("reward_variance", 0.0) or 0.0) for row in records]
    trained = [int(row.get("allocated_branch_factor", 0) or 0) for row in records]
    lower = [0] * len(records)
    alternative_counts = allocate(coeffs, lower, upper, budget, alternative)
    total_rollouts = max(sum(trained), sum(alternative_counts), 1)
    changed = sum(abs(a - b) for a, b in zip(trained, alternative_counts)) / (2 * total_rollouts)
    return {
        "n_nodes": len(records),
        "budget": int(budget),
        "trained_counts": trained,
        "alternative_counts": alternative_counts,
        "rollout_share_reassigned": changed,
        "J_trained": _proxy_objective(coeffs, trained),
        "J_alternative": _proxy_objective(coeffs, alternative_counts),
    }


def _proxy_objective(coeffs: Sequence[float], counts: Sequence[int]) -> float:
    result = 0.0
    for c, n in zip(coeffs, counts):
        if n == 0 and c > 0:
            return float("inf")
        if n:
            result += c / n
    return result


def analyze_records(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group flat node rows by stage/depth and queue; no input tree is retained."""
    stage_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    queue_groups: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        iteration = int(row.get("iteration", 0) or 0)
        stage = "0-49" if iteration < 50 else "50-149" if iteration < 150 else "150-235"
        stage_groups[(stage, int(row.get("depth", 0) or 0))].append(row)
        if row.get("queue_id") is not None:
            queue_groups[(row.get("iteration"), row.get("prompt_id"), row.get("queue_id"))].append(row)
    summary_rows = []
    for (stage, depth), group in sorted(stage_groups.items()):
        summary_rows.append({"stage": stage, "depth": depth, **summarize_allocations(group)})
    queue_rows = []
    for key, group in sorted(queue_groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        allocations = [int(row.get("allocated_branch_factor", 0) or 0) for row in group]
        queue_rows.append({
            "iteration": key[0], "prompt_id": key[1], "queue_id": key[2],
            "n_nodes": len(group), "allocated": sum(allocations),
            "requested": sum(int(row.get("requested_branch_factor", 0) or 0) for row in group),
            "floor_loss": sum(max(int(row.get("requested_branch_factor", 0) or 0) - a, 0)
                               for row, a in zip(group, allocations)),
        })
    return summary_rows, queue_rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
