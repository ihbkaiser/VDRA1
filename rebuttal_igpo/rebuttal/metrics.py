"""Metrics and prompt-clustered uncertainty for frozen-policy diagnostics."""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence


def j_objective(coeffs: Sequence[float], counts: Sequence[int]) -> float:
    if len(coeffs) != len(counts):
        raise ValueError("coefficients and counts must have the same length")
    total = 0.0
    for coefficient, count in zip(coeffs, counts):
        c, n = max(float(coefficient), 0.0), int(count)
        if n < 0:
            raise ValueError("allocation counts must be nonnegative")
        if n == 0:
            if c > 0:
                return float("inf")
            continue
        total += c / n
    return total


def parent_value_mse(estimates: Sequence[float], reference_means: Sequence[float]) -> float:
    if len(estimates) != len(reference_means) or not estimates:
        raise ValueError("estimate/reference vectors must have the same nonzero length")
    return sum((float(x) - float(y)) ** 2 for x, y in zip(estimates, reference_means)) / len(estimates)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(order)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        average = (left + 1 + right) / 2.0
        for position in range(left, right):
            ranks[order[position]] = average
        left = right
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires equal vectors of length at least two")
    x, y = _ranks(left), _ranks(right)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def split_mc_halves(samples: Sequence[Any], *, seed: int = 0) -> tuple[list[Any], list[Any]]:
    """Split an MC sample into disjoint reproducible halves."""
    if len(samples) < 2:
        raise ValueError("at least two MC samples are required for independent halves")
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    split = len(indices) // 2
    return (
        [copy.deepcopy(samples[i]) for i in indices[:split]],
        [copy.deepcopy(samples[i]) for i in indices[split:]],
    )


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str | None = None,
    value_fn: Callable[[Mapping[str, Any]], float] | None = None,
    cluster_key: str = "prompt_id",
    replicates: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Bootstrap equally weighted prompt means, retaining all rows per prompt."""
    if (value_key is None) == (value_fn is None):
        raise ValueError("provide exactly one of value_key and value_fn")
    if not rows or replicates < 1 or not 0 < alpha < 1:
        raise ValueError("rows, positive replicates and alpha in (0,1) are required")
    groups: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        if cluster_key not in row:
            raise KeyError(f"missing cluster field {cluster_key!r}")
        value = value_fn(row) if value_fn is not None else row[value_key]  # type: ignore[index]
        groups[row[cluster_key]].append(float(value))
    cluster_means = [sum(values) / len(values) for _, values in sorted(groups.items(), key=lambda kv: str(kv[0]))]
    if not cluster_means:
        raise ValueError("no clusters found")
    rng = random.Random(seed)
    estimates = []
    n = len(cluster_means)
    for _ in range(replicates):
        sample = [cluster_means[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lower_i = max(0, min(replicates - 1, int((alpha / 2) * replicates)))
    upper_i = max(0, min(replicates - 1, int((1 - alpha / 2) * replicates) - 1))
    return {
        "estimate": sum(cluster_means) / n,
        "ci_low": estimates[lower_i],
        "ci_high": estimates[upper_i],
        "n_clusters": n,
        "replicates": replicates,
    }


def allocation_summary(
    reference_variances: Sequence[float],
    counts: Sequence[int],
    uniform_counts: Sequence[int],
    oracle_counts: Sequence[int],
) -> dict[str, float]:
    actual = j_objective(reference_variances, counts)
    uniform = j_objective(reference_variances, uniform_counts)
    oracle = j_objective(reference_variances, oracle_counts)
    denominator = uniform - oracle
    return {
        "J": actual,
        "J_uniform": uniform,
        "J_oracle": oracle,
        "percent_vs_uniform": 100.0 * (actual / uniform - 1.0) if uniform else 0.0,
        "oracle_gap_recovery": (uniform - actual) / denominator if denominator > 0 else float("nan"),
        "oracle_regret": actual - oracle,
    }


def eq5_losses(
    node_errors: Sequence[float],
    edge_errors: Sequence[float],
    edge_counts: Sequence[int],
    heldout_errors: Sequence[float],
) -> dict[str, float]:
    if len(edge_errors) != len(edge_counts):
        raise ValueError("edge errors and edge counts must have equal lengths")
    if not node_errors or not heldout_errors:
        raise ValueError("node and held-out errors must be nonempty")
    count_sum = sum(int(n) for n in edge_counts)
    return {
        "L_node": sum(float(e) for e in node_errors) / len(node_errors),
        "L_edge": (
            sum(int(n) * float(e) for e, n in zip(edge_errors, edge_counts)) / count_sum
            if count_sum else float("nan")
        ),
        "L_heldout": sum(float(e) for e in heldout_errors) / len(heldout_errors),
    }


def calibration_bins(
    proxy: Sequence[float],
    target: Sequence[float],
    *,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    if len(proxy) != len(target) or not proxy:
        raise ValueError("proxy and target must have the same nonzero length")
    if bins < 1:
        raise ValueError("bins must be positive")
    pairs = sorted(zip(map(float, proxy), map(float, target)), key=lambda pair: pair[0])
    out = []
    for b in range(min(bins, len(pairs))):
        start = b * len(pairs) // min(bins, len(pairs))
        end = (b + 1) * len(pairs) // min(bins, len(pairs))
        group = pairs[start:end]
        out.append({
            "bin": b,
            "n": len(group),
            "mean_proxy": sum(x for x, _ in group) / len(group),
            "mean_target": sum(y for _, y in group) / len(group),
        })
    return out


def top_quartile_overlap(proxy: Sequence[float], target: Sequence[float]) -> float:
    if len(proxy) != len(target) or not proxy:
        raise ValueError("proxy and target must have the same nonzero length")
    n_top = max(1, math.ceil(len(proxy) / 4))
    proxy_top = set(sorted(range(len(proxy)), key=lambda i: (-proxy[i], i))[:n_top])
    target_top = set(sorted(range(len(target)), key=lambda i: (-target[i], i))[:n_top])
    return len(proxy_top & target_top) / n_top
