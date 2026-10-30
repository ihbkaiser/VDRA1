"""Deterministic allocation rules and integer-bound helpers."""

from __future__ import annotations

import heapq
import math
import random
import statistics
from typing import Iterable, Sequence


def _vectors(n: int, lower: int | Sequence[int], upper: int | None | Sequence[int]):
    lo = [int(lower)] * n if isinstance(lower, int) else [int(v) for v in lower]
    if len(lo) != n or any(v < 0 for v in lo):
        raise ValueError("lower bounds must be nonnegative and match the number of coefficients")
    if upper is None:
        hi = [None] * n
    elif isinstance(upper, int):
        hi = [int(upper)] * n
    else:
        hi = [None if v is None else int(v) for v in upper]
    if len(hi) != n:
        raise ValueError("upper bounds must match the number of coefficients")
    for i, cap in enumerate(hi):
        if cap is not None and cap < lo[i]:
            raise ValueError("upper bound is below lower bound")
    return lo, hi


def bounds_for_regime(
    regime: str,
    n: int,
    *,
    lower: int | Sequence[int] = 0,
    upper: int | Sequence[int] | None = None,
    candidates: Sequence[int] | None = None,
    default_cap: int = 6,
) -> tuple[list[int], list[int | None]]:
    lo, hi = _vectors(n, lower, upper)
    if regime == "trained":
        return lo, hi
    if regime == "capped":
        caps = [default_cap] * n
    elif regime == "capped_k":
        if candidates is None or len(candidates) != n:
            raise ValueError("capped_k requires one candidate count per node")
        caps = [min(default_cap, max(int(k), 0)) for k in candidates]
    else:
        raise ValueError(f"unknown bound regime: {regime}")
    out = [caps[i] if hi[i] is None else min(caps[i], hi[i]) for i in range(n)]
    if any(out[i] < lo[i] for i in range(n)):
        raise ValueError("bound regime is infeasible under the requested lower bounds")
    return lo, out


def _check_budget(lower: Sequence[int], upper: Sequence[int | None], budget: int) -> None:
    if budget < sum(lower):
        raise ValueError("infeasible budget: below the sum of lower bounds")
    if all(cap is not None for cap in upper) and budget > sum(int(cap) for cap in upper):
        raise ValueError("infeasible budget: above the sum of upper bounds")


def _upper_room(counts: Sequence[int], upper: Sequence[int | None], i: int) -> bool:
    return upper[i] is None or counts[i] < int(upper[i])


def _eq7_exact(coeffs: Sequence[float], lo: list[int], hi: list[int | None], budget: int) -> list[int]:
    _check_budget(lo, hi, budget)
    counts = list(lo)
    positive_without_samples = sum(
        1 for i, c in enumerate(coeffs) if float(c) > 0.0 and counts[i] == 0
    )
    if budget < sum(counts) + positive_without_samples:
        raise ValueError("infeasible budget: positive coefficients require at least one sample")

    def marginal_gain(i: int) -> float:
        c = max(float(coeffs[i]), 0.0)
        if c > 0.0 and counts[i] == 0:
            return float("inf")
        n = counts[i]
        return c / (n * (n + 1)) if n else 0.0

    heap: list[tuple[float, int]] = []
    for i, coeff in enumerate(coeffs):
        if _upper_room(counts, hi, i):
            heapq.heappush(heap, (-marginal_gain(i), i))
    while sum(counts) < budget:
        if not heap:
            raise ValueError("infeasible budget: no node has remaining upper-bound capacity")
        _, i = heapq.heappop(heap)
        counts[i] += 1
        if _upper_room(counts, hi, i):
            heapq.heappush(heap, (-marginal_gain(i), i))
    return counts


def _weighted_largest_remainder(
    weights: Sequence[float], lo: list[int], hi: list[int | None], budget: int
) -> list[int]:
    _check_budget(lo, hi, budget)
    counts = list(lo)
    remaining = budget - sum(counts)
    active = {i for i in range(len(weights)) if _upper_room(counts, hi, i)}
    while remaining and active:
        mass = sum(max(float(weights[i]), 0.0) for i in active)
        use_uniform = mass <= 0.0 or not math.isfinite(mass)
        effective = {i: (1.0 if use_uniform else max(float(weights[i]), 0.0)) for i in active}
        mass = sum(effective.values())
        quotas = {i: remaining * effective[i] / mass for i in active}
        base = {i: int(math.floor(quotas[i])) for i in active}
        capped: set[int] = set()
        placed = 0
        for i in sorted(active):
            room = budget if hi[i] is None else int(hi[i]) - counts[i]
            add = min(base[i], room)
            counts[i] += add
            placed += add
            if hi[i] is not None and counts[i] >= int(hi[i]):
                capped.add(i)
        remaining -= placed
        active -= capped
        if not remaining or not active:
            break
        if placed == 0:
            order = sorted(active, key=lambda i: (-(quotas.get(i, 0.0) % 1.0), i))
            progressed = False
            for i in order:
                if not remaining:
                    break
                if _upper_room(counts, hi, i):
                    counts[i] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        else:
            fractions = sorted(active, key=lambda i: (-(quotas.get(i, 0.0) % 1.0), i))
            for i in fractions:
                if not remaining:
                    break
                if _upper_room(counts, hi, i):
                    counts[i] += 1
                    remaining -= 1
            if not remaining:
                break
    if remaining:
        raise ValueError("infeasible budget: upper-bound capacity exhausted")
    return counts


def repair_counts(
    proposed: Sequence[int], lower: Sequence[int], upper: Sequence[int | None], budget: int
) -> tuple[list[int], list[dict[str, int]]]:
    counts = []
    for i, value in enumerate(proposed):
        count = max(int(value), int(lower[i]))
        if upper[i] is not None:
            count = min(count, int(upper[i]))
        counts.append(count)
    repairs: list[dict[str, int]] = []
    while sum(counts) > budget:
        choices = [i for i in range(len(counts)) if counts[i] > lower[i]]
        if not choices:
            raise ValueError("infeasible budget while repairing permutation")
        i = max(choices, key=lambda j: (counts[j], j))
        counts[i] -= 1
        repairs.append({"node": i, "delta": -1})
    while sum(counts) < budget:
        choices = [i for i in range(len(counts)) if _upper_room(counts, upper, i)]
        if not choices:
            raise ValueError("infeasible budget while repairing permutation")
        i = min(choices, key=lambda j: (counts[j], j))
        counts[i] += 1
        repairs.append({"node": i, "delta": 1})
    return counts, repairs


def signal_coefficients(
    signal: str,
    *,
    sigma2: Sequence[float] | None = None,
    entropy: Sequence[float] | None = None,
    nll: Sequence[float] | None = None,
    raw_tv: Sequence[float] | None = None,
    parent_counts: Sequence[int] | None = None,
    oracle_sigma2: Sequence[float] | None = None,
) -> list[float]:
    values = {
        "entropy": entropy,
        "nll": nll,
        "raw_tv": raw_tv,
        "oracle": oracle_sigma2,
    }
    if signal == "vdra_lambda":
        if sigma2 is None or parent_counts is None or len(sigma2) != len(parent_counts):
            raise ValueError("vdra_lambda requires sigma2 and parent_counts")
        return [
            max(float(c), 0.0) * max(int(parent_n) - 1, 0) / max(int(parent_n), 1)
            for c, parent_n in zip(sigma2, parent_counts)
        ]
    if signal not in values or values[signal] is None:
        raise ValueError(f"signal {signal!r} requires its corresponding coefficient vector")
    return [max(float(v), 0.0) for v in values[signal]]


def allocate_gear_as_trained(
    coeffs: Sequence[float],
    budget: int,
    *,
    node_ids: Sequence[str] | None = None,
    lambda_: float = 0.02,
    n_min: int = 0,
    distribute_remainder: bool = False,
    fallback_uniform: bool = False,
) -> list[int]:
    """Reproduce the supplied GEAR snapshot's allocator for a single queue.

    `coeffs` are stored `gear_reward_variance` (σ²), not standard deviations.
    The source routine gives `n_min` only to nodes with σ² < lambda, floors
    proportional weights for the others, and optionally distributes positive
    fractional remainders by node id.
    """
    ids = [str(x) for x in node_ids] if node_ids is not None else [f"node_{i}" for i in range(len(coeffs))]
    if len(ids) != len(coeffs) or len(set(ids)) != len(ids):
        raise ValueError("node_ids must be unique and match coefficient count")
    budget = max(int(budget), 0)
    threshold = max(float(lambda_), 0.0)
    weights: dict[str, float] = {}
    allocations: dict[str, int] = {}
    raw: dict[str, float] = {}
    for node_id, coefficient in zip(ids, coeffs):
        variance = max(float(coefficient), 0.0)
        margin = variance - threshold
        if margin < 0:
            weights[node_id] = 0.0
            allocations[node_id] = max(int(n_min), 0)
        else:
            weights[node_id] = math.sqrt(margin)
    weight_sum = sum(weights.values())
    if fallback_uniform and budget > 0 and weight_sum <= 0 and weights:
        for node_id in weights:
            weights[node_id] = 1.0
        weight_sum = float(len(weights))
    remaining = max(budget - sum(allocations.values()), 0)
    if remaining > 0 and weight_sum > 0:
        for node_id, weight in weights.items():
            if node_id in allocations:
                raw[node_id] = float(allocations[node_id])
                continue
            quota = remaining * weight / weight_sum
            raw[node_id] = quota
            allocations[node_id] = int(math.floor(quota))
    for node_id, value in allocations.items():
        raw.setdefault(node_id, float(value))
    if distribute_remainder:
        leftover = max(budget - sum(allocations.values()), 0)
        fractions = sorted(
            ((raw.get(node_id, 0.0) - math.floor(raw.get(node_id, 0.0)), node_id) for node_id in weights),
            key=lambda item: (-item[0], item[1]),
        )
        idx = 0
        while leftover > 0 and fractions:
            fraction, node_id = fractions[idx % len(fractions)]
            if fraction <= 0.0 and weight_sum > 0.0:
                break
            allocations[node_id] = allocations.get(node_id, 0) + 1
            leftover -= 1
            idx += 1
    return [allocations.get(node_id, 0) for node_id in ids]


def allocate(
    coeffs: Sequence[float],
    lower: int | Sequence[int] = 0,
    upper: int | None | Sequence[int | None] = None,
    budget: int = 0,
    rule: str = "eq7_exact",
    *,
    lambda_: float = 0.0,
    distribute_remainder: bool = False,
    fallback_uniform: bool = False,
    base_counts: Sequence[int] | None = None,
    seed: int = 0,
    n_min: int = 0,
) -> list[int]:
    """Allocate a fixed integer budget under optional per-node bounds."""
    n = len(coeffs)
    lo, hi = _vectors(n, lower, upper)
    if budget < sum(lo):
        raise ValueError("infeasible budget: below the sum of lower bounds")
    if rule in {"eq7_exact", "entropy", "nll", "raw_tv", "vdra_lambda", "oracle"}:
        return _eq7_exact(coeffs, lo, hi, int(budget))
    if rule == "uniform":
        return _weighted_largest_remainder([1.0] * n, lo, hi, int(budget))
    if rule == "fixed_branch":
        return _weighted_largest_remainder([math.sqrt(max(float(c), 0.0)) for c in coeffs], lo, hi, int(budget))
    if rule == "as_trained":
        if all(cap is None for cap in hi) and not any(lo):
            return allocate_gear_as_trained(
                coeffs, budget, lambda_=lambda_, n_min=n_min,
                distribute_remainder=distribute_remainder,
                fallback_uniform=fallback_uniform,
            )
        weights = [math.sqrt(max(float(c) - max(float(lambda_), 0.0), 0.0)) for c in coeffs]
        counts = list(lo)
        remaining = max(int(budget) - sum(counts), 0)
        active = [i for i in range(n) if _upper_room(counts, hi, i)]
        if fallback_uniform and active and sum(weights[i] for i in active) <= 0:
            weights = [1.0] * n
        weight_sum = sum(weights[i] for i in active)
        if remaining and weight_sum > 0:
            for i in active:
                counts[i] += min(int(math.floor(remaining * weights[i] / weight_sum)),
                                 int(hi[i]) - counts[i] if hi[i] is not None else remaining)
        if distribute_remainder:
            counts, _ = repair_counts(counts, lo, hi, int(budget))
        return counts
    if rule == "random_perm":
        if base_counts is None or len(base_counts) != n:
            raise ValueError("random_perm requires base_counts with one count per node")
        shuffled = list(base_counts)
        random.Random(seed).shuffle(shuffled)
        repaired, _ = repair_counts(shuffled, lo, hi, int(budget))
        return repaired
    raise ValueError(f"unknown allocation rule: {rule}")


def proportional_queue_budgets(queue_sizes: Sequence[int], budget: int) -> list[int]:
    total = sum(max(int(x), 0) for x in queue_sizes)
    if total <= 0:
        return [0] * len(queue_sizes)
    return [int(math.floor(max(int(budget), 0) * max(int(size), 0) / total)) for size in queue_sizes]


def noise_sensitivity(
    coeffs: Sequence[float],
    *,
    budget: int,
    lower: int | Sequence[int] = 1,
    upper: int | None | Sequence[int | None] = None,
    repetitions: int = 50,
    noise_levels: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    seed: int = 0,
) -> list[dict[str, float | int]]:
    """Measure allocation sensitivity to lognormal coefficient noise (A2)."""
    from .metrics import j_objective

    baseline = allocate(coeffs, lower, upper, budget, "eq7_exact")
    scale = max(sum(baseline), 1)
    rng = random.Random(seed)
    rows = []
    for sigma in noise_levels:
        if sigma < 0:
            raise ValueError("noise levels must be nonnegative")
        changed, objectives = [], []
        for _ in range(repetitions):
            noisy = [max(float(c), 0.0) * math.exp(rng.gauss(0.0, sigma)) for c in coeffs]
            counts = allocate(noisy, lower, upper, budget, "eq7_exact")
            changed.append(sum(abs(a - b) for a, b in zip(baseline, counts)) / (2 * scale))
            objectives.append(j_objective(coeffs, counts))
        rows.append({
            "noise_sigma": float(sigma),
            "mean_rollout_share_reassigned": statistics.mean(changed),
            "mean_true_objective": statistics.mean(objectives),
            "repetitions": repetitions,
        })
    return rows
