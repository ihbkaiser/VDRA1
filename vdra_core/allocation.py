"""Exact bounded allocation for VDRA (Alg. 1, line 22).

    minimise   sum_v  C_v / n_v
    subject to sum_v n_v = min(B_q, sum_v u_v),   n_min <= n_v <= u_v,  n_v integer.

Because each term C/n is convex in n, the marginal reductions
Delta_v(n) = C_v/n - C_v/(n+1) = C_v / (n(n+1)) are non-increasing, and
greedily granting one rollout at a time to the node with the largest feasible
marginal reduction is exactly optimal (standard result for separable convex
resource allocation).  Ties are broken toward the node with fewer rollouts and
then the lower index, so that equal coefficients give a uniform split.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class QueueItem:
    """One expandable node waiting for an allocation."""

    node_id: str
    coef: float                 # C_v (already multiplied by lambda_v if used)
    k_v: int                    # retained pilot count after redundancy pruning
    nominal: int                # this node's nominal contribution to B_q (e.g. 6)
    capped: bool = False        # True when redundancy pruning applies (k_v < k0)


def upper_bounds(items: Sequence[QueueItem], budget: int, n_min: int) -> List[int]:
    """u_v from Alg. 1 line 21: k_v when the node is capped, otherwise the
    largest count that still leaves n_min for every other queued node."""
    q = len(items)
    uncapped = max(budget - (q - 1) * n_min, n_min)
    out = []
    for it in items:
        u = max(int(it.k_v), n_min) if it.capped else uncapped
        out.append(int(u))
    return out


def solve_bounded(
    coefs: Sequence[float],
    lower: Sequence[int],
    upper: Sequence[int],
    budget: int,
) -> List[int]:
    """Greedy marginal-gain solver; returns integer counts.

    The total allocated is min(budget, sum(upper)).  If budget < sum(lower)
    (should not happen with B_q = nominal * |Q| and n_min = 1), the lower
    bounds are granted in order of decreasing coefficient until the budget is
    exhausted, and remaining nodes get 0.
    """
    n = len(coefs)
    if n == 0:
        return []
    if len(lower) != n or len(upper) != n:
        raise ValueError("coefs, lower and upper must have equal length")
    lower = [int(x) for x in lower]
    upper = [max(int(u), int(l)) for u, l in zip(upper, lower)]
    coefs = [max(float(c), 0.0) for c in coefs]
    budget = int(budget)

    total_lower = sum(lower)
    if budget < total_lower:
        order = sorted(range(n), key=lambda i: (-coefs[i], i))
        alloc = [0] * n
        rem = budget
        for i in order:
            give = min(lower[i], rem)
            alloc[i] = give
            rem -= give
        return alloc

    alloc = list(lower)
    target = min(budget, sum(upper))
    remaining = target - total_lower

    def gain(i: int) -> float:
        c, k = coefs[i], alloc[i]
        if k <= 0:
            # From 0 to 1 rollout: infinite reduction for c>0; treat as very large.
            return float("inf") if c > 0 else 0.0
        return c / (k * (k + 1))

    heap = []
    for i in range(n):
        if alloc[i] < upper[i]:
            heapq.heappush(heap, (-gain(i), alloc[i], i))
    while remaining > 0 and heap:
        _, _, i = heapq.heappop(heap)
        alloc[i] += 1
        remaining -= 1
        if alloc[i] < upper[i]:
            heapq.heappush(heap, (-gain(i), alloc[i], i))
    return alloc


def allocate_queue(
    items: Sequence[QueueItem],
    *,
    n_min: int = 1,
    budget: Optional[int] = None,
) -> List[int]:
    """Allocate one dispatched queue.  B_q defaults to sum of nominal shares."""
    if not items:
        return []
    b = int(budget) if budget is not None else int(sum(it.nominal for it in items))
    ub = upper_bounds(items, b, n_min)
    lb = [min(n_min, u) for u in ub]
    return solve_bounded([it.coef for it in items], lb, ub, b)


def objective(coefs: Sequence[float], alloc: Sequence[int]) -> float:
    total = 0.0
    for c, k in zip(coefs, alloc):
        if k <= 0:
            if c > 0:
                return float("inf")
            continue
        total += c / k
    return total


def incoming_weight(n_parent: Optional[int]) -> float:
    """lambda = (n_p - 1)/n_p; roots (no parent) get 1."""
    if n_parent is None:
        return 1.0
    n_parent = int(n_parent)
    if n_parent <= 0:
        return 0.0
    return (n_parent - 1) / n_parent
