"""Shared helpers for Exp 1 / Exp 2: oracle quantities and queue allocation."""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdra_core import QueueItem, allocate_queue  # noqa: E402


def child_mean(child: Dict, half: Optional[str] = None) -> float:
    """Mean reward of a pool child: all refs, the draw half, or the reference half."""
    if "p_true" in child and half == "true":
        return float(child["p_true"])
    refs = child["refs"]
    if half == "draw":
        refs = refs[: len(refs) // 2]
    elif half == "ref":
        refs = refs[len(refs) // 2:]
    return float(np.mean(refs)) if refs else 0.0


def child_noise_var(child: Dict, half: str = "all") -> float:
    """Unbiased estimate of Var(mean of the refs used) for a Bernoulli child."""
    if child.get("finished"):
        return 0.0
    refs = child["refs"]
    if half == "ref":
        refs = refs[len(refs) // 2:]
    elif half == "draw":
        refs = refs[: len(refs) // 2]
    n = len(refs)
    if n < 2:
        return 0.0
    p = float(np.mean(refs))
    return p * (1 - p) / (n - 1)


def oracle_dispersion(pool: Sequence[Dict], half: str = "all") -> float:
    """sigma^2_MC: between-child variance of child values, corrected for the
    finite number of reference completions per child (clipped at 0)."""
    if any("p_true" in c for c in pool) and half == "true":
        vals = np.array([c["p_true"] for c in pool], dtype=float)
        return float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0
    vals = np.array([child_mean(c, half if half != "all" else None) for c in pool], dtype=float)
    if len(vals) < 2:
        return 0.0
    noise = np.mean([child_noise_var(c, half) for c in pool])
    return float(max(np.var(vals, ddof=1) - noise, 0.0))


def allocate_in_queues(
    coefs: Sequence[float],
    k_v: Sequence[int],
    capped: Sequence[bool],
    nominal: int,
    order: Sequence[int],
    capacity: int,
    n_min: int = 1,
    hard_cap: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Partition nodes (in the given order) into queues of `capacity`, allocate
    each with the exact solver.  `hard_cap` = physical pool size per node."""
    n = len(coefs)
    out = np.zeros(n, dtype=int)
    for s in range(0, n, capacity):
        idx = list(order[s:s + capacity])
        items = []
        for i in idx:
            cap_i = int(k_v[i]) if capped[i] else None
            if hard_cap is not None:
                cap_i = int(hard_cap[i]) if cap_i is None else min(cap_i, int(hard_cap[i]))
            items.append(QueueItem(node_id=str(i), coef=float(coefs[i]), k_v=cap_i if cap_i is not None else 10 ** 6,
                                   nominal=nominal, capped=cap_i is not None))
        alloc = allocate_queue(items, n_min=n_min)
        for i, a in zip(idx, alloc):
            out[i] = a
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(x[ok], y[ok]).correlation)


def quantiles(xs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray([x for x in xs if x is not None and math.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {"median": float(np.median(a)), "lo": float(np.quantile(a, 0.025)), "hi": float(np.quantile(a, 0.975))}
