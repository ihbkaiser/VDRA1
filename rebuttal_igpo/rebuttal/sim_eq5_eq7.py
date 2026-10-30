"""Synthetic checks for variance identities and adaptive-selection caveats."""

from __future__ import annotations

import json
import math
import random
from typing import Sequence


def check_eq5_parent_identity(
    *, n_parent: int = 4, variances: Sequence[float] = (0.2, 0.5, 1.0, 1.4),
    replicates: int = 20000, seed: int = 0,
) -> dict[str, float | int]:
    if n_parent < 1 or len(variances) != n_parent:
        raise ValueError("one nonnegative variance is required per parent child")
    if any(v < 0 for v in variances):
        raise ValueError("variances must be nonnegative")
    rng = random.Random(seed)
    totals = 0.0
    for _ in range(replicates):
        sample = [rng.gauss(0.0, math.sqrt(v)) for v in variances]
        mean = sum(sample) / n_parent
        s_p = sum((x - mean) ** 2 for x in sample) / n_parent
        totals += n_parent * s_p
    empirical = totals / replicates
    expected = (n_parent - 1) / n_parent * sum(variances)
    return {"n_parent": n_parent, "empirical": empirical, "expected": expected, "replicates": replicates}


def check_tau_identity(
    *, n_children: int = 5, child_means: Sequence[float] = (-1.0, -0.4, 0.2, 0.7, 1.4),
    within_variance: float = 0.36, replicates: int = 20000, seed: int = 0,
) -> dict[str, float | int]:
    if n_children < 1 or len(child_means) != n_children:
        raise ValueError("one child mean is required per sampled child")
    if within_variance < 0:
        raise ValueError("within-child variance must be nonnegative")
    mean = sum(child_means) / n_children
    sigma2 = sum((x - mean) ** 2 for x in child_means) / n_children
    rng = random.Random(seed)
    squares = 0.0
    for _ in range(replicates):
        estimate = sum(
            rng.choice(child_means) + rng.gauss(0.0, math.sqrt(within_variance))
            for _ in range(n_children)
        ) / n_children
        squares += (estimate - mean) * (estimate - mean)
    empirical = squares / replicates
    expected = (sigma2 + within_variance) / n_children
    return {"n_children": n_children, "sigma2": sigma2, "tau_bar2": within_variance,
            "empirical": empirical, "expected": expected, "replicates": replicates}


def simulate_selection_bias(*, replicates: int = 20000, seed: int = 0) -> dict[str, float | int]:
    """Synthetic top-logprob pilot reuse where log-likelihood correlates with reward."""
    rng = random.Random(seed)
    all_means = 0.0
    top_means = 0.0
    candidates, selected = 32, 8
    for _ in range(replicates):
        values = [rng.gauss(0.0, 1.0) for _ in range(candidates)]
        logp = [0.7 * value + rng.gauss(0.0, 0.8) for value in values]
        keep = sorted(range(candidates), key=lambda i: (-logp[i], i))[:selected]
        all_means += sum(values) / candidates
        top_means += sum(values[i] for i in keep) / selected
    return {
        "all_mean": all_means / replicates,
        "top_mean": top_means / replicates,
        "candidate_count": candidates,
        "selected_count": selected,
        "replicates": replicates,
    }


def simulate_adaptive_plugin_bias(*, replicates: int = 20000, seed: int = 0) -> dict[str, float | int]:
    """Show selection optimism when the same noisy pilot coefficients choose n."""
    rng = random.Random(seed)
    true = (0.5, 2.0, 5.0, 12.0)
    budget = 20
    predicted_sum = true_sum = 0.0
    for _ in range(replicates):
        estimates = []
        for variance in true:
            values = [rng.gauss(0.0, math.sqrt(variance)) for _ in range(4)]
            mean = sum(values) / len(values)
            estimates.append(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
        weights = [math.sqrt(max(x, 0.0)) for x in estimates]
        allocations = [1] * len(true)
        left = budget - len(true)
        total = sum(weights)
        raw = [left * w / total if total else left / len(true) for w in weights]
        base = [int(math.floor(x)) for x in raw]
        allocations = [a + b for a, b in zip(allocations, base)]
        left -= sum(base)
        for i in sorted(range(len(true)), key=lambda j: (-(raw[j] % 1.0), j))[:left]:
            allocations[i] += 1
        predicted_sum += sum(c / n for c, n in zip(estimates, allocations))
        true_sum += sum(c / n for c, n in zip(true, allocations))
    return {
        "predicted_objective": predicted_sum / replicates,
        "true_objective": true_sum / replicates,
        "budget": budget,
        "replicates": replicates,
    }


def main() -> int:
    result = {
        "eq5_fixed_allocation": check_eq5_parent_identity(),
        "tau_iid": check_tau_identity(),
        "top_logprob_reuse": simulate_selection_bias(),
        "adaptive_plugin_selection": simulate_adaptive_plugin_bias(),
        "n_parent_1_root": check_eq5_parent_identity(n_parent=1, variances=(1.2,)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
