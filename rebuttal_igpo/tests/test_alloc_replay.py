import itertools

import pytest

from rebuttal_igpo.rebuttal.alloc_replay import (
    allocate,
    allocate_gear_as_trained,
    bounds_for_regime,
    noise_sensitivity,
    signal_coefficients,
)


def _brute(coeffs, lower, upper, budget):
    best = None
    best_value = float("inf")
    for counts in itertools.product(*(range(lo, hi + 1) for lo, hi in zip(lower, upper))):
        if sum(counts) != budget:
            continue
        value = sum(c / n if n else (0.0 if c == 0 else float("inf")) for c, n in zip(coeffs, counts))
        if value < best_value:
            best, best_value = list(counts), value
    return best


@pytest.mark.parametrize(
    "coeffs,lower,upper,budget",
    [
        ([1.0, 4.0], [0, 0], [4, 4], 4),
        ([0.5, 1.5, 3.0], [1, 0, 0], [3, 3, 3], 5),
        ([0.0, 2.0, 5.0], [0, 1, 0], [2, 3, 4], 6),
    ],
)
def test_eq7_exact_matches_brute_force(coeffs, lower, upper, budget):
    assert allocate(coeffs, lower, upper, budget, "eq7_exact") == _brute(
        coeffs, lower, upper, budget
    )


def test_fixed_branch_fallback_and_rounding_are_feasible():
    counts = allocate([0.0, 0.0, 0.0], [1, 1, 1], [4, 4, 4], 8, "fixed_branch")
    assert counts == [3, 3, 2]
    assert sum(counts) == 8


def test_uniform_water_fills_evenly_under_bounds():
    assert allocate([99, 0, 2], [0, 1, 0], [3, 4, 4], 7, "uniform") == [2, 3, 2]


def test_as_trained_uses_floor_and_reports_underallocation_semantics():
    counts = allocate([4.0, 1.0, 0.0], [0, 0, 0], [None, None, None], 5, "as_trained")
    # sqrt weights are 2:1:0; floor rounding leaves one unit unused.
    assert counts == [3, 1, 0]


def test_capped_regimes_and_vdra_lambda_signal():
    assert bounds_for_regime("capped", 3, upper=[10, 5, 4]) == ([0, 0, 0], [6, 5, 4])
    assert bounds_for_regime("capped_k", 2, candidates=[4, 9], default_cap=6) == ([0, 0], [4, 6])
    vals = signal_coefficients(
        "vdra_lambda", sigma2=[2.0, 3.0], parent_counts=[1, 4]
    )
    assert vals == pytest.approx([0.0, 2.25])


def test_random_permutation_repairs_bounds_and_budget():
    counts = allocate(
        [1, 1, 1], [1, 0, 0], [2, 3, 2], 6, "random_perm",
        base_counts=[0, 5, 1], seed=3,
    )
    assert counts == [2, 2, 2]
    assert sum(counts) == 6
    assert all(1 <= value <= cap for value, cap in zip(counts, [2, 3, 2]))


def test_infeasible_budget_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        allocate([1, 1], [2, 2], [3, 3], 3, "eq7_exact")


def test_noise_sensitivity_zero_noise_reproduces_baseline():
    row = noise_sensitivity([1, 2, 5], budget=12, repetitions=4, noise_levels=[0.0])[0]
    assert row["mean_rollout_share_reassigned"] == 0.0


def test_as_trained_matches_supplied_gear_allocator_when_available():
    import importlib.util
    import sys
    from pathlib import Path

    source = Path("/workspace/src/gear_source_rebuttal/gear/treetune/gear/budget_allocation.py")
    if not source.exists():
        pytest.skip("source snapshot is not available in this checkout")
    spec = importlib.util.spec_from_file_location("gear_budget_allocation_for_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    variances = [0.0, 0.01, 0.02, 0.12, 0.36]
    ids = ["root", "a", "b", "c", "d"]
    nodes = [
        {"gear_segment_id": node_id, "gear_reward_variance": variance}
        for node_id, variance in zip(ids, variances)
    ]
    for distribute, fallback in ((False, False), (True, False), (True, True)):
        expected = module.allocate_branch_factors(
            nodes, total_budget=17, lambda_=0.02, n_min=1,
            distribute_remainder=distribute, fallback_uniform=fallback,
        ).allocations
        actual = allocate_gear_as_trained(
            variances, 17, node_ids=ids, lambda_=0.02, n_min=1,
            distribute_remainder=distribute, fallback_uniform=fallback,
        )
        assert actual == [expected[node_id] for node_id in ids]
