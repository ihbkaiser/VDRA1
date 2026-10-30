import pytest

from rebuttal_igpo.rebuttal.metrics import (
    calibration_bins,
    cluster_bootstrap,
    j_objective,
    parent_value_mse,
    spearman,
    split_mc_halves,
    top_quartile_overlap,
)


def test_j_objective_and_parent_value_mse():
    assert j_objective([2.0, 3.0], [2, 3]) == pytest.approx(2.0)
    assert parent_value_mse([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)
    assert j_objective([1.0], [0]) == float("inf")


def test_spearman_handles_ties():
    assert spearman([1, 2, 2, 4], [4, 3, 3, 1]) == pytest.approx(-1.0)


def test_prompt_cluster_bootstrap_is_deterministic():
    rows = [
        {"prompt_id": "a", "value": 1.0},
        {"prompt_id": "a", "value": 3.0},
        {"prompt_id": "b", "value": 5.0},
    ]
    first = cluster_bootstrap(rows, value_key="value", replicates=100, seed=9)
    second = cluster_bootstrap(rows, value_key="value", replicates=100, seed=9)
    assert first == second
    assert first["n_clusters"] == 2


def test_mc_halves_are_disjoint_and_reproducible():
    a, b = split_mc_halves(list(range(20)), seed=5)
    a2, b2 = split_mc_halves(list(range(20)), seed=5)
    assert a == a2 and b == b2
    assert set(a).isdisjoint(b)
    assert len(a) + len(b) == 20


def test_calibration_and_top_quartile_overlap():
    bins = calibration_bins([1, 2, 3, 4], [2, 4, 6, 8], bins=2)
    assert [row["n"] for row in bins] == [2, 2]
    assert bins[0]["mean_target"] == pytest.approx(3)
    assert top_quartile_overlap([1, 2, 3, 4], [2, 1, 4, 3]) == 0.0
