import json

from rebuttal_igpo.rebuttal.bank_analysis import (
    a2_noise_sensitivity,
    a4_tv_value_calibration,
    a5_allocation_errors,
)


def _record(node_id, depth, proxy, rewards, child_values):
    return {
        "node_id": node_id,
        "prompt_id": "p0",
        "depth": depth,
        "proxy": {
            "sigma2": proxy,
            "pair_tvs": [{"i": 0, "j": 1, "tv": 0.1}],
        },
        "R_a": {"rewards": rewards},
        "R_b": {"rewards": rewards[::-1]},
        "child_values": child_values,
    }


def test_a2_zero_noise_preserves_allocation():
    rows = [
        _record("root", 0, 1.0, [0, 1, 0, 1], [
            {"rewards": [0, 0, 0, 0]}, {"rewards": [1, 1, 1, 1]},
        ]),
        _record("child", 1, 2.0, [0, 0, 1, 1], []),
    ]
    per_prompt, summary = a2_noise_sensitivity(
        rows, noise_levels=(0.0, 0.5), repetitions=8, bootstrap_repetitions=50
    )
    zero = next(row for row in per_prompt if row["noise_sigma"] == 0.0)
    assert zero["mean_rollout_share_reassigned"] == 0.0
    assert summary[0]["n_prompt_clusters"] == 1
    assert summary[0]["mean_heldout_J_Rb"] >= 0


def test_a4_uses_candidate_value_gaps_and_keeps_pair_provenance():
    record = _record("root", 0, 1.0, [0, 1, 0, 1], [
        {"rewards": [0, 0, 0, 0]},
        {"rewards": [1, 1, 1, 1]},
    ])
    pairs, summary = a4_tv_value_calibration([record], bins=2)
    assert len(pairs) == 1
    assert pairs[0]["absolute_value_gap"] == 1.0
    assert summary["n_pairs"] == 1


def test_a5_uses_common_feasible_budget_and_prompt_cluster_rows():
    candidates = [{"full_text": f"child-{i}"} for i in range(4)]
    children = [{"rewards": [0, 1, 0, 1]} for _ in candidates]
    records = [
        {
            **_record("root", 0, 0.2, [0, 1, 0, 1], children),
            "proxy": {
                "sigma2": 0.2,
                "candidates": candidates,
                "pair_tvs": [{"i": 0, "j": 1, "tv": 0.2}],
                "pilot_entropy_mean_top5": 0.5,
                "pilot_nll_mean": 0.7,
            },
        },
        {
            **_record("child", 1, 0.4, [0, 0, 1, 1], children[:3]),
            "proxy": {
                "sigma2": 0.4,
                "candidates": candidates[:3],
                "pair_tvs": [{"i": 0, "j": 1, "tv": 0.3}],
                "pilot_entropy_mean_top5": 0.6,
                "pilot_nll_mean": 0.8,
            },
        },
    ]
    per_prompt, summary = a5_allocation_errors(records, bootstrap_repetitions=50)
    assert len(per_prompt) == 3 * 9
    assert len(summary) == 3 * 9
    assert {row["budget"] for row in per_prompt} == {7}
    trained = next(row for row in per_prompt if row["bound_regime"] == "trained" and row["strategy"] == "eq7_exact")
    assert sum(map(int, json.loads(trained["allocations_json"]))) == 7
