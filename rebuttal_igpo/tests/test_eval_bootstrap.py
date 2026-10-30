import pytest

from rebuttal_igpo.rebuttal.eval_bootstrap import paired_bootstrap


def test_paired_eval_bootstrap_matches_prompts_and_reports_difference():
    spo = [
        {"prompt_id": "p1", "avg": 0.25, "pass": 1.0},
        {"prompt_id": "p2", "avg": 0.5, "pass": 1.0},
        {"prompt_id": "p3", "avg": 0.0, "pass": 0.0},
    ]
    vdra = [
        {"prompt_id": "p3", "avg": 0.25, "pass": 1.0},
        {"prompt_id": "p1", "avg": 0.5, "pass": 1.0},
        {"prompt_id": "p2", "avg": 0.75, "pass": 1.0},
    ]
    rows, summaries = paired_bootstrap(
        spo, vdra, dataset_name="fixture", samples=4, replicates=100, seed=7
    )
    avg_diff = next(
        row for row in summaries
        if row["metric"] == "avg@4" and row["policy"] == "vdra-spo"
    )
    assert len(rows) == 3
    assert avg_diff["estimate"] == pytest.approx(0.25)
    assert avg_diff["ci_low"] <= avg_diff["estimate"] <= avg_diff["ci_high"]
    assert avg_diff["n_prompts"] == 3


def test_paired_eval_bootstrap_rejects_prompt_mismatch():
    left = [{"prompt_id": "a", "avg": 0.0, "pass": 0.0}]
    right = [{"prompt_id": "b", "avg": 1.0, "pass": 1.0}]
    with pytest.raises(ValueError, match="same nonempty prompt identities"):
        paired_bootstrap(left, right, dataset_name="fixture")
