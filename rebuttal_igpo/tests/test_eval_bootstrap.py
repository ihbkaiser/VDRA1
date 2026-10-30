import pytest

from rebuttal_igpo.rebuttal import eval_bootstrap
from rebuttal_igpo.rebuttal.eval_bootstrap import paired_bootstrap, summarize_smoke_token_rate


def test_smoke_retokenization_counts_visible_candidate_text():
    class WhitespaceTokenizer:
        def encode(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return text.split()

    counts = eval_bootstrap.recount_candidate_answer_tokens(
        [["one two", "three"], ["", "four five six"]],
        WhitespaceTokenizer(),
    )
    assert counts == [[2, 1], [0, 3]]


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


def test_eval_smoke_rate_uses_recorded_tokens_and_projects_with_margin():
    result = summarize_smoke_token_rate(
        [[10, 20], [30, 40]], elapsed_seconds=10.0, prompts_in_full_eval=100,
        safety_margin=1.5,
    )
    assert result["output_tokens"] == 100
    assert result["smoke_completions"] == 4
    assert result["output_tokens_per_second"] == pytest.approx(10.0)
    assert result["projected_full_eval_seconds_with_margin"] == pytest.approx(750.0)
    assert result["token_count_source"] == "recorded"
