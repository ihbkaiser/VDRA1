import pytest

from modal_app.logprob_compare import compare_runs


def _run(stack, token_ids, token_logprobs):
    return {
        "stack": stack,
        "model_revision": "ad9f0ae",
        "tokenizer_sha256": {"tokenizer.json": "same-tokenizer"},
        "prompts": [
            {
                "text": "fixed prompt",
                "token_ids": token_ids,
                "token_logprobs": token_logprobs,
            }
        ],
    }


def test_compare_runs_reports_per_token_maximum_and_skips_bos():
    legacy = _run("vllm-0.4.0.post1", [1, 2, 3], [None, -0.4, -1.2])
    current = _run("vllm-0.9.2", [1, 2, 3], [None, -0.45, -1.1])

    result = compare_runs(legacy, current)

    assert result["prompts"][0]["compared_tokens"] == 2
    assert result["prompts"][0]["max_abs_diff"] == pytest.approx(0.1)
    assert result["max_abs_per_token_diff"] == pytest.approx(0.1)


def test_compare_runs_rejects_model_or_tokenizer_mismatch():
    legacy = _run("old", [1, 2], [None, -0.4])
    current = _run("new", [1, 3], [None, -0.4])

    with pytest.raises(ValueError, match="token IDs differ"):
        compare_runs(legacy, current)


def test_compare_runs_rejects_incomparable_missing_logprobs():
    legacy = _run("old", [1, 2], [None, -0.4])
    current = _run("new", [1, 2], [None, None])

    with pytest.raises(ValueError, match="missing logprob"):
        compare_runs(legacy, current)
