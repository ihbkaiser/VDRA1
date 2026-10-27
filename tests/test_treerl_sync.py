import importlib.util
from types import SimpleNamespace


_SYNC_SPEC = importlib.util.find_spec("modal_app.treerl_sync")
treerl_sync = __import__("modal_app.treerl_sync", fromlist=["*"]) if _SYNC_SPEC else None


def test_prompt_logprob_comparison_checks_vllm_candidates_against_actor():
    assert treerl_sync is not None, "TreeRL sync verifier module is missing"
    actor_logprobs = [-0.1, -0.4, -1.2]
    vllm_logprobs = {0: SimpleNamespace(logprob=-0.11), 2: SimpleNamespace(logprob=-1.19)}

    report = treerl_sync.compare_prompt_logprobs(actor_logprobs, vllm_logprobs, atol=0.02)

    assert report["matched"] is True
    assert report["compared_tokens"] == 2
    assert report["max_abs_diff"] < 0.02


def test_prompt_logprob_comparison_fails_outside_bfloat16_tolerance():
    assert treerl_sync is not None, "TreeRL sync verifier module is missing"
    report = treerl_sync.compare_prompt_logprobs([-0.1, -0.4], {1: SimpleNamespace(logprob=-0.8)}, atol=0.05)

    assert report["matched"] is False
    assert report["max_abs_diff"] > 0.05
