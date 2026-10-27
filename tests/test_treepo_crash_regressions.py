from pathlib import Path

import numpy as np

from vdra_core.treepo import concat_non_tensor_arrays, random_fallback_block
from integrations.treepo import vdra_math_reward
from modal_app.phase_c import _assert_treepo_stability_patch, _phase_f_treepo_fix_paths


ROOT = Path(__file__).resolve().parents[1]


def test_empty_random_fallback_uses_zero_block_without_sampling(monkeypatch):
    def unexpected_randint(*args, **kwargs):
        raise AssertionError("empty response must not sample a fallback block")

    monkeypatch.setattr(np.random, "randint", unexpected_randint)

    assert random_fallback_block(0, max_fallback_window=-1) == 0


def test_nonempty_random_fallback_preserves_treepo_block_selection(monkeypatch):
    calls = []

    def choose_first_block(low, high):
        calls.append((low, high))
        return low

    monkeypatch.setattr(np.random, "randint", choose_first_block)

    # Four blocks, but max_fallback_window limits the choice to one.
    assert random_fallback_block(4, max_fallback_window=1) == 1
    assert calls == [(1, 2)]


def test_treepo_concat_keeps_ragged_non_tensor_rows_as_objects():
    flat = np.array(["a", "b"], dtype=object)
    nested = np.empty((2, 2, 2), dtype=object)
    nested[0] = [[1, 2], [3, 4]]
    nested[1] = [[5, 6], [7, 8]]

    result = concat_non_tensor_arrays([flat, nested])

    assert result.shape == (4,)
    assert result[:2].tolist() == ["a", "b"]
    np.testing.assert_array_equal(result[2], nested[0])
    np.testing.assert_array_equal(result[3], nested[1])


def test_treepo_concat_preserves_uniform_non_tensor_array_shape():
    left = np.arange(6).reshape(2, 3)
    right = np.arange(6, 12).reshape(2, 3)

    result = concat_non_tensor_arrays([left, right])

    assert result.shape == (4, 3)
    np.testing.assert_array_equal(result, np.concatenate([left, right], axis=0))


def test_treepo_installer_applies_the_runtime_crash_fix_patch():
    installer = (ROOT / "integrations/treepo/install.sh").read_text(encoding="utf-8")
    patch = (ROOT / "integrations/treepo/training_stability.patch").read_text(
        encoding="utf-8"
    )
    reward_patch = (ROOT / "integrations/treepo/reward_timeout.patch").read_text(
        encoding="utf-8"
    )
    ragged_patch = (ROOT / "integrations/treepo/ragged_dataproto.patch").read_text(
        encoding="utf-8"
    )

    assert "training_stability.patch" in installer
    assert installer.index("vdra_treepo.patch") < installer.index("training_stability.patch")
    assert installer.index("training_stability.patch") < installer.index("reward_timeout.patch")
    assert "--unidiff-zero" in installer
    assert "enable_prefix_caching=True" in patch
    assert "enable_prefix_caching=False" in patch
    assert "random_fallback_block" in patch
    assert "reward_timeout.patch" in installer
    assert "self.timeout_seconds = 30" in reward_patch
    assert "ragged_dataproto.patch" in installer
    assert "--unidiff-zero" in installer
    assert "concat_non_tensor_arrays" in ragged_patch
    train_script = (ROOT / "scripts/20_train.sh").read_text(encoding="utf-8")
    assert "export VDRA_SINGLE_RANK_COLLECTIVES=1" in train_script
    assert '${VDRA_HOME}/modal_app/runtime' in train_script


def test_treepo_reward_returns_the_mapping_dapo_requires(monkeypatch):
    monkeypatch.setattr(vdra_math_reward.verifier, "score", lambda *_args: 1.0)

    result = vdra_math_reward.compute_score(
        data_source="math",
        solution_str="\\boxed{1}",
        ground_truth="1",
        extra_info={"problem": "1+0"},
    )

    assert result == {"score": 1.0, "acc": 1.0, "extra_info": {}}


def test_modal_treepo_preflight_rejects_an_image_without_both_fixes(tmp_path):
    source_path = tmp_path / "vllm_rollout_spmd.py"
    reward_path = tmp_path / "vdra_math_reward.py"
    manager_path = tmp_path / "dapo.py"
    collective_path = ROOT / "modal_app/single_rank_collectives.py"
    protocol_path = tmp_path / "protocol.py"
    source_path.write_text(
        "enable_prefix_caching=True\n"
        "fallback_block = np.random.randint(1, total_blocks + 1)\n",
        encoding="utf-8",
    )
    reward_path.write_text("return float(verifier.score(...))\n", encoding="utf-8")
    manager_path.write_text("self.timeout_seconds = 5\n", encoding="utf-8")
    protocol_path.write_text("non_tensor_batch[key] = np.concatenate(val, axis=0)\n", encoding="utf-8")

    try:
        _assert_treepo_stability_patch(
            source_path, reward_path, manager_path, collective_path, protocol_path
        )
    except RuntimeError as exc:
        assert "prefix_cache_disabled" in str(exc)
        assert "empty_fallback_guarded" in str(exc)
        assert "reward_schema_valid" in str(exc)
        assert "reward_timeout_extended" in str(exc)
        assert "ragged_non_tensor_concat_guarded" in str(exc)
    else:
        raise AssertionError("the Modal preflight accepted a stale TreePO image")


def test_modal_treepo_preflight_accepts_the_fixed_source(tmp_path):
    source_path = tmp_path / "vllm_rollout_spmd.py"
    reward_path = tmp_path / "vdra_math_reward.py"
    manager_path = tmp_path / "dapo.py"
    collective_path = ROOT / "modal_app/single_rank_collectives.py"
    protocol_path = tmp_path / "protocol.py"
    source_path.write_text(
        "enable_prefix_caching=False\n"
        "fallback_block = random_fallback_block(total_blocks, max_fallback_window)\n",
        encoding="utf-8",
    )
    reward_path.write_text(
        'return {"score": score, "acc": score, "extra_info": {}}\n',
        encoding="utf-8",
    )
    manager_path.write_text("self.timeout_seconds = 30\n", encoding="utf-8")
    protocol_path.write_text(
        "non_tensor_batch[key] = concat_non_tensor_arrays(val)\n", encoding="utf-8"
    )

    assert _assert_treepo_stability_patch(
        source_path, reward_path, manager_path, collective_path, protocol_path
    ) == {
        "prefix_cache_disabled": True,
        "empty_fallback_guarded": True,
        "reward_schema_valid": True,
        "reward_timeout_extended": True,
        "fsdp_world_one_broadcast_guarded": True,
        "ragged_non_tensor_concat_guarded": True,
    }


def test_modal_treepo_smoke_paths_use_posix_separators():
    assert _phase_f_treepo_fix_paths("/work/mock/phase_f_treepo_fix_run1", "R3") == {
        "RUNS_DIR": "/work/mock/phase_f_treepo_fix_run1/R3/runs",
        "RESULTS_DIR": "/work/mock/phase_f_treepo_fix_run1/R3/results",
        "BANKS_DIR": "/work/mock/phase_f_treepo_fix_run1/R3/banks",
    }
