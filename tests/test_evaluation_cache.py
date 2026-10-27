import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from eval_checkpoints import (  # noqa: E402
    evaluation_signature,
    refresh_cached_gpu_hours,
    result_is_current,
    write_result_metadata,
)
from eval_pending import pending  # noqa: E402
from vdra_core import verifier  # noqa: E402


def _signature(checkpoint: Path, *, problems=None, **overrides):
    values = {
        "run_id": "R1_s42",
        "step": 2,
        "checkpoint_path": str(checkpoint),
        "benchmark": "math500",
        "dataset_spec": "data/math:test",
        "problems": problems or [{"id": "p1", "problem": "1+1", "answer": "2"}],
        "mode": "greedy",
        "max_model_len": 2048,
        "tokenizer": None,
        "seed": 42,
        "temperature": 0.0,
        "sample_k": 1,
        "verifier_backend": "treetune",
        "verifier_fingerprint": "implementation-a",
        "gpu_hours": 1.5,
        "limit": None,
    }
    values.update(overrides)
    return evaluation_signature(**values)


def test_evaluation_signature_binds_checkpoint_dataset_and_settings(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"step": 2}', encoding="utf-8")

    reference = _signature(checkpoint)
    assert _signature(checkpoint) == reference
    assert _signature(
        checkpoint,
        problems=[{"id": "p1", "problem": "1+1", "answer": "3"}],
    ) != reference
    assert _signature(checkpoint, limit=1) != reference
    assert _signature(checkpoint, verifier_backend="boxed_exact") != reference
    assert _signature(checkpoint, verifier_fingerprint="implementation-b") != reference
    assert _signature(checkpoint, gpu_hours=9.0) == reference

    (checkpoint / "config.json").write_text('{"step": 3}', encoding="utf-8")
    assert _signature(checkpoint) != reference


def test_evaluation_cache_requires_matching_metadata_and_intact_rows(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    signature = _signature(checkpoint)
    result = tmp_path / "R1_s42.step2.math500.greedy.jsonl"
    result.write_text('{"problem_id":"p1"}\n', encoding="utf-8")

    # A legacy filename or a partial result must be reevaluated.
    assert not result_is_current(result, signature, expected_rows=1)

    write_result_metadata(result, signature, row_count=1)
    assert result_is_current(result, signature, expected_rows=1)
    assert not result_is_current(result, "different-signature", expected_rows=1)
    assert not result_is_current(result, signature, expected_rows=2)

    result.write_text('{"problem_id":"p2"}\n', encoding="utf-8")
    assert not result_is_current(result, signature, expected_rows=1)


def test_cached_gpu_hours_annotation_refresh_keeps_evaluation_current(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    signature = _signature(checkpoint)
    result = tmp_path / "result.jsonl"
    result.write_text('{"problem_id":"p1","gpu_hours":1.0}\n', encoding="utf-8")
    write_result_metadata(result, signature, row_count=1)

    assert refresh_cached_gpu_hours(
        str(result), signature, expected_rows=1, gpu_hours=2.5
    )
    assert result_is_current(result, signature, expected_rows=1)
    assert json.loads(result.read_text(encoding="utf-8"))["gpu_hours"] == 2.5


def test_tokenizer_assets_are_fingerprinted_in_evaluation_signature(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    tokenizer_file = tokenizer / "tokenizer.json"
    tokenizer_file.write_text('{"vocab":{"a":0}}', encoding="utf-8")

    reference = _signature(checkpoint, tokenizer=str(tokenizer))
    assert _signature(checkpoint, tokenizer=str(tokenizer)) == reference
    tokenizer_file.write_text('{"vocab":{"a":1}}', encoding="utf-8")
    assert _signature(checkpoint, tokenizer=str(tokenizer)) != reference


def test_verifier_fingerprint_tracks_the_active_scorer_source(monkeypatch):
    def scorer_a(response, gold, problem):
        return 1.0

    def scorer_b(response, gold, problem):
        return 0.0

    monkeypatch.setattr(verifier, "_BACKEND", ("test", scorer_a))
    fingerprint_a = verifier.backend_fingerprint()
    assert verifier.backend_fingerprint() == fingerprint_a

    monkeypatch.setattr(verifier, "_BACKEND", ("test", scorer_b))
    assert verifier.backend_fingerprint() != fingerprint_a


def test_pending_skips_only_a_cache_for_the_current_evaluation_spec(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    problems = [{"id": "p1", "problem": "1+1", "answer": "2"}]

    class Args:
        out_dir = str(tmp_path)
        sample_k = 0
        sample_datasets = ""
        sample_min_step = 100
        max_model_len = 2048
        tokenizer = None
        base_model = None
        seed = 42
        temperature = 0.6
        limit = None

    signature = evaluation_signature(
        run_id="R1_s42",
        step=2,
        checkpoint_path=str(checkpoint),
        benchmark="math500",
        dataset_spec="data/math:test",
        problems=problems,
        mode="greedy",
        max_model_len=2048,
        tokenizer=str(checkpoint),
        seed=42,
        temperature=0.0,
        sample_k=1,
        verifier_backend=verifier.backend_name(),
        verifier_fingerprint=verifier.backend_fingerprint(),
        gpu_hours=None,
        limit=None,
    )
    output = Path(Args.out_dir) / "R1_s42.step2.math500.greedy.jsonl"
    output.write_text('{"problem_id":"p1"}\n', encoding="utf-8")
    write_result_metadata(str(output), signature, row_count=1)

    benches = {"math500": "data/math:test"}
    cache_inputs = {
        "checkpoint_path": str(checkpoint),
        "problems_by_bench": {"math500": problems},
    }
    assert pending("R1_s42", 2, benches, Args, **cache_inputs) == []

    changed_problems = [{"id": "p1", "problem": "1+1", "answer": "3"}]
    assert pending(
        "R1_s42",
        2,
        benches,
        Args,
        checkpoint_path=str(checkpoint),
        problems_by_bench={"math500": changed_problems},
    ) == ["math500"]
