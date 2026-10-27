import inspect
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_app.phase_c import _training_control_overrides
from modal_app import phase_d
from modal_app.phase_d import (
    PHASE_D_STOP_USD,
    PROJECT_SPEND_BEFORE_PROFILE_SWITCH_USD,
    MOCK_PAIRS,
    MOCK_TIMEOUT_SECONDS,
    INTERRUPT_DEADLINE_SECONDS,
    PHASE_D_ROOT,
    _all_pairs_complete,
    _check_phase_d_budget,
    _igpo_logprob_profile_environment,
    _is_checkpoint_for_interrupt,
    _is_monotonic_gpu_hours,
    _is_safe_interrupt_checkpoint,
    _phase_d_cost_so_far,
    _validate_resume_equivalence,
    mock_environment,
    resume_environment,
)
from scripts import run_phase_d_with_budget as budgeted_phase_d


def test_phase_d_pairs_and_settings_match_the_mock_spec():
    assert MOCK_PAIRS == {
        "igpo": ("R1", "R2"),
        "treepo": ("R3", "R4"),
        "treerl": ("R5", "R6"),
    }
    env = mock_environment()
    expected = {
        "STOP_STEP": "4",
        "EVAL_EVERY_N": "2",
        "EVAL_AT_END": "1",
        "EVAL_STEP0": "1",
        "RESUME_EVERY": "2",
        "KEEP_EVERY": "2",
        "SEEDS": "42",
        "SAMPLE_K": "2",
        "VDRA_EVAL_LIMIT": "16",
        "COMPUTE_MATCHED": "0",
        "MAX_RESTARTS": "0",
    }
    assert {key: env[key] for key in expected} == expected
    assert env["RUNS_DIR"] == str(PHASE_D_ROOT / "runs")
    assert env["RESULTS_DIR"] == str(PHASE_D_ROOT / "results")


def test_igpo_mock_environment_supports_an_isolated_sequence_limit_override():
    assert "igpo_max_num_seqs" in inspect.signature(mock_environment).parameters
    assert callable(getattr(phase_d, "phase_d_run_root", None))
    assert "igpo_max_num_seqs" in inspect.signature(phase_d.phase_d_run_root).parameters
    assert mock_environment()["INGPO_VLLM_MAX_NUM_SEQS"] == "64"
    assert mock_environment()["INGPO_MAX_CONCURRENT_GENERATIONS"] == "64"
    assert mock_environment()["INGPO_DATASET_NUM_SAMPLES_PER_ITERATION"] == "4"
    assert mock_environment()["INGPO_NUM_EPISODES_PER_ITERATION"] == "64"
    assert mock_environment(igpo_max_num_seqs=96)["INGPO_VLLM_MAX_NUM_SEQS"] == "96"
    sweep_env = mock_environment(igpo_max_num_seqs=128)
    assert sweep_env["INGPO_VLLM_MAX_NUM_SEQS"] == "128"
    assert sweep_env["INGPO_MAX_CONCURRENT_GENERATIONS"] == "64"
    assert sweep_env["INGPO_DATASET_NUM_SAMPLES_PER_ITERATION"] == "4"
    assert sweep_env["INGPO_NUM_EPISODES_PER_ITERATION"] == "64"
    common_script = (
        Path(__file__).resolve().parents[1] / "work" / "igpo" / "scripts" / "_common.sh"
    ).read_text(encoding="utf-8")
    assert 'local vllm_max_num_seqs="${INGPO_VLLM_MAX_NUM_SEQS:-1024}"' in common_script
    assert "max_num_seqs: ${vllm_max_num_seqs}" in common_script
    with pytest.raises(ValueError, match="between 1 and 128"):
        mock_environment(igpo_max_num_seqs=0)
    assert phase_d.phase_d_run_root("seq96") == PHASE_D_ROOT.parent / "phase_d_v7_seq96"
    assert phase_d.phase_d_run_root(
        "seq96", igpo_max_num_seqs=96
    ) == PHASE_D_ROOT.parent / "phase_d_v7_seq96"
    with pytest.raises(ValueError, match="isolated run_tag"):
        phase_d.phase_d_run_root("", igpo_max_num_seqs=96)
    with pytest.raises(ValueError, match="alphanumeric"):
        phase_d.phase_d_run_root("../seq96")


def test_phase_d_speed_override_is_exposed_through_budgeted_cli():
    root = Path(__file__).resolve().parents[1]
    runner = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_phase_d_with_budget.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert runner.returncode == 0, runner.stderr
    assert "--igpo-max-num-seqs" in runner.stdout
    assert "--run-tag" in runner.stdout

    build_command = getattr(budgeted_phase_d, "build_modal_command", None)
    assert callable(build_command)
    args = SimpleNamespace(
        framework="igpo",
        igpo_config_preflight=False,
        igpo_image_preflight=False,
        igpo_logprob_profile=False,
        igpo_r1_reference=False,
        igpo_max_num_seqs=96,
        run_tag="seq96",
    )
    command = build_command(args, "modal_app/phase_d.py", "20261027")
    assert "--detach" in command
    assert command[-6:] == [
        "--framework", "igpo",
        "--igpo-max-num-seqs", "96",
        "--run-tag", "seq96",
    ]

    profile_args = SimpleNamespace(
        framework="igpo",
        igpo_config_preflight=False,
        igpo_image_preflight=False,
        igpo_logprob_profile=True,
        igpo_r1_reference=False,
        igpo_max_num_seqs=96,
        run_tag="seq96profile",
    )
    profile_command = build_command(
        profile_args, "modal_app/phase_d.py", "20261027"
    )
    assert profile_command[-7:] == [
        "--igpo-logprob-profile",
        "--framework", "igpo",
        "--igpo-max-num-seqs", "96",
        "--run-tag", "seq96profile",
    ]
    assert "seq96profile" in profile_command[profile_command.index("--name") + 1]

    modal_help = subprocess.run(
        ["modal", "run", "modal_app/phase_d.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert modal_help.returncode == 0, modal_help.stderr
    assert "--igpo-max-num-seqs" in modal_help.stdout


def test_r2_no_prefix_cache_config_is_not_overridden_by_runtime_overlay():
    root = Path(__file__).resolve().parents[1]
    common_script = (root / "work" / "igpo" / "scripts" / "_common.sh").read_text(
        encoding="utf-8"
    )
    active_runtime_lines = "\n".join(
        line for line in common_script.splitlines() if not line.lstrip().startswith("#")
    )
    no_prefix_config = (
        root / "work" / "igpo" / "configs" / "episode_generators" / "no_prefix_caching.jsonnet"
    ).read_text(encoding="utf-8")
    r2_launcher = (root / "scripts" / "20_train.sh").read_text(encoding="utf-8")

    assert "enable_prefix_caching:" not in active_runtime_lines
    assert "enable_prefix_caching: false" in no_prefix_config
    assert 'configs/episode_generators/no_prefix_caching.jsonnet' in r2_launcher


def test_igpo_logprob_profile_uses_fresh_root_and_sequence_limit(tmp_path):
    root = tmp_path / "phase_d_v7_seq96profile"
    env = _igpo_logprob_profile_environment(
        root,
        "/work/hf_cache/model",
        96,
    )

    assert env["INGPO_VLLM_MAX_NUM_SEQS"] == "96"
    assert env["INGPO_MAX_CONCURRENT_GENERATIONS"] == "64"
    assert env["RUNS_DIR"] == str(root / "runs")
    assert env["RESULTS_DIR"] == str(root / "results")
    assert env["VDRA_METRICS_JSONL"] == str(
        root / "runs" / "R1_s42" / "phase_d_metrics.jsonl"
    )
    assert env["STOP_STEP"] == "1"
    assert env["EVAL_EVERY_N"] == "0"


def test_mock_control_values_survive_training_environment_defaults():
    controls = _training_control_overrides(mock_environment())
    assert controls == {
        "MAX_RESTARTS": "0",
        "EVAL_EVERY_N": "2",
        "EVAL_AT_END": "1",
        "EVAL_STEP0": "1",
    }
    assert _training_control_overrides({}) == {
        "MAX_RESTARTS": "0",
        "EVAL_EVERY_N": "25",
        "EVAL_AT_END": "1",
        "EVAL_STEP0": "1",
    }


def test_gpu_hour_series_must_be_monotonic():
    assert _is_monotonic_gpu_hours({"0": 0.0, "2": 0.1, "4": 0.3})
    assert not _is_monotonic_gpu_hours({"0": 0.0, "2": 0.3, "4": 0.2})


def test_mock_entrypoint_only_accepts_complete_pair_results():
    assert _all_pairs_complete({"igpo": {"status": "complete"}, "treepo": {"status": "complete"}})
    assert not _all_pairs_complete({"igpo": {"status": "complete"}, "treerl": {"status": "failed"}})


def test_resume_mock_keeps_stop_eval_settings_in_an_isolated_root():
    env = resume_environment()
    assert env["STOP_STEP"] == "4"
    assert env["EVAL_EVERY_N"] == "2"
    assert env["RUNS_DIR"] == str(PHASE_D_ROOT / "resume" / "runs")
    assert env["RESULTS_DIR"] == str(PHASE_D_ROOT / "resume" / "results")


def test_resume_controller_interrupts_only_at_target_checkpoint():
    assert _is_checkpoint_for_interrupt(2, 2)
    assert not _is_checkpoint_for_interrupt(0, 2)
    assert not _is_checkpoint_for_interrupt(4, 2)
    safe_state = {
        "resumable_update": 2,
        "last_launch": {"chunk_stop": 2},
        "heartbeat_age_s": None,
    }
    assert _is_safe_interrupt_checkpoint(safe_state, 2)
    assert not _is_safe_interrupt_checkpoint({**safe_state, "heartbeat_age_s": 1.0}, 2)
    assert not _is_safe_interrupt_checkpoint({**safe_state, "last_launch": {"chunk_stop": 4}}, 2)


def test_resume_equivalence_compares_eval_rows_and_ignores_gpu_hour_annotation(tmp_path):
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"
    for root, hours in ((reference_root, 0.1), (resumed_root, 0.3)):
        eval_dir = root / "eval"
        eval_dir.mkdir(parents=True)
        for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS:
            rows = [
                {"run_id": "R1_s42", "step": 4, "benchmark": benchmark, "mode": "greedy",
                 "problem_id": f"p{i}", "correct": 1.0, "gpu_hours": hours}
                for i in range(phase_d.PHASE_D_EVAL_LIMIT)
            ]
            path = eval_dir / f"R1_s42.step4.{benchmark}.greedy.jsonl"
            payload = "".join(json.dumps(row) + "\n" for row in rows)
            path.write_bytes(payload.encode("utf-8"))
            (eval_dir / f"{path.name}.meta.json").write_text(
                json.dumps({
                    "schema": 1,
                    "signature": "a" * 64,
                    "row_count": len(rows),
                    "result_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                }), encoding="utf-8",
            )

    signatures = {benchmark: "a" * 64 for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS}
    report = _validate_resume_equivalence(
        "R1", reference_root, resumed_root,
        reference_expected_signatures=signatures,
        resumed_expected_signatures=signatures,
    )

    assert report["matched"] is True
    assert report["mismatched_files"] == []
    assert (reference_root / "equivalence" / "eval_step4_R1_s42.json").is_file()


def test_resume_equivalence_fails_when_evaluation_rows_differ(tmp_path):
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"
    for root, correct in ((reference_root, 1.0), (resumed_root, 0.0)):
        eval_dir = root / "eval"
        eval_dir.mkdir(parents=True)
        for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS:
            rows = [
                {"run_id": "R1_s42", "step": 4, "benchmark": benchmark, "mode": "greedy",
                 "problem_id": f"p{i}", "correct": correct}
                for i in range(phase_d.PHASE_D_EVAL_LIMIT)
            ]
            path = eval_dir / f"R1_s42.step4.{benchmark}.greedy.jsonl"
            payload = "".join(json.dumps(row) + "\n" for row in rows)
            path.write_bytes(payload.encode("utf-8"))
            (eval_dir / f"{path.name}.meta.json").write_text(
                json.dumps({
                    "schema": 1,
                    "signature": "b" * 64,
                    "row_count": len(rows),
                    "result_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                }), encoding="utf-8",
            )

    with pytest.raises(RuntimeError, match="resumed step-4 evaluations differ"):
        signatures = {benchmark: "b" * 64 for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS}
        _validate_resume_equivalence(
            "R1", reference_root, resumed_root,
            reference_expected_signatures=signatures,
            resumed_expected_signatures=signatures,
        )


def _write_phase_d_evaluations(eval_dir: Path, run_id: str, *, digest_override=None):
    for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS:
        path = eval_dir / f"{run_id}.step4.{benchmark}.greedy.jsonl"
        rows = [
            {"run_id": run_id, "step": 4, "benchmark": benchmark, "mode": "greedy",
             "problem_id": f"p{i}", "correct": 1.0}
            for i in range(phase_d.PHASE_D_EVAL_LIMIT)
        ]
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        path.write_bytes(payload.encode("utf-8"))
        digest = digest_override if digest_override is not None else hashlib.sha256(payload.encode()).hexdigest()
        (eval_dir / f"{path.name}.meta.json").write_text(
            json.dumps({"schema": 1, "signature": "c" * 64,
                        "row_count": len(rows), "result_sha256": digest}), encoding="utf-8",
        )


def test_phase_d_evaluation_rows_reject_a_well_formed_but_stale_signature(tmp_path):
    root = tmp_path / "reference"
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True)
    _write_phase_d_evaluations(eval_dir, "R1_s42")
    current = {benchmark: "c" * 64 for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS}

    rows = phase_d._evaluation_rows(
        root, "R1_s42", 4, expected_signatures=current
    )
    assert len(rows) == len(phase_d.PHASE_D_EVAL_BENCHMARKS)

    stale = {benchmark: "a" * 64 for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS}
    with pytest.raises(RuntimeError, match="signature does not match current inputs"):
        phase_d._evaluation_rows(root, "R1_s42", 4, expected_signatures=stale)


def test_phase_d_expected_signature_changes_with_checkpoint_or_dataset(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text('{"revision":1}', encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"vocab":{"a":0}}', encoding="utf-8")
    dataset_specs = {}
    dataset_paths = {}
    for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS:
        dataset = tmp_path / f"{benchmark}.jsonl"
        rows = [
            {"id": f"{benchmark}-{i}", "problem": f"problem {i}", "answer": str(i)}
            for i in range(phase_d.PHASE_D_EVAL_LIMIT)
        ]
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        dataset_specs[benchmark] = str(dataset)
        dataset_paths[benchmark] = dataset

    def current_signatures():
        return phase_d._phase_d_expected_eval_signatures(
            "R1_s42", 4, str(checkpoint), str(tokenizer), dataset_specs,
            limit=phase_d.PHASE_D_EVAL_LIMIT,
        )

    reference = current_signatures()
    assert current_signatures() == reference
    config.write_text('{"revision":2}', encoding="utf-8")
    assert current_signatures() != reference

    reference = current_signatures()
    changed_rows = [
        {"id": f"math500-{i}", "problem": f"problem {i}", "answer": str(i)}
        for i in range(phase_d.PHASE_D_EVAL_LIMIT)
    ]
    changed_rows[0]["answer"] = "changed"
    dataset_paths["math500"].write_text(
        "".join(json.dumps(row) + "\n" for row in changed_rows), encoding="utf-8"
    )
    assert current_signatures() != reference


def test_phase_d_step_zero_signature_uses_base_model_even_if_run_has_step_zero_ckpt(tmp_path):
    run_dir = tmp_path / "run"
    for step in (0, 2, 4):
        checkpoint = (
            run_dir / "checkpoints" / f"ckpt--iter_{step:04d}--epoch_1.00" / "hf_pretrained"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    base_model = str(tmp_path / "base_model")

    checkpoints = phase_d._phase_d_checkpoint_paths(run_dir, base_model)

    assert checkpoints[0] == base_model
    assert Path(checkpoints[2]).is_relative_to(run_dir)
    assert Path(checkpoints[4]).is_relative_to(run_dir)


def test_resume_equivalence_rejects_missing_benchmark_files(tmp_path):
    root = tmp_path / "reference"
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True)
    _write_phase_d_evaluations(eval_dir, "R1_s42")
    missing = eval_dir / f"R1_s42.step4.{phase_d.PHASE_D_EVAL_BENCHMARKS[-1]}.greedy.jsonl"
    missing.unlink()
    Path(f"{missing}.meta.json").unlink()

    with pytest.raises(RuntimeError, match="expected evaluation set"):
        phase_d._evaluation_rows(
            root,
            "R1_s42",
            4,
            expected_signatures={
                benchmark: "c" * 64 for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS
            },
        )


def test_resume_equivalence_rejects_result_bytes_that_do_not_match_sidecar(tmp_path):
    root = tmp_path / "reference"
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True)
    _write_phase_d_evaluations(eval_dir, "R1_s42", digest_override="0" * 64)

    with pytest.raises(RuntimeError, match="sidecar digest"):
        phase_d._evaluation_rows(
            root,
            "R1_s42",
            4,
            expected_signatures={
                benchmark: "c" * 64 for benchmark in phase_d.PHASE_D_EVAL_BENCHMARKS
            },
        )


def test_phase_d_budget_gate_counts_prior_spend_and_reserves_one_function(tmp_path):
    mock_root = tmp_path / "mock"
    resume_root = tmp_path / "resume"
    (mock_root / "modal_sessions").mkdir(parents=True)
    (resume_root / "modal_sessions").mkdir(parents=True)
    (mock_root / "modal_sessions" / "normal.json").write_text('{"approx_cost_usd": 1.25}', encoding="utf-8")
    (resume_root / "modal_sessions" / "resume.json").write_text('{"approx_cost_usd": 0.75}', encoding="utf-8")

    spent = _phase_d_cost_so_far(mock_root, resume_root)

    assert spent == pytest.approx(PROJECT_SPEND_BEFORE_PROFILE_SWITCH_USD + 2.0)
    _check_phase_d_budget(spent, reserve_seconds=1)
    with pytest.raises(RuntimeError, match="would exceed"):
        _check_phase_d_budget(PHASE_D_STOP_USD, reserve_seconds=1)
    assert MOCK_TIMEOUT_SECONDS == 27 * 60
    assert INTERRUPT_DEADLINE_SECONDS == 9 * 60
    # Include the two real startup attempts already charged before training began.
    assert PROJECT_SPEND_BEFORE_PROFILE_SWITCH_USD + 0.5635 + 3 * MOCK_TIMEOUT_SECONDS * 0.001736 < PHASE_D_STOP_USD
