from pathlib import Path

import json
import pytest

from modal_app import phase_c
from modal_app.phase_c import (
    PHASE_F_RECOVERY_RUN_TIMEOUT_SECONDS,
    PHASE_F_TREEPO_R4_TIMEOUT_SECONDS,
    PHASE_F_TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT,
    _load_completed_phase_f_r2_tv,
    _phase_f_smokes_complete,
    _phase_f_audits_succeeded,
    _phase_f_treepo_smoke_function,
    phase_f_treepo_r4_smoke,
    phase_f_treepo_smoke,
    _training_paths,
)


def test_phase_f_r3_r6_smoke_gate_is_partial_and_checks_method_audits():
    results = {
        "null_R2_tv": {"status": "skipped"},
        "null_R3_spo": {"status": "complete"},
        "null_R4_tv": {"status": "complete"},
        "null_R5_spo": {"status": "complete"},
        "null_R6_tv": {"status": "complete"},
    }
    audits = {
        "null_R4_tv": {"status": "complete", "audit_exit_code": 0, "gate_G1_budget_ok": True},
        "null_R6_tv": {"status": "complete", "audit_exit_code": 0, "gate_G1_budget_ok": True},
    }

    gate = getattr(phase_c, "_phase_f_r3_r6_smokes_complete", None)
    assert callable(gate)
    assert gate(results, audits)

    results["null_R5_spo"]["status"] = "failed"
    assert not gate(results, audits)
    results["null_R5_spo"]["status"] = "complete"
    audits["null_R6_tv"]["gate_G1_budget_ok"] = False
    assert not gate(results, audits)


def test_r3_r6_only_smoke_does_not_require_igpo_null_baselines():
    baseline_dirs = [Path("/missing/null_R1"), Path("/missing/null_R2_constant")]
    missing_baselines = getattr(phase_c, "_phase_f_missing_baseline_dirs", None)
    assert callable(missing_baselines)
    assert missing_baselines(baseline_dirs, r3_r6_only=True) == []
    assert missing_baselines(baseline_dirs, r3_r6_only=False) == baseline_dirs


def test_training_paths_respect_run_specific_roots():
    paths = _training_paths(
        {
            "WORK": "/work",
            "RUNS_DIR": "/work/mock/runs",
            "RESULTS_DIR": "/work/mock/results",
            "BANKS_DIR": "/work/mock/banks",
        }
    )

    assert paths == (
        Path("/work"),
        Path("/work/mock/runs"),
        Path("/work/mock/results"),
        Path("/work/mock/banks"),
    )


def _write_completed_r2_tv_bundle(root):
    report_dir = root / "results"
    run_dir = root / "null_R2_tv" / "runs" / "R2_s42"
    report_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoints" / "ckpt--iter_0002--epoch_1.00--step_0004"
    pretrained = checkpoint / "hf_pretrained"
    pretrained.mkdir(parents=True)
    (pretrained / "config.json").write_text("{}", encoding="utf-8")
    (pretrained / "model.safetensors").write_bytes(b"weights")
    events = [
        {
            "event": "start",
            "run": "R2",
            "seed": 42,
            "stop": 2,
            "chunk_stop": 2,
            "attempt": 1,
        },
        {"event": "end", "rc": 0},
        {
            "event": "modal_container_end",
            "container_call_id": "r2-tv-call",
            "status": "complete",
            "error": None,
        },
    ]
    (run_dir / "run_meta.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (run_dir / "training_timing.jsonl").write_text(
        json.dumps(
            {
                "iteration": 0,
                "global_step": 2,
                "timing/iter/episode_generation_seconds": 10.0,
                "timing/iter/training_step_seconds": 2.0,
                "timing/iter/train_total_seconds": 12.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "phase_f_null_remainder.json").write_text(
        json.dumps(
            {
                "null_R2_tv": {
                    "experiment": "R2",
                    "status": "complete",
                    "reused_saved_run_path": True,
                    "call_id": "r2-tv-call",
                },
                "audits": {
                    "null_R2_tv": {
                        "budget_evidence_complete": True,
                        "gate_G1_budget_ok": True,
                        "budget_violations": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return report_dir / "phase_f_null_remainder.json", run_dir


def test_completed_r2_tv_smoke_is_reused_only_with_passed_budget_evidence(tmp_path):
    root = tmp_path / "phase_f_null_test"
    report_path, _ = _write_completed_r2_tv_bundle(root)

    assert _load_completed_phase_f_r2_tv(root)["status"] == "complete"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["audits"]["null_R2_tv"]["budget_violations"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert _load_completed_phase_f_r2_tv(root) is None


def test_fresh_completed_r2_tv_run_does_not_need_reuse_marker(tmp_path):
    root = tmp_path / "phase_f_null_test"
    report_path, _ = _write_completed_r2_tv_bundle(root)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["null_R2_tv"]["reused_saved_run_path"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert _load_completed_phase_f_r2_tv(root)["status"] == "complete"


def test_completed_r2_tv_reuse_rejects_cancelled_training_with_later_completed_container(tmp_path):
    root = tmp_path / "phase_f_null_test"
    report_path, run_dir = _write_completed_r2_tv_bundle(root)
    run_meta = [
        {
            "event": "start",
            "run": "R2",
            "seed": 42,
            "stop": 2,
            "chunk_stop": 2,
            "attempt": 1,
        },
        {
            "event": "modal_container_end",
            "container_call_id": "training-call",
            "status": "failed",
            "error": "InputCancellation: Input was cancelled by user",
        },
        {
            "event": "modal_container_end",
            "container_call_id": "r2-tv-call",
            "status": "complete",
            "error": None,
        },
    ]
    (run_dir / "run_meta.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in run_meta), encoding="utf-8"
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["null_R2_tv"]["call_id"] = "r2-tv-call"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert _load_completed_phase_f_r2_tv(root) is None


def test_completed_r2_tv_reuse_rejects_checkpoint_without_model_weights(tmp_path):
    root = tmp_path / "phase_f_null_test"
    _, run_dir = _write_completed_r2_tv_bundle(root)
    checkpoint = run_dir / "checkpoints" / "ckpt--iter_0002--epoch_1.00--step_0004"
    (checkpoint / "hf_pretrained" / "model.safetensors").unlink()

    assert _load_completed_phase_f_r2_tv(root) is None


def test_r4_smoke_function_cap_allows_ray_startup_to_finish():
    assert PHASE_F_TREEPO_R4_TIMEOUT_SECONDS > int(PHASE_F_TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT)
    assert _phase_f_treepo_smoke_function("R3") is phase_f_treepo_smoke
    assert _phase_f_treepo_smoke_function("R4") is phase_f_treepo_r4_smoke
    with pytest.raises(ValueError):
        _phase_f_treepo_smoke_function("R5")


def test_r2_tv_recovery_timeout_covers_recorded_smoke_duration():
    assert PHASE_F_RECOVERY_RUN_TIMEOUT_SECONDS > 209.02


def test_phase_f_remainder_requires_every_expected_smoke_label_to_complete():
    labels = ("null_R2_tv", "null_R3_spo", "null_R4_tv", "null_R5_spo", "null_R6_tv")
    complete = {label: {"status": "complete"} for label in labels}
    assert _phase_f_smokes_complete(complete, labels)

    failed = dict(complete)
    failed["null_R3_spo"] = {"status": "failed", "error": "timeout"}
    assert not _phase_f_smokes_complete(failed, labels)

    missing = dict(complete)
    del missing["null_R5_spo"]
    assert not _phase_f_smokes_complete(missing, labels)


def test_phase_f_gate_rejects_a_failed_audit_with_stale_passing_fields():
    assert _phase_f_audits_succeeded([{"pass": True, "gate_G1_budget_ok": True}])
    assert not _phase_f_audits_succeeded([
        {"status": "failed", "pass": True, "gate_G1_budget_ok": True}
    ])
    assert not _phase_f_audits_succeeded([
        {"pass": True, "gate_G1_budget_ok": True, "audit_exit_code": 1}
    ])


def test_phase_f_audit_failure_retains_the_written_gate_report(tmp_path):
    from modal_app.phase_c import _run_phase_f_audit
    import sys

    report = tmp_path / "audit.json"
    script = tmp_path / "write_failed_audit.py"
    script.write_text(
        "import json, sys; json.dump({'pass': False, 'budget_violations': 2}, open(sys.argv[1], 'w')); sys.exit(1)\n",
        encoding="utf-8",
    )

    result = _run_phase_f_audit([sys.executable, str(script), str(report)], cwd=str(tmp_path))

    assert result["status"] == "failed"
    assert result["pass"] is False
    assert result["budget_violations"] == 2
    assert "CalledProcessError" in result["execution_error"]
