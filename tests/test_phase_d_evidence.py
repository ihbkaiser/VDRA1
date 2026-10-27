import json

import pytest

import modal_app.phase_d as phase_d


def _event(update, rollout_count, generated_tokens, *, mean_reward=0.5, loss=0.1, grad_norm=0.2):
    return {
        "framework": "treepo",
        "update": update,
        "rollout_count": rollout_count,
        "generated_tokens": generated_tokens,
        "mean_reward": mean_reward,
        "loss": loss,
        "grad_norm": grad_norm,
    }


def _write_events(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_reference_artifact_saves_required_metrics_for_updates_one_and_two(tmp_path):
    assert hasattr(phase_d, "_build_reference_artifact"), "fixed-seed reference artifact writer is missing"
    metrics_path = tmp_path / "mock" / "R3_s42.metrics.jsonl"
    _write_events(metrics_path, [_event(1, 48, 4000), _event(2, 48, 4200), _event(3, 48, 4300)])

    report = phase_d._build_reference_artifact("R3", "treepo", metrics_path, tmp_path / "results")

    path = tmp_path / "results" / "equivalence" / "reference_R3_s42.json"
    assert path.is_file()
    assert report["updates"] == {"1": _event(1, 48, 4000), "2": _event(2, 48, 4200)}


def test_reference_artifact_fails_if_an_early_update_is_missing(tmp_path):
    assert hasattr(phase_d, "_build_reference_artifact"), "fixed-seed reference artifact writer is missing"
    metrics_path = tmp_path / "R3_s42.metrics.jsonl"
    _write_events(metrics_path, [_event(2, 48, 4200)])

    with pytest.raises(RuntimeError, match="missing updates \\[1\\]"):
        phase_d._build_reference_artifact("R3", "treepo", metrics_path, tmp_path / "results")


def test_resume_requires_rollout_and_token_counts_to_match_uninterrupted_run(tmp_path):
    assert hasattr(phase_d, "_compare_resume_metric_events"), "resume metrics comparison is missing"
    reference = tmp_path / "reference.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    _write_events(reference, [_event(1, 48, 4000), _event(2, 48, 4200), _event(3, 48, 4300), _event(4, 48, 4400)])
    _write_events(resumed, [_event(1, 48, 4000), _event(2, 48, 4200), _event(3, 48, 4300, loss=9.0), _event(4, 48, 4400, loss=8.0)])

    report = phase_d._compare_resume_metric_events(reference, resumed, "R3_s42")

    assert report["matched"] is True
    assert report["compared_updates"] == [3, 4]


def test_resume_fails_if_rollout_or_generated_token_counts_change(tmp_path):
    assert hasattr(phase_d, "_compare_resume_metric_events"), "resume metrics comparison is missing"
    reference = tmp_path / "reference.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    _write_events(reference, [_event(3, 48, 4300), _event(4, 48, 4400)])
    _write_events(resumed, [_event(3, 47, 4300), _event(4, 48, 4399)])

    with pytest.raises(RuntimeError, match="rollout/token counts differ"):
        phase_d._compare_resume_metric_events(reference, resumed, "R3_s42")


def test_interruption_pause_does_not_change_gpu_hours_at_saved_checkpoint():
    assert hasattr(phase_d, "_assert_gpu_hours_pause_excluded"), "interruption GPU-hour check is missing"
    phase_d._assert_gpu_hours_pause_excluded({"2": 0.125}, {"2": 0.125})

    with pytest.raises(RuntimeError, match="GPU-hours changed during interruption pause"):
        phase_d._assert_gpu_hours_pause_excluded({"2": 0.125}, {"2": 0.126})


def test_gpu_memory_parser_reads_used_and_total_mib():
    assert hasattr(phase_d, "_parse_gpu_memory_output"), "TreeRL GPU memory parser is missing"
    assert phase_d._parse_gpu_memory_output("100, 180000\n") == {"used_mib": 100, "total_mib": 180000}


def test_gpu_memory_parser_rejects_empty_or_invalid_output():
    assert hasattr(phase_d, "_parse_gpu_memory_output"), "TreeRL GPU memory parser is missing"
    with pytest.raises(RuntimeError, match="could not parse nvidia-smi memory output"):
        phase_d._parse_gpu_memory_output("memory.used, memory.total\n")


def test_checkpoint_scanner_reports_steps_for_each_framework(tmp_path):
    assert hasattr(phase_d, "_checkpoint_updates"), "Phase D checkpoint scanner is missing"
    layouts = {
        "treetune": ("checkpoints/ckpt--iter_2", "checkpoints/ckpt--iter_4"),
        "verl": ("ckpts/global_step_2", "ckpts/global_step_4"),
        "openrlhf": ("ckpt/_actor_resume/update2", "ckpt/_actor_resume/update4"),
    }
    for framework, paths in layouts.items():
        run_dir = tmp_path / framework
        for relative in paths:
            (run_dir / relative).mkdir(parents=True)
        assert phase_d._checkpoint_updates(run_dir, framework) == [2, 4]
