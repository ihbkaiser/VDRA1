import importlib.util
import json
import math


_METRIC_CAPTURE_SPEC = importlib.util.find_spec("modal_app.metric_capture")
metric_capture = importlib.import_module("modal_app.metric_capture") if _METRIC_CAPTURE_SPEC else None


def test_metric_capture_appends_canonical_update_metrics(tmp_path, monkeypatch):
    assert metric_capture is not None, "Phase D metric capture module is missing"
    path = tmp_path / "R1_s42.metrics.jsonl"
    monkeypatch.setenv("VDRA_METRICS_JSONL", str(path))

    wrote = metric_capture.write_training_metrics(
        "treetune",
        1,
        rollout_count=8,
        generated_tokens=123,
        mean_reward=0.5,
        loss=0.125,
        grad_norm=0.75,
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert wrote is True
    assert row == {
        "framework": "treetune",
        "update": 1,
        "rollout_count": 8,
        "generated_tokens": 123,
        "mean_reward": 0.5,
        "loss": 0.125,
        "grad_norm": 0.75,
    }


def test_metric_capture_is_a_noop_without_an_output_path(monkeypatch):
    assert metric_capture is not None, "Phase D metric capture module is missing"
    monkeypatch.delenv("VDRA_METRICS_JSONL", raising=False)

    assert metric_capture.write_training_metrics(
        "treetune", 1, rollout_count=8, generated_tokens=123,
        mean_reward=0.5, loss=0.125, grad_norm=0.75,
    ) is False


def test_metric_capture_rejects_non_finite_values(tmp_path, monkeypatch):
    assert metric_capture is not None, "Phase D metric capture module is missing"
    monkeypatch.setenv("VDRA_METRICS_JSONL", str(tmp_path / "metrics.jsonl"))

    try:
        metric_capture.write_training_metrics(
            "treetune", 1, rollout_count=8, generated_tokens=123,
            mean_reward=math.nan, loss=0.125, grad_norm=0.75,
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite metrics must be rejected")
