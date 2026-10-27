import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.budget_audit import audit


def test_empty_budget_audit_does_not_pass_without_queue_evidence(tmp_path):
    report = audit(str(tmp_path))

    assert report["n_queues"] == 0
    assert report["gate_G1_budget_ok"] is False
    assert report["gate_G1_headroom_ok"] is None
    json.dumps(report, allow_nan=False)


def test_budget_audit_passes_when_observed_queue_spends_its_budget(tmp_path):
    (tmp_path / "vdra_queues.jsonl").write_text(
        json.dumps({"budget": 3, "n_v": [1, 2], "capped": [False, False], "waited_s": 0.2}) + "\n",
        encoding="utf-8",
    )

    report = audit(str(tmp_path))

    assert report["n_queues"] == 1
    assert report["budget_violations"] == 0
    assert report["gate_G1_budget_ok"] is True
    assert report["gate_G1_headroom_ok"] is True
    json.dumps(report, allow_nan=False)


def test_budget_audit_rejects_a_queue_that_does_not_spend_its_budget(tmp_path):
    (tmp_path / "vdra_queues.jsonl").write_text(
        json.dumps({"budget": 4, "n_v": [1, 2], "capped": [False, False], "waited_s": 0.2}) + "\n",
        encoding="utf-8",
    )

    report = audit(str(tmp_path))

    assert report["budget_violations"] == 1
    assert report["gate_G1_budget_ok"] is False


def test_budget_audit_does_not_pass_when_queue_budget_is_missing(tmp_path):
    (tmp_path / "vdra_queues.jsonl").write_text(
        json.dumps({"n_v": [1, 2], "capped": [False, False], "waited_s": 0.2}) + "\n",
        encoding="utf-8",
    )

    report = audit(str(tmp_path))

    assert report["budget_violations"] == 0
    assert report["gate_G1_budget_ok"] is False


def test_cli_can_require_queue_budget_evidence(tmp_path):
    output = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "analysis/budget_audit.py"
    result = subprocess.run(
        [sys.executable, str(script), "--logs", str(tmp_path), "--out", str(output), "--require-budget-evidence"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["gate_G1_budget_ok"] is False


def test_all_capped_queue_can_leave_budget_unspent(tmp_path):
    (tmp_path / "vdra_queues.jsonl").write_text(
        json.dumps({"budget": 5, "n_v": [2, 2], "capped": [True, True], "waited_s": 0.2}) + "\n",
        encoding="utf-8",
    )

    report = audit(str(tmp_path))

    assert report["budget_violations"] == 0
    assert report["gate_G1_budget_ok"] is True


def test_all_capped_queue_still_fails_on_budget_overspend(tmp_path):
    (tmp_path / "vdra_queues.jsonl").write_text(
        json.dumps({"budget": 3, "n_v": [2, 2], "capped": [True, True], "waited_s": 0.2}) + "\n",
        encoding="utf-8",
    )

    report = audit(str(tmp_path))

    assert report["budget_violations"] == 1
    assert report["gate_G1_budget_ok"] is False


def test_budget_audit_rejects_a_capped_vector_with_the_wrong_length(tmp_path):
    (tmp_path / "vdra_queues.jsonl").write_text(
        json.dumps({"budget": 3, "n_v": [1, 2], "capped": [True], "waited_s": 0.2}) + "\n",
        encoding="utf-8",
    )

    report = audit(str(tmp_path))

    assert report["budget_evidence_complete"] is False
    assert report["gate_G1_budget_ok"] is False


def test_budget_audit_rejects_non_boolean_capped_flags(tmp_path):
    (tmp_path / "vdra_queues.jsonl").write_text(
        json.dumps({"budget": 3, "n_v": [1, 1], "capped": ["false", "false"]}) + "\n",
        encoding="utf-8",
    )

    report = audit(str(tmp_path))

    assert report["budget_evidence_complete"] is False
    assert report["budget_violations"] == 1
    assert report["gate_G1_budget_ok"] is False
    assert report["share_capped"] is None


@pytest.mark.parametrize(
    ("budget", "n_v", "expected_budget_violations"),
    [
        (3, [1.5, 1.5], 1),
        (3, [-1, 4], 1),
        (3, [True, 2], 1),
        ("3", [1, 2], 0),
        (3, ["1", "2"], 1),
    ],
)
def test_budget_audit_rejects_non_integer_or_negative_queue_counts(
    tmp_path, budget, n_v, expected_budget_violations
):
    (tmp_path / "vdra_queues.jsonl").write_text(
        json.dumps({"budget": budget, "n_v": n_v, "capped": [False] * len(n_v)}) + "\n",
        encoding="utf-8",
    )

    report = audit(str(tmp_path))

    assert report["budget_evidence_complete"] is False
    assert report["budget_violations"] == expected_budget_violations
    assert report["gate_G1_budget_ok"] is False
