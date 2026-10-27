from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import eval_checkpoints  # noqa: E402
import eval_pending  # noqa: E402


def test_problem_limit_caps_each_benchmark_problem_list():
    problems = [{"id": i} for i in range(20)]

    assert eval_checkpoints.limit_problems(problems, 16) == problems[:16]
    assert eval_checkpoints.limit_problems(problems, None) == problems


def test_pending_evaluator_command_propagates_problem_limit():
    command = ["python", "eval_checkpoints.py"]

    assert eval_pending.add_eval_limit(command, 16) == command + ["--limit", "16"]
    assert eval_pending.add_eval_limit(command, None) == command


@pytest.mark.parametrize("limit", [0, -1])
def test_problem_limit_must_be_positive(limit):
    with pytest.raises(ValueError, match="positive"):
        eval_checkpoints.limit_problems([{"id": 1}], limit)


@pytest.mark.parametrize("limit", [0, -1])
def test_pending_limit_must_be_positive(limit):
    with pytest.raises(ValueError, match="positive"):
        eval_pending.add_eval_limit(["python", "eval_checkpoints.py"], limit)
