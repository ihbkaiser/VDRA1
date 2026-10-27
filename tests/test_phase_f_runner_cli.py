import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_app.phase_d_budget import (
    MODAL_PROFILE_ADDITIONAL_CAP_USD,
    MODAL_PROFILE_BASELINE_METERED_USD,
)


ROOT = Path(__file__).resolve().parents[1]


def test_local_additional_spend_cap_matches_latest_approval():
    assert MODAL_PROFILE_ADDITIONAL_CAP_USD == 27.0
    assert MODAL_PROFILE_BASELINE_METERED_USD == 37.95116494


@pytest.mark.parametrize(
    "script_name",
    [
        "run_phase_f_remaining_smoke_with_budget.py",
        "run_phase_f_remaining_with_budget.py",
    ],
)
def test_phase_f_help_exits_before_budget_check_or_modal_launch(monkeypatch, script_name):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"phase_f_runner_{script_name[:-3]}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def unexpected_call(*args, **kwargs):
        pytest.fail("--help must not check billing or launch Modal")

    monkeypatch.setattr(module, "check_live_phase_d_budget", unexpected_call)
    monkeypatch.setattr(module.subprocess, "run", unexpected_call)

    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])

    assert exc.value.code == 0


def test_phase_f_remaining_runner_reserves_fresh_r2_and_survives_disconnect(monkeypatch):
    script_path = ROOT / "scripts" / "run_phase_f_remaining_smoke_with_budget.py"
    spec = importlib.util.spec_from_file_location("phase_f_recovery_runner", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reserves = []
    commands = []
    monkeypatch.setattr(
        module,
        "check_live_phase_d_budget",
        lambda reserve_seconds: reserves.append(reserve_seconds) or 70.0,
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs))
        or SimpleNamespace(returncode=0),
    )

    result = module.main(["--run-id", "fresh123", "--baseline-run-id", "1629e2315f"])

    assert result == 0
    assert reserves == [55 * 60]
    command = commands[0][0]
    assert "--detach" in command
    assert command[command.index("--run-id") + 1] == "fresh123"
    assert command[command.index("--baseline-run-id") + 1] == "1629e2315f"


def test_phase_f_r3_r6_only_runner_reserves_45_minutes_and_marks_scope(monkeypatch):
    script_path = ROOT / "scripts" / "run_phase_f_remaining_smoke_with_budget.py"
    spec = importlib.util.spec_from_file_location("phase_f_r3_r6_runner", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reserves = []
    commands = []
    monkeypatch.setattr(
        module,
        "check_live_phase_d_budget",
        lambda reserve_seconds: reserves.append(reserve_seconds) or 70.0,
    )
    monkeypatch.setattr(
        module,
        "subprocess",
        SimpleNamespace(run=lambda command, **kwargs: commands.append(command)
                        or SimpleNamespace(returncode=0)),
    )

    result = module.main([
        "--r3-r6-only", "--run-id", "r3r6only7", "--baseline-run-id", "1629e2315f"
    ])

    assert result == 0
    assert reserves == [45 * 60]
    assert "--r3-r6-only" in commands[0]
