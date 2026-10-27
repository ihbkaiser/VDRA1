"""Budget-gate a minimal R4/R6 Phase F retry and audit saved runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_app.phase_d_budget import (  # noqa: E402
    GPU_RATE_USD_PER_SECOND,
    MODAL_PROFILE,
    PHASE_D_NON_GPU_RESERVE_USD,
    check_live_phase_d_budget,
)


RUN_ID = "1629e2315f"
R4_TIMEOUT_SECONDS = 15 * 60
R6_TIMEOUT_SECONDS = 10 * 60


def main() -> int:
    reserve_seconds = R4_TIMEOUT_SECONDS + R6_TIMEOUT_SECONDS
    try:
        metered_cost = check_live_phase_d_budget(reserve_seconds=reserve_seconds)
    except RuntimeError as exc:
        print(f"Phase F final-method smoke blocked before Modal launch: {exc}", file=sys.stderr)
        return 2

    reserve_usd = reserve_seconds * GPU_RATE_USD_PER_SECOND + PHASE_D_NON_GPU_RESERVE_USD
    print(
        f"Phase F final-method budget gate passed: ${metered_cost:.4f} metered + "
        f"${reserve_usd:.4f} reserved; one-update R4/R6 smokes capped at 15/10 minutes "
        f"(profile {MODAL_PROFILE})",
        flush=True,
    )
    command = [
        "modal", "run", "--timestamps", "--detach", "--profile", MODAL_PROFILE,
        "modal_app/phase_c.py", "--action", "phase_f_final_methods_smoke", "--run-id", RUN_ID,
    ]
    modal_env = os.environ.copy()
    modal_env["PYTHONIOENCODING"] = "utf-8"
    modal_env["PYTHONUTF8"] = "1"
    return subprocess.run(command, cwd=ROOT, check=False, env=modal_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
