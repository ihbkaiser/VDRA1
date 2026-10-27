"""Budget-gate Phase F R3-R6 smokes, reusing verified R2-TV evidence."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
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


BASELINE_RUN_ID = "1629e2315f"
R2_RECOVERY_TIMEOUT_SECONDS = 10 * 60
TREEPO_TIMEOUT_SECONDS = 10 * 60
TREEPO_R4_TIMEOUT_SECONDS = 15 * 60
TREERL_TIMEOUT_SECONDS = 10 * 60


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="fresh alphanumeric Phase F run tag")
    parser.add_argument(
        "--baseline-run-id",
        default=BASELINE_RUN_ID,
        help="existing run tag holding the R1 and constant-R2 null baselines",
    )
    args = parser.parse_args(argv)
    run_id = args.run_id or f"phasef{uuid.uuid4().hex[:8]}"
    if not run_id.isalnum() or not args.baseline_run_id.isalnum():
        parser.error("run ids must contain only letters and digits")
    # Recover R2-TV in an isolated run root, then smoke R3-R6 under B200 caps.
    reserve_seconds = (
        R2_RECOVERY_TIMEOUT_SECONDS
        + TREEPO_TIMEOUT_SECONDS
        + TREEPO_R4_TIMEOUT_SECONDS
        + 2 * TREERL_TIMEOUT_SECONDS
    )
    try:
        metered_cost = check_live_phase_d_budget(reserve_seconds=reserve_seconds)
    except RuntimeError as exc:
        print(f"Phase F remaining smoke blocked before Modal launch: {exc}", file=sys.stderr)
        return 2

    reserve_usd = reserve_seconds * GPU_RATE_USD_PER_SECOND + PHASE_D_NON_GPU_RESERVE_USD
    print(
        f"Phase F remaining-smoke budget gate passed: ${metered_cost:.4f} metered + "
        f"${reserve_usd:.4f} reserved; fresh R2-TV recovery followed by R3-R6 serial smokes "
        f"(10-minute caps for R2/R3/R5/R6, 15-minute R4 cap; run {run_id}, "
        f"null baselines {args.baseline_run_id}; profile {MODAL_PROFILE})",
        flush=True,
    )
    command = [
        "modal", "run", "--timestamps", "--detach", "--profile", MODAL_PROFILE,
        "modal_app/phase_c.py", "--action", "phase_f_remaining_smoke",
        "--run-id", run_id, "--baseline-run-id", args.baseline_run_id,
    ]
    modal_env = os.environ.copy()
    modal_env["PYTHONIOENCODING"] = "utf-8"
    modal_env["PYTHONUTF8"] = "1"
    return subprocess.run(command, cwd=ROOT, check=False, env=modal_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
