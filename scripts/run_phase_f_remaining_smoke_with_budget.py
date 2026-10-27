"""Budget-gate Phase F remaining or R3-R6-only smokes."""

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
R3_TIMEOUT_SECONDS = 10 * 60
R4_TIMEOUT_SECONDS = 15 * 60
R5_TIMEOUT_SECONDS = 10 * 60
R6_TIMEOUT_SECONDS = 10 * 60


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="fresh alphanumeric Phase F run tag")
    parser.add_argument(
        "--baseline-run-id",
        default=BASELINE_RUN_ID,
        help="existing run tag holding the R1 and constant-R2 null baselines",
    )
    parser.add_argument(
        "--r3-r6-only",
        action="store_true",
        help="smoke R3-R6 without the separate fresh R2-TV recovery; does not certify full Phase F",
    )
    args = parser.parse_args(argv)
    run_id = args.run_id or f"phasef{uuid.uuid4().hex[:8]}"
    if not run_id.isalnum() or not args.baseline_run_id.isalnum():
        parser.error("run ids must contain only letters and digits")
    reserve_seconds = (
        0 if args.r3_r6_only else R2_RECOVERY_TIMEOUT_SECONDS
    ) + (
        R3_TIMEOUT_SECONDS
        + R4_TIMEOUT_SECONDS
        + R5_TIMEOUT_SECONDS
        + R6_TIMEOUT_SECONDS
    )
    try:
        metered_cost = check_live_phase_d_budget(reserve_seconds=reserve_seconds)
    except RuntimeError as exc:
        print(f"Phase F remaining smoke blocked before Modal launch: {exc}", file=sys.stderr)
        return 2

    reserve_usd = reserve_seconds * GPU_RATE_USD_PER_SECOND + PHASE_D_NON_GPU_RESERVE_USD
    scope = (
        "R3-R6 only; R2-TV recovery and full Phase F certification are skipped"
        if args.r3_r6_only
        else "fresh R2-TV recovery followed by R3-R6"
    )
    caps = (
        "10-minute caps for R3/R5/R6, 15-minute R4 cap"
        if args.r3_r6_only
        else "10-minute caps for R2/R3/R5/R6, 15-minute R4 cap"
    )
    print(
        f"Phase F remaining-smoke budget gate passed: ${metered_cost:.4f} metered + "
        f"${reserve_usd:.4f} reserved; {scope} serial smokes "
        f"({caps}; run {run_id}, "
        f"null baselines {args.baseline_run_id}; profile {MODAL_PROFILE})",
        flush=True,
    )
    command = [
        "modal", "run", "--timestamps", "--detach", "--profile", MODAL_PROFILE,
        "modal_app/phase_c.py", "--action", "phase_f_remaining_smoke",
        "--run-id", run_id, "--baseline-run-id", args.baseline_run_id,
    ]
    if args.r3_r6_only:
        command.append("--r3-r6-only")
    modal_env = os.environ.copy()
    modal_env["PYTHONIOENCODING"] = "utf-8"
    modal_env["PYTHONUTF8"] = "1"
    return subprocess.run(command, cwd=ROOT, check=False, env=modal_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
