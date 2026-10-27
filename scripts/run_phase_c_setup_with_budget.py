#!/usr/bin/env python3
"""Run the CPU-only Modal data setup after reserving its worst-case cost."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_app.phase_d_budget import (  # noqa: E402
    MODAL_PROFILE_ADDITIONAL_CAP_USD,
    MODAL_PROFILE_ADDITIONAL_HARD_STOP_USD,
    MODAL_PROFILE_BASELINE_METERED_USD,
    MODAL_PROFILE,
    PHASE_D_NON_GPU_RESERVE_USD,
    check_live_phase_d_budget,
)


# setup_data() requests 8 CPU cores and 32 GiB for at most four hours. At
# Modal's current CPU/RAM rates that is $2.532096; the $3.10 job reserve adds
# room for image preparation and network transfer, before the shared $0.25
# non-GPU reserve.
SETUP_DATA_JOB_RESERVE_USD = 3.10


def main() -> int:
    # Modal emits Unicode status glyphs; Windows may otherwise default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        metered_cost = check_live_phase_d_budget(reserve_usd=SETUP_DATA_JOB_RESERVE_USD)
    except RuntimeError as exc:
        print(f"Phase C setup blocked before image build: {exc}", file=sys.stderr)
        return 2

    run_suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "remote_artifacts" / f"modal_phase_c_setup_{run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preflight.json").write_text(
        json.dumps(
            {
                "profile": MODAL_PROFILE,
                "metered_cost_usd": metered_cost,
                "starting_metered_cost_usd": MODAL_PROFILE_BASELINE_METERED_USD,
                "additional_cap_usd": MODAL_PROFILE_ADDITIONAL_CAP_USD,
                "additional_hard_stop_usd": MODAL_PROFILE_ADDITIONAL_HARD_STOP_USD,
                "job_reserve_usd": SETUP_DATA_JOB_RESERVE_USD,
                "non_gpu_reserve_usd": PHASE_D_NON_GPU_RESERVE_USD,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    command = [
        "modal",
        "run",
        "--timestamps",
        "--name",
        f"vdra-phase-c-setup-{run_suffix}",
        "--profile",
        MODAL_PROFILE,
        "modal_app/phase_c.py",
        "--action",
        "setup",
    ]
    modal_env = os.environ.copy()
    modal_env["PYTHONIOENCODING"] = "utf-8"
    modal_env["PYTHONUTF8"] = "1"
    print(
        f"Phase C budget preflight passed: ${metered_cost:.2f} metered + "
        f"${SETUP_DATA_JOB_RESERVE_USD + PHASE_D_NON_GPU_RESERVE_USD:.2f} reserved "
        f"(profile {MODAL_PROFILE})"
    )

    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=modal_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        exit_code = process.wait()

    (run_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    billing = subprocess.run(
        ["modal", "billing", "summary", "--json", "--profile", MODAL_PROFILE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=modal_env,
    )
    if billing.returncode == 0:
        (run_dir / "billing_summary.json").write_text(
            billing.stdout.rstrip() + "\n", encoding="utf-8"
        )
    else:
        (run_dir / "billing_summary_error.txt").write_text(
            billing.stderr.rstrip() + "\n", encoding="utf-8"
        )
    print(f"Phase C setup exit code: {exit_code}; artifacts: {run_dir.relative_to(ROOT)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
