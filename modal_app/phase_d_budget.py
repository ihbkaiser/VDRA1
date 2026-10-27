"""Fail-closed Modal billing gate shared by the Phase D runner and launcher."""

from __future__ import annotations

import json
import math
import subprocess
from typing import Any


MODAL_PROFILE = "billionaireproject8"
PHASE_D_STOP_USD = 48.0
PHASE_D_LIVE_HARD_STOP_FRACTION = 0.95
PHASE_D_LIVE_HARD_STOP_USD = PHASE_D_STOP_USD * PHASE_D_LIVE_HARD_STOP_FRACTION
MODAL_PROFILE_BASELINE_METERED_USD = 37.95116494
MODAL_PROFILE_ADDITIONAL_CAP_USD = 27.0
MODAL_PROFILE_ADDITIONAL_HARD_STOP_USD = (
    MODAL_PROFILE_BASELINE_METERED_USD
    + MODAL_PROFILE_ADDITIONAL_CAP_USD * PHASE_D_LIVE_HARD_STOP_FRACTION
)
PHASE_D_NON_GPU_RESERVE_USD = 0.25
GPU_RATE_USD_PER_SECOND = 0.001736
MOCK_TIMEOUT_SECONDS = 27 * 60
IGPO_PAIR_TIMEOUT_SECONDS = 80 * 60
IGPO_PROFILE_TIMEOUT_SECONDS = 10 * 60
IGPO_R1_REFERENCE_TIMEOUT_SECONDS = 15 * 60


def _live_metered_cost_usd() -> float:
    try:
        completed = subprocess.run(
            [
                "modal",
                "billing",
                "summary",
                "--json",
                "--profile",
                MODAL_PROFILE,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        summary: dict[str, Any] = json.loads(completed.stdout)
        metered_cost = float(summary["metered_cost"])
    except (
        OSError,
        subprocess.CalledProcessError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            "Live Modal billing could not be read; refusing to start a budgeted Modal job"
        ) from exc

    if not math.isfinite(metered_cost) or metered_cost < 0:
        raise RuntimeError(f"Modal returned an invalid metered cost: {metered_cost!r}")
    return metered_cost


def check_live_phase_d_budget(
    reserve_seconds: int = MOCK_TIMEOUT_SECONDS,
    *,
    reserve_usd: float | None = None,
) -> float:
    """Return metered spend if the reserved job fits within the profile's added-spend cap."""
    if reserve_usd is None:
        if reserve_seconds <= 0:
            raise ValueError("reserve_seconds must be positive")
        job_reserve = reserve_seconds * GPU_RATE_USD_PER_SECOND
    else:
        job_reserve = float(reserve_usd)
        if not math.isfinite(job_reserve) or job_reserve <= 0:
            raise ValueError("reserve_usd must be a positive finite number")

    metered_cost = _live_metered_cost_usd()
    projected = metered_cost + job_reserve + PHASE_D_NON_GPU_RESERVE_USD
    if projected > MODAL_PROFILE_ADDITIONAL_HARD_STOP_USD:
        current_additional = max(0.0, metered_cost - MODAL_PROFILE_BASELINE_METERED_USD)
        projected_additional = max(0.0, projected - MODAL_PROFILE_BASELINE_METERED_USD)
        raise RuntimeError(
            f"Live Modal budget gate for profile {MODAL_PROFILE}: reports ${metered_cost:.2f} "
            f"metered this month (${current_additional:.2f} since the "
            f"${MODAL_PROFILE_BASELINE_METERED_USD:.2f} starting meter); "
            f"${job_reserve:.2f} job reserve plus ${PHASE_D_NON_GPU_RESERVE_USD:.2f} "
            f"reserve projects ${projected_additional:.2f} additional usage, above the "
            f"${MODAL_PROFILE_ADDITIONAL_CAP_USD * PHASE_D_LIVE_HARD_STOP_FRACTION:.2f} "
            f"local hard stop for the ${MODAL_PROFILE_ADDITIONAL_CAP_USD:.2f} additional cap"
        )
    return metered_cost
