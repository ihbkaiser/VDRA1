"""Check live Modal spend before image builds, then start the Phase D app."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_app.phase_d_budget import (  # noqa: E402
    GPU_RATE_USD_PER_SECOND,
    IGPO_PAIR_TIMEOUT_SECONDS,
    IGPO_PROFILE_TIMEOUT_SECONDS,
    IGPO_R1_REFERENCE_TIMEOUT_SECONDS,
    MOCK_TIMEOUT_SECONDS,
    MODAL_PROFILE,
    PHASE_D_NON_GPU_RESERVE_USD,
    check_live_phase_d_budget,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework",
        choices=("all", "igpo", "treepo", "treerl"),
        default="all",
    )
    parser.add_argument(
        "--igpo-max-num-seqs",
        type=int,
        default=64,
        help="IGPO vLLM sequence limit for an isolated Phase D pair (1-128; default 64)",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="alphanumeric tag for an isolated Phase D volume root; required for nondefault sequence limit",
    )
    parser.add_argument(
        "--igpo-config-preflight",
        action="store_true",
        help="run only the CPU IGPO Jsonnet preflight with a conservative budget reserve",
    )
    parser.add_argument(
        "--igpo-image-preflight",
        action="store_true",
        help="build the IGPO training image and run its CPU-only config preflight",
    )
    parser.add_argument(
        "--igpo-logprob-profile",
        action="store_true",
        help="run an isolated IGPO R1 logprob profile with a 10-minute B200 timeout",
    )
    parser.add_argument(
        "--igpo-r1-reference",
        action="store_true",
        help="train a clean IGPO R1 baseline chunk through update 2 and capture reference metrics",
    )
    args = parser.parse_args()

    if not 1 <= args.igpo_max_num_seqs <= 128:
        parser.error("--igpo-max-num-seqs must be between 1 and 128")
    if args.run_tag and not args.run_tag.isalnum():
        parser.error("--run-tag must be alphanumeric")
    igpo_tuning_requested = (
        args.run_tag
        or args.igpo_max_num_seqs != 64
    )
    if igpo_tuning_requested and args.framework != "igpo":
        parser.error("IGPO sequence tuning requires --framework igpo")
    if args.igpo_max_num_seqs != 64 and not args.run_tag:
        parser.error("nondefault IGPO sequence limit requires --run-tag to isolate artifacts")

    selected_modes = sum(
        (
            args.igpo_config_preflight,
            args.igpo_image_preflight,
            args.igpo_logprob_profile,
            args.igpo_r1_reference,
        )
    )
    if selected_modes > 1:
        parser.error("choose only one special Phase D mode")
    if selected_modes and igpo_tuning_requested and not (
        args.igpo_logprob_profile and bool(args.run_tag)
    ):
        parser.error(
            "IGPO sequence tuning with a special mode requires an isolated logprob profile run tag"
        )

    if args.igpo_config_preflight or args.igpo_image_preflight:
        reserve_seconds = 5 * 60
        modal_target = (
            "modal_app/phase_d.py"
            if args.igpo_image_preflight
            else "modal_app/igpo_preflight.py::preflight_igpo_eval_config"
        )
    elif args.igpo_logprob_profile:
        reserve_seconds = IGPO_PROFILE_TIMEOUT_SECONDS
        modal_target = "modal_app/phase_d.py"
    elif args.igpo_r1_reference:
        reserve_seconds = IGPO_R1_REFERENCE_TIMEOUT_SECONDS
        modal_target = "modal_app/phase_d.py"
    else:
        if args.framework == "all":
            reserve_seconds = IGPO_PAIR_TIMEOUT_SECONDS + 2 * MOCK_TIMEOUT_SECONDS
        elif args.framework == "igpo":
            reserve_seconds = IGPO_PAIR_TIMEOUT_SECONDS
        else:
            reserve_seconds = MOCK_TIMEOUT_SECONDS
        modal_target = "modal_app/phase_d.py"
    try:
        metered_cost = check_live_phase_d_budget(reserve_seconds=reserve_seconds)
    except RuntimeError as exc:
        print(f"Phase D launch blocked before image build: {exc}", file=sys.stderr)
        return 2

    reserved = reserve_seconds * GPU_RATE_USD_PER_SECOND + PHASE_D_NON_GPU_RESERVE_USD
    target_description = (
        "IGPO training image preflight"
        if args.igpo_image_preflight
        else "lightweight IGPO config preflight"
        if args.igpo_config_preflight
        else "IGPO R1 logprob profile"
        if args.igpo_logprob_profile
        else "clean IGPO R1 reference through update 2"
        if args.igpo_r1_reference
        else (
            f"IGPO B200 pair at max_num_seqs={args.igpo_max_num_seqs}, "
            "max_concurrent_generations=64 (fixed)"
            if args.framework == "igpo"
            else f"{args.framework} B200 run"
        )
    )
    print(
        f"Phase D budget preflight passed for {target_description}: "
        f"${metered_cost:.2f} metered + ${reserved:.2f} reserved "
        f"(profile {MODAL_PROFILE})"
    )
    run_suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    command = build_modal_command(args, modal_target, run_suffix)
    modal_env = os.environ.copy()
    modal_env["PYTHONIOENCODING"] = "utf-8"
    modal_env["PYTHONUTF8"] = "1"
    completed = subprocess.run(command, cwd=ROOT, check=False, env=modal_env)
    return completed.returncode


def build_modal_command(args: argparse.Namespace, modal_target: str, run_suffix: str) -> list[str]:
    if args.igpo_config_preflight:
        run_label = "vdra-igpo-config-preflight"
    elif args.igpo_image_preflight:
        run_label = "vdra-igpo-image-preflight"
    elif args.igpo_logprob_profile:
        run_label = "vdra-igpo-logprob-profile"
        if args.run_tag:
            run_label += f"-{args.run_tag}"
    elif args.igpo_r1_reference:
        run_label = "vdra-igpo-r1-reference"
    else:
        run_label = f"vdra-phase-d-{args.framework}"
        if args.run_tag:
            run_label += f"-{args.run_tag}"

    command = [
        "modal",
        "run",
        "--timestamps",
        "--detach",
        "--name",
        f"{run_label}-{run_suffix}",
        "--profile",
        MODAL_PROFILE,
        modal_target,
    ]
    if args.igpo_image_preflight:
        command.append("--igpo-image-preflight")
    elif args.igpo_logprob_profile:
        command.extend(
            [
                "--igpo-logprob-profile",
                "--framework",
                "igpo",
                "--igpo-max-num-seqs",
                str(args.igpo_max_num_seqs),
            ]
        )
        if args.run_tag:
            command.extend(["--run-tag", args.run_tag])
    elif args.igpo_r1_reference:
        command.append("--igpo-r1-reference")
    elif not args.igpo_config_preflight and not args.igpo_image_preflight:
        command.extend(["--framework", args.framework])
        if args.framework == "igpo":
            command.extend(["--igpo-max-num-seqs", str(args.igpo_max_num_seqs)])
            if args.run_tag:
                command.extend(["--run-tag", args.run_tag])
    return command


if __name__ == "__main__":
    raise SystemExit(main())
