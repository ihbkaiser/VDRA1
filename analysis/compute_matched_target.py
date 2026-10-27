#!/usr/bin/env python3
"""Choose an idempotent baseline update target for a GPU-hour match."""

from __future__ import annotations

import argparse
import json
import math


def compute_target(baseline: dict[int, float], hybrid: dict[int, float], stop: int, checkpoint_spacing: int) -> int:
    if stop not in hybrid:
        raise KeyError(f"hybrid GPU-hour record has no step {stop}")
    if not baseline or checkpoint_spacing < 1:
        raise ValueError("baseline hours and a positive checkpoint spacing are required")
    last = max(baseline)
    target_hours = float(hybrid[stop])
    current_hours = float(baseline[last])
    if current_hours >= target_hours:
        target = last
    else:
        per_step = current_hours / last if last else 0.0
        if per_step <= 0:
            raise ValueError("cannot estimate baseline cost per update from zero GPU-hours")
        extra = math.ceil((target_hours - current_hours) / per_step)
        target = last + max(extra, 2)
    return int(math.ceil(target / checkpoint_spacing) * checkpoint_spacing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--checkpoint-spacing", type=int, required=True)
    args = parser.parse_args()
    baseline = {int(k): float(v) for k, v in json.load(open(args.baseline, encoding="utf-8")).items()}
    hybrid = {int(k): float(v) for k, v in json.load(open(args.hybrid, encoding="utf-8")).items()}
    print(compute_target(baseline, hybrid, args.stop, args.checkpoint_spacing))


if __name__ == "__main__":
    main()
