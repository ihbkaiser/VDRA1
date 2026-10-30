"""Wrapper for the source repository's evaluator with explicit sample count."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence


def build_command(
    script: str | Path,
    *,
    config: str | Path,
    model: str,
    datasets: Sequence[str],
    output_dir: str | Path,
    samples: int = 16,
    seed: int = 0,
    generation_overlay: str | Path | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not datasets:
        raise ValueError("at least one evaluation dataset is required")
    command = ["bash", str(script), str(config), model]
    if generation_overlay is not None:
        command += ["--config", str(generation_overlay)]
    command += [
        "--datasets", ",".join(datasets),
        "--metrics", "once_hit,correct_frac",
        "--num-samples", str(samples),
        *extra_args,
    ]
    # The source evaluator reads these settings from its environment.
    command.extend(["--", f"APP_DIRECTORY={output_dir}", f"APP_EXPERIMENT_NAME=avgk{samples}", f"APP_SEED={seed}"])
    return command


def run(command: Sequence[str], *, dry_run: bool = False) -> int:
    if dry_run:
        print(" ".join(command))
        return 0
    env = None
    if "--" in command:
        split = list(command).index("--")
        assignments = command[split + 1 :]
        command = command[:split]
        import os

        env = os.environ.copy()
        for assignment in assignments:
            key, value = assignment.split("=", 1)
            env[key] = value
    return subprocess.run(list(command), check=False, env=env).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", default="aime24,aime25,amc23")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-overlay")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = build_command(
        args.script, config=args.config, model=args.model, datasets=args.datasets.split(","),
        output_dir=args.output_dir, samples=args.samples, seed=args.seed,
        generation_overlay=args.generation_overlay, extra_args=args.extra_args,
    )
    return run(command, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
