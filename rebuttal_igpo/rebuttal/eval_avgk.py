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
    tokenizer: str | Path | None = None,
    context_length: int | None = None,
    max_new_tokens: int | None = None,
    generation_overlay: str | Path | None = None,
    server_overlay: str | Path | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not datasets:
        raise ValueError("at least one evaluation dataset is required")
    command = ["bash", str(script), str(config), model]
    if tokenizer is not None:
        command += ["--tokenizer", str(tokenizer)]
    if context_length is not None:
        command += ["--context-length", str(context_length)]
    if max_new_tokens is not None:
        command += ["--max-new-tokens", str(max_new_tokens)]
    if generation_overlay is not None:
        command += ["--config", str(generation_overlay)]
    if server_overlay is not None:
        command += ["--config", str(server_overlay)]
    command += [
        "--datasets", ",".join(datasets),
        "--num-samples", str(samples),
        *extra_args,
    ]
    # The source evaluator reads these settings from its environment.
    assignments = [
        f"APP_DIRECTORY={output_dir}",
        f"APP_EXPERIMENT_NAME=avgk{samples}",
        f"APP_SEED={seed}",
        # evaluate.sh defaults to minimizing files, which moves result datasets
        # to temporary storage. Keep per-prompt answers so A7 can bootstrap them.
        "APP_MINIMIZE_STORED_FILES=False",
        # The evaluator calls wandb.save() with the raw inference dataset.
        # Disabled mode makes that call a no-op with respect to remote services.
        "WANDB_MODE=disabled",
        "WANDB_DISABLE_UPLOAD_FILES_TO_CLOUD=True",
    ]
    if server_overlay is not None:
        server_script = Path(__file__).with_name("start_vllm_server_shared_tokenizer.sh").resolve()
        assignments.append(f"APP_VLLM_SHARED_TOKENIZER_SERVER={server_script}")
    command.extend([
        "--",
        *assignments,
    ])
    return command


def run(
    command: Sequence[str],
    *,
    dry_run: bool = False,
    working_directory: str | Path | None = None,
) -> int:
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
    return subprocess.run(
        list(command), check=False, env=env,
        cwd=str(working_directory) if working_directory is not None else None,
    ).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", default="aime24,aime25,amc23")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--generation-overlay")
    parser.add_argument("--server-overlay")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args, passthrough = parser.parse_known_args(argv)
    extra_args = [*args.extra_args, *passthrough]
    command = build_command(
        args.script, config=args.config, model=args.model, datasets=args.datasets.split(","),
        output_dir=args.output_dir, samples=args.samples, seed=args.seed,
        tokenizer=args.tokenizer, context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        generation_overlay=args.generation_overlay, server_overlay=args.server_overlay,
        extra_args=extra_args,
    )
    repository_root = Path(args.script).resolve().parent.parent
    return run(command, dry_run=args.dry_run, working_directory=repository_root)


if __name__ == "__main__":
    raise SystemExit(main())
