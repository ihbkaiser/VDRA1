"""Run the pre-registered 8-prompt vLLM timing smoke on local MATH data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from .node_bank import VLLMClient


DEFAULT_TOKENIZER = "/workspace/rebuttal_out/tokenizer_iter100_fixed"
DEFAULT_CHECKPOINT = "/workspace/src/vdra_iter_0100/checkpoints/ckpt--iter_0100--epoch_1.00--step_0400/hf_pretrained"


QUESTION_TEMPLATE = (
    "<｜begin▁of▁sentence｜><｜User｜>Solve the following math problem efficiently and clearly.  "
    "The last line of your response should be of the following format: 'Therefore, the final answer is: "
    "$\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number "
    "or expression that solves the problem. Think step by step before answering.\n\n"
    "{query}<｜Assistant｜><think>\n"
)


def checkpoint_description(*, checkpoint_path: str, model: str) -> dict[str, str]:
    return {"checkpoint": str(checkpoint_path), "served_model": str(model)}


def _gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _monitor_memory(stop: threading.Event, samples: list[int]) -> None:
    while not stop.is_set():
        value = _gpu_memory_mib()
        if value is not None:
            samples.append(value)
        stop.wait(0.5)


def run_smoke(
    *, dataset_dir: str | Path, output: str | Path, base_url: str,
    seed: int = 42, prompts_total: int = 32, smoke_prompts: int = 8,
    samples_per_prompt: int = 6, max_tokens: int = 1024,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER,
    checkpoint_path: str = DEFAULT_CHECKPOINT, model: str = "iter100",
) -> dict[str, Any]:
    if prompts_total < smoke_prompts or smoke_prompts < 1:
        raise ValueError("prompt sample must contain at least the smoke subset")
    dataset = load_from_disk(str(dataset_dir))
    selected = random.Random(seed).sample(range(len(dataset)), prompts_total)
    records = []
    for ordinal, row_index in enumerate(selected):
        row = dataset[row_index]
        prompt = QUESTION_TEMPLATE.format(query=row["problem"])
        records.append({
            "ordinal": ordinal,
            "row_index": row_index,
            "prompt_id": str(row.get("unique_id", row_index)),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "request_seed": seed + ordinal,
        })

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_dir": str(dataset_dir),
        "dataset_rows": len(dataset),
        "selection": "Python random.Random(seed).sample without replacement",
        "seed": seed,
        "prompts_total": prompts_total,
        "smoke_prompts": smoke_prompts,
        "samples_per_prompt": samples_per_prompt,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "prompt_template": "gear/configs/prompt_library/qwen_MATH.jsonnet tree.question_template",
        "prompts": records,
    }
    manifest_path = out.with_name("controlled_prompt_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    client = VLLMClient(base_url, timeout=1200)
    memory_samples: list[int] = []
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor_memory, args=(stop, memory_samples), daemon=True)
    monitor.start()
    call_rows = []
    wall_started = time.perf_counter()
    try:
        for row in records[:smoke_prompts]:
            item = dataset[int(row["row_index"])]
            prompt = QUESTION_TEMPLATE.format(query=item["problem"])
            started = time.perf_counter()
            response = client.complete(
                prompt,
                n=samples_per_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=0.95,
                seed=int(row["request_seed"]),
            )
            seconds = time.perf_counter() - started
            usage = response.get("usage", {})
            call_rows.append({
                **row,
                "elapsed_seconds": seconds,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reasons": [choice.get("finish_reason") for choice in response.get("choices", [])],
            })
    finally:
        stop.set()
        monitor.join(timeout=3)
    wall_seconds = time.perf_counter() - wall_started
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in call_rows)
    result = {
        **checkpoint_description(checkpoint_path=checkpoint_path, model=model),
        "model": "Qwen2 causal LM from checkpoint config.json",
        "base_url": base_url,
        "gpu": "NVIDIA GeForce RTX 3090",
        "gpu_compute_capability": [8, 6],
        "precision": "bfloat16",
        "tokenizer_path": str(tokenizer_path),
        "fix_mistral_regex": True,
        "wall_seconds": wall_seconds,
        "requests_completed": len(call_rows),
        "completions_requested": len(call_rows) * samples_per_prompt,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / wall_seconds if wall_seconds else None,
        "peak_observed_gpu_memory_mib": max(memory_samples) if memory_samples else None,
        "memory_samples_mib": len(memory_samples),
        "prompt_manifest": str(manifest_path),
        "calls": call_rows,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts-total", type=int, default=32)
    parser.add_argument("--smoke-prompts", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default="iter100", help="vLLM served-model alias")
    args = parser.parse_args(argv)
    result = run_smoke(
        dataset_dir=args.dataset_dir,
        output=args.output,
        base_url=args.base_url,
        seed=args.seed,
        prompts_total=args.prompts_total,
        smoke_prompts=args.smoke_prompts,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens=args.max_tokens,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint_path,
        model=args.model,
    )
    print(json.dumps({key: result[key] for key in (
        "wall_seconds", "requests_completed", "completions_requested", "completion_tokens",
        "completion_tokens_per_second", "peak_observed_gpu_memory_mib", "prompt_manifest",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
