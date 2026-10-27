"""Compare prompt-token logprobs on the pinned legacy and B200 stacks.

This probe is independent of training data. It uses the model snapshot already
stored on ``vdra-work`` and the vLLM ``prompt_logprobs=0`` path used by IGPO's
scorer.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import modal

from modal_app.images import ROOT, _PROJECT_IGNORE, common_image
from modal_app.telemetry import append_jsonl, run_start_benchmark


APP_NAME = "vdra-logprob-compare"
VOLUME_NAME = "vdra-work"
WORK_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
MODEL_PATH = (
    "/work/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
    f"/snapshots/{MODEL_REVISION}"
)
PROMPTS = (
    "Compute 2 + 2.",
    "If $x^2 - 5x + 6=0$, find the roots.",
    "What is the period of $y = \\tan \\frac{x}{2}$?",
    "A rectangle has side lengths 3 and 4. What is its diagonal length?",
)
H100_PRICE_PER_SECOND = 0.001097
B200_PRICE_PER_SECOND = 0.001736
PROBE_TIMEOUT_SECONDS = 600

LEGACY_VLLM_WHEEL = (
    "https://github.com/McGill-NLP/VinePPO/releases/download/latest/"
    "vllm-0.4.0.post1-cp310-cp310-linux_x86_64.whl"
)
LEGACY_FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/"
    "flash_attn-2.5.5+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)
LEGACY_IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel")
    .pip_install(
        "numpy==1.26.4",
        "triton==2.1.0",
        "transformers==4.38.1",
        "tokenizers==0.15.2",
        "sentencepiece==0.2.0",
        "xformers==0.0.23.post1",
        LEGACY_FLASH_ATTN_WHEEL,
        LEGACY_VLLM_WHEEL,
        "modal==1.6.1",
        "dill==0.3.8",
    )
    .run_commands(
        "python -c 'import torch, vllm; assert torch.__version__.startswith(\"2.1.2\"); "
        "assert vllm.__version__.startswith(\"0.4.0\"); print(torch.__version__, vllm.__version__)'"
    )
    .add_local_dir(ROOT, "/opt/vdra", copy=True, ignore=_PROJECT_IGNORE)
    .env(
        {
            "PYTHONPATH": "/opt/vdra",
            "HF_HOME": "/work/hf_cache",
            "HF_HUB_CACHE": "/work/hf_cache/hub",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
)
CURRENT_IMAGE = common_image()
app = modal.App(APP_NAME)


def _tokenizer_hashes(model_path: str) -> dict[str, str]:
    hashes = {}
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
    ):
        path = Path(model_path) / filename
        if path.is_file():
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _extract_prompt_logprobs(output: Any, token_ids: list[int]) -> list[float | None]:
    values = getattr(output, "prompt_logprobs", None)
    if values is None:
        raise RuntimeError("vLLM returned no prompt_logprobs")
    if len(values) != len(token_ids):
        raise RuntimeError(
            f"vLLM returned {len(values)} prompt logprob positions for {len(token_ids)} tokens"
        )
    result: list[float | None] = []
    for token_id, candidates in zip(token_ids, values):
        if candidates is None:
            result.append(None)
            continue
        token_value = candidates.get(token_id)
        if token_value is None:
            result.append(None)
            continue
        logprob = token_value.get("logprob") if isinstance(token_value, dict) else token_value.logprob
        result.append(float(logprob))
    return result


def compare_runs(legacy: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Align per-token values and report the maximum absolute difference."""
    if legacy["model_revision"] != current["model_revision"]:
        raise ValueError("model revisions differ")
    if legacy["tokenizer_sha256"] != current["tokenizer_sha256"]:
        raise ValueError("tokenizer hashes differ")
    if len(legacy["prompts"]) != len(current["prompts"]):
        raise ValueError("prompt counts differ")

    prompt_summaries = []
    all_diffs = []
    for old, new in zip(legacy["prompts"], current["prompts"]):
        if old["text"] != new["text"]:
            raise ValueError("prompt texts differ")
        if old["token_ids"] != new["token_ids"]:
            raise ValueError(f"token IDs differ for prompt {old['text']!r}")
        old_values = old["token_logprobs"]
        new_values = new["token_logprobs"]
        if len(old_values) != len(new_values) or len(old_values) != len(old["token_ids"]):
            raise ValueError(f"logprob lengths differ for prompt {old['text']!r}")

        diffs = []
        for old_value, new_value in zip(old_values, new_values):
            if old_value is None and new_value is None:
                continue
            if old_value is None or new_value is None:
                raise ValueError(f"missing logprob on one stack for prompt {old['text']!r}")
            if not math.isfinite(float(old_value)) or not math.isfinite(float(new_value)):
                raise ValueError(f"non-finite logprob for prompt {old['text']!r}")
            diffs.append(abs(float(old_value) - float(new_value)))
        if not diffs:
            raise ValueError(f"no comparable token logprobs for prompt {old['text']!r}")
        max_diff = max(diffs)
        all_diffs.extend(diffs)
        prompt_summaries.append(
            {
                "text": old["text"],
                "token_count": len(old["token_ids"]),
                "compared_tokens": len(diffs),
                "max_abs_diff": max_diff,
            }
        )
    return {
        "model_revision": legacy["model_revision"],
        "tokenizer_sha256": legacy["tokenizer_sha256"],
        "legacy_stack": legacy["stack"],
        "current_stack": current["stack"],
        "prompts": prompt_summaries,
        "compared_tokens": len(all_diffs),
        "max_abs_per_token_diff": max(all_diffs),
    }


def _capture(stack: str, expected_capability: tuple[int, int], price_per_second: float) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    started_at = time.time()
    run_dir = Path(f"/work/probes/logprob_comparison/{stack}")
    run_meta_path = run_dir / "run_meta.jsonl"
    run_meta_dir = run_dir / "results.json"
    append_jsonl(run_meta_path, run_start_benchmark(expected_capability=expected_capability))

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, local_files_only=True)
        prompts = list(PROMPTS)
        token_ids = [tokenizer(prompt, add_special_tokens=True)["input_ids"] for prompt in prompts]
        llm = LLM(
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            dtype="bfloat16",
            max_model_len=128,
            gpu_memory_utilization=0.25,
            enforce_eager=True,
            tensor_parallel_size=1,
        )
        params = SamplingParams(
            n=1,
            temperature=1.0,
            max_tokens=1,
            prompt_logprobs=0,
        )
        outputs = llm.generate(prompts, params)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")

        rows = []
        for prompt, expected_ids, output in zip(prompts, token_ids, outputs):
            output_ids = list(output.prompt_token_ids)
            if output_ids != expected_ids:
                raise RuntimeError(f"vLLM tokenization differed from Transformers for {prompt!r}")
            rows.append(
                {
                    "text": prompt,
                    "token_ids": output_ids,
                    "token_logprobs": _extract_prompt_logprobs(output, output_ids),
                }
            )

        stack_info = {
            "name": stack,
            "torch": torch.__version__,
            "vllm": __import__("vllm").__version__,
            "transformers": __import__("transformers").__version__,
            "tokenizers": __import__("tokenizers").__version__,
        }
        result = {
            "stack": stack_info,
            "model_revision": MODEL_REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_sha256": _tokenizer_hashes(MODEL_PATH),
            "prompts": rows,
            "elapsed_seconds": time.time() - started_at,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        }
        result["approx_cost_usd"] = result["elapsed_seconds"] * price_per_second
        run_dir.mkdir(parents=True, exist_ok=True)
        run_meta_dir.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        append_jsonl(
            run_meta_path,
            {
                "event": "logprob_probe_end",
                "time": time.time(),
                "wall_seconds": result["elapsed_seconds"],
                "approx_cost_usd": result["approx_cost_usd"],
                "ok": True,
            },
        )
        WORK_VOLUME.commit()
        return result
    except Exception as exc:
        append_jsonl(
            run_meta_path,
            {
                "event": "logprob_probe_end",
                "time": time.time(),
                "wall_seconds": time.time() - started_at,
                "approx_cost_usd": (time.time() - started_at) * price_per_second,
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
            },
        )
        WORK_VOLUME.commit()
        raise


@app.function(
    image=LEGACY_IMAGE,
    gpu="H100",
    timeout=PROBE_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def legacy_h100() -> dict[str, Any]:
    return _capture("legacy_h100", (9, 0), H100_PRICE_PER_SECOND)


@app.function(
    image=CURRENT_IMAGE,
    gpu="B200",
    timeout=PROBE_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def current_b200() -> dict[str, Any]:
    return _capture("current_b200", (10, 0), B200_PRICE_PER_SECOND)


@app.function(
    image=CURRENT_IMAGE,
    gpu="H100",
    timeout=PROBE_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def current_h100() -> dict[str, Any]:
    return _capture("current_h100", (9, 0), H100_PRICE_PER_SECOND)


def _save_raw_result(stack: str, result: dict[str, Any]) -> Path:
    output = Path(__file__).resolve().parents[1] / f"codex/phase_b_results/{stack}_logprob.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _build_local_comparison() -> dict[str, Any]:
    result_dir = Path(__file__).resolve().parents[1] / "codex/phase_b_results"
    legacy = json.loads((result_dir / "legacy_h100_logprob.json").read_text(encoding="utf-8"))
    current_h100_result = json.loads((result_dir / "current_h100_logprob.json").read_text(encoding="utf-8"))
    current_b200_result = json.loads((result_dir / "current_b200_logprob.json").read_text(encoding="utf-8"))
    same_gpu = compare_runs(legacy, current_h100_result)
    architecture = compare_runs(current_h100_result, current_b200_result)
    combined = compare_runs(legacy, current_b200_result)
    return {
        "ok": True,
        "model_revision": legacy["model_revision"],
        "tokenizer_sha256": legacy["tokenizer_sha256"],
        "same_gpu_stack_comparison": same_gpu,
        "current_stack_gpu_comparison": architecture,
        "combined_old_stack_to_b200_comparison": combined,
        "max_abs_per_token_diff": same_gpu["max_abs_per_token_diff"],
        "cross_architecture_max_abs_diff": combined["max_abs_per_token_diff"],
        "legacy_elapsed_seconds": legacy["elapsed_seconds"],
        "legacy_approx_cost_usd": legacy["approx_cost_usd"],
        "current_h100_elapsed_seconds": current_h100_result["elapsed_seconds"],
        "current_h100_approx_cost_usd": current_h100_result["approx_cost_usd"],
        "current_b200_elapsed_seconds": current_b200_result["elapsed_seconds"],
        "current_b200_approx_cost_usd": current_b200_result["approx_cost_usd"],
    }


@app.local_entrypoint()
def main(capture: str = "all") -> None:
    if capture == "current_h100":
        current_h100_result = current_h100.remote()
        _save_raw_result("current_h100", current_h100_result)
        comparison = _build_local_comparison()
    elif capture == "all":
        legacy = legacy_h100.remote()
        current_h100_result = current_h100.remote()
        current_b200_result = current_b200.remote()
        _save_raw_result("legacy_h100", legacy)
        _save_raw_result("current_h100", current_h100_result)
        _save_raw_result("current_b200", current_b200_result)
        comparison = _build_local_comparison()
    else:
        raise ValueError("capture must be all or current_h100")
    output = Path(__file__).resolve().parents[1] / "codex/phase_b_results/logprob_compare_oldstack.json"
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
