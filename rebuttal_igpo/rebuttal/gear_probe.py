"""Controlled frozen-checkpoint probe using the archived GEAR TV estimator."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx
from datasets import load_from_disk
from transformers import AutoTokenizer

from .node_bank import ResumableBank, VLLMClient, collect_node_bank
from .smoke_vllm import QUESTION_TEMPLATE


SOURCE_ROOT = Path("/workspace/src/gear_source_rebuttal/gear")
DEFAULT_TOKENIZER = Path("/workspace/rebuttal_out/tokenizer_iter100_fixed")


def _stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def _source_imports():
    root = str(SOURCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from treetune.gear.tv_estimators import ConditionalTVEstimator
    from treetune.gear.vllm_scorer import VLLMLogprobClient, make_lp_scorer
    from treetune.tasks.math import MATH

    return ConditionalTVEstimator, VLLMLogprobClient, make_lp_scorer, MATH


class VLLMNodeExpander:
    """Small async adapter exposing vLLM outputs as GEAR nodes."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        tokenizer: Any,
        temperature: float,
        top_p: float,
        seed: int,
        client: httpx.AsyncClient,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.seed = int(seed)
        self.client = client
        self.call_index = 0
        self.stats: dict[str, Any] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "seconds": 0.0,
        }

    async def expand(
        self,
        *,
        current_node: Mapping[str, Any],
        prefix: str,
        depth: int,
        max_tokens: int,
        branch_factor: int,
    ) -> list[dict[str, Any]]:
        self.call_index += 1
        parent_id = str(current_node.get("node_id", "parent"))
        seed = _stable_seed(self.seed, f"{parent_id}|{depth}|{self.call_index}")
        started = time.perf_counter()
        response = await self.client.post(
            f"{self.base_url}/completions",
            json={
                "model": self.model,
                "prompt": prefix,
                "n": int(branch_factor),
                "max_tokens": int(max_tokens),
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": seed,
                "logprobs": 5,
                "echo": False,
            },
        )
        if response.is_error:
            raise RuntimeError(
                f"vLLM generation failed with HTTP {response.status_code}: {response.text[:1000]}"
            )
        payload = response.json()
        elapsed = time.perf_counter() - started
        usage = payload.get("usage", {})
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self.stats["seconds"] += elapsed

        nodes = []
        for choice in payload.get("choices", []):
            text = choice.get("text", "")
            logprobs = choice.get("logprobs") or {}
            token_lps = [float(x) for x in (logprobs.get("token_logprobs") or []) if x is not None]
            entropies = []
            for row in logprobs.get("top_logprobs") or []:
                finite = []
                for value in (row or {}).values():
                    try:
                        lp = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(lp):
                        finite.append(math.exp(lp))
                total = sum(finite)
                if total > 0:
                    entropies.append(-sum((p / total) * math.log(p / total) for p in finite if p > 0))
            choice_index = int(choice.get("index", len(nodes)))
            nodes.append({
                "node_id": f"{parent_id}:expand:{depth}:{self.call_index}:{choice_index}",
                "text": text,
                "depth": int(depth),
                "full_text": prefix + text,
                "finish_reason": choice.get("finish_reason"),
                "sum_logprobs": sum(token_lps),
                "num_tokens": len(token_lps),
                "mean_nll": -sum(token_lps) / len(token_lps) if token_lps else None,
                "mean_top5_entropy": sum(entropies) / len(entropies) if entropies else None,
                "generation_seed": seed,
                "generated_tokens": len(self.tokenizer.encode(text, add_special_tokens=False)),
                "request_seconds": elapsed,
            })
        if len(nodes) != int(branch_factor):
            raise RuntimeError(f"vLLM returned {len(nodes)} of {branch_factor} requested branches")
        return nodes


def _make_math_task(MATH):
    """Use the archived MATH task's extraction and grading implementation."""
    task = MATH.__new__(MATH)
    task.use_minerva_few_shot_prompt = False
    task.answer_prefix = None
    task.inplace_split_solution = True
    task.use_dataset_answer = False
    return task


def _task_reward(task: Any, response: str, row: Mapping[str, Any]) -> float:
    predicted = task.extract_predicted_answer_from_text(response, row["problem"])
    if len(response.split("####")) > 2:
        return -2.0
    if predicted is None:
        return 0.0  # the selected source config penalizes unfinished answers by 0
    return float(task.grade_answer(
        given_answer=predicted,
        ground_truth=row.get("answer"),
        item=dict(row),
    ))


def _serialize_node(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: node.get(key)
        for key in (
            "node_id", "text", "full_text", "finish_reason", "sum_logprobs",
            "num_tokens", "mean_nll", "mean_top5_entropy", "generation_seed",
            "generated_tokens", "request_seconds",
        )
    }


def estimate_proxy(
    parent: Mapping[str, Any],
    *,
    base_url: str,
    model: str,
    tokenizer: Any,
    seed: int,
    gamma: float = 0.9,
    n_tv_estimates: int = 16,
    first_tokens: int = 120,
    second_tokens: int = 60,
) -> dict[str, Any]:
    ConditionalTVEstimator, LogprobClient, make_lp_scorer, _ = _source_imports()

    async def run() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=1200.0) as http_client:
            expander = VLLMNodeExpander(
                base_url=base_url,
                model=model,
                tokenizer=tokenizer,
                temperature=0.6,
                top_p=0.95,
                seed=seed,
                client=http_client,
            )
            scorer_client = LogprobClient(
                api_base=base_url,
                model=model,
                timeout=1200.0,
                max_concurrency=64,
                retry_attempts=3,
            )
            scoring = {"requests": 0, "prompt_tokens": 0, "seconds": 0.0}
            original_score = scorer_client.prompt_logprobs

            async def tracked_score(prompt: str) -> list[float]:
                started = time.perf_counter()
                values = await original_score(prompt)
                scoring["requests"] += 1
                scoring["prompt_tokens"] += len(tokenizer.encode(prompt, add_special_tokens=False))
                scoring["seconds"] += time.perf_counter() - started
                return values

            scorer_client.prompt_logprobs = tracked_score
            estimator = ConditionalTVEstimator(
                scorer=make_lp_scorer(scorer_client, lambda text: tokenizer.encode(text, add_special_tokens=False)),
                node_expander=expander,
                gamma=gamma,
                mode="hierarchical",
                n_tv_estimates=n_tv_estimates,
                first_phase_tokens=first_tokens,
                second_phase_tokens=second_tokens,
                tv_includes_half_factor=True,
            )
            started = time.perf_counter()
            result = await estimator.estimate_for_parent(parent, depth=int(parent["depth"]))
            await scorer_client.aclose()
            return {
                "sigma2": result.reward_variance,
                "pair_tvs": [
                    {"i": i, "j": j, "tv": value}
                    for (i, j), value in sorted(result.pair_tvs.items())
                ],
                "logp_matrix": result.logp_matrix,
                "prob_matrix": result.prob_matrix,
                "pilot_samples": [
                    {"first": _serialize_node(s.first), "second": _serialize_node(s.second)}
                    for s in result.samples
                ],
                "candidates": [_serialize_node(node) for node in result.candidates],
                "pilot_entropy_mean_top5": _mean(
                    [node.get("mean_top5_entropy") for node in result.candidates]
                ),
                "pilot_nll_mean": _mean([node.get("mean_nll") for node in result.candidates]),
                "generation": dict(expander.stats),
                "scoring": scoring,
                "elapsed_seconds": time.perf_counter() - started,
                "estimator": {
                    "source": "gear_source_rebuttal/treetune/gear/tv_estimators.py::ConditionalTVEstimator",
                    "gamma": gamma,
                    "mode": "hierarchical",
                    "n_tv_estimates": n_tv_estimates,
                    "first_phase_tokens": first_tokens,
                    "second_phase_tokens": second_tokens,
                    "tv_includes_half_factor": True,
                },
            }

    return asyncio.run(run())


def _mean(values: Sequence[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def run_probe(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER,
    base_url: str = "http://127.0.0.1:18000/v1",
    model: str = "iter100",
    seed: int = 42,
    prompt_count: int = 16,
    segment_tokens: int = 100,
    max_new_tokens: int = 1024,
    mc_per_half: int = 32,
    child_mc: int = 16,
    include_depth1: bool = True,
) -> dict[str, Any]:
    ConditionalTVEstimator, _, _, MATH = _source_imports()
    del ConditionalTVEstimator
    dataset = load_from_disk(str(dataset_dir))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if int(manifest["seed"]) != seed or int(manifest["prompts_total"]) < prompt_count:
        raise ValueError("prompt manifest does not match the fixed probe seed/scope")
    if prompt_count > 32:
        raise ValueError("this frozen protocol manifest contains at most 32 prompts")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    task = _make_math_task(MATH)
    row_by_id: dict[str, Mapping[str, Any]] = {}
    expander_stats: list[dict[str, Any]] = []
    parents = []
    sync_client = VLLMClient(base_url, timeout=1200.0)

    for prompt_record in manifest["prompts"][:prompt_count]:
        row = dataset[int(prompt_record["row_index"])]
        prompt_id = str(prompt_record["prompt_id"])
        query_prompt = QUESTION_TEMPLATE.format(query=row["problem"])
        row_by_id[prompt_id] = row
        root_id = f"{prompt_id}:depth0:root"
        root = {
            "node_id": root_id,
            "prompt_id": prompt_id,
            "depth": 0,
            "prefix": query_prompt,
            "full_text": query_prompt,
            "original_prompt": query_prompt,
            "request_seed": int(prompt_record["request_seed"]),
        }
        if include_depth1:
            started = time.perf_counter()
            root_seed = _stable_seed(seed, f"{root_id}|depth1")
            root_response = sync_client.complete(
                query_prompt,
                n=6,
                model=model,
                max_tokens=segment_tokens,
                temperature=0.6,
                top_p=0.95,
                seed=root_seed,
            )
            elapsed = time.perf_counter() - started
            usage = root_response.get("usage", {})
            depth1_nodes = []
            for choice in root_response.get("choices", []):
                text = choice.get("text", "")
                child_index = int(choice.get("index", len(depth1_nodes)))
                node = {
                    "node_id": f"{prompt_id}:depth1:{child_index}",
                    "prompt_id": prompt_id,
                    "depth": 1,
                    "prefix": query_prompt + text,
                    "full_text": query_prompt + text,
                    "original_prompt": query_prompt,
                    "request_seed": int(prompt_record["request_seed"]) + child_index + 1,
                "parent_generation_seed": root_seed,
                    "parent_node_id": root_id,
                    "finish_reason": choice.get("finish_reason"),
                }
                depth1_nodes.append(node)
            if len(depth1_nodes) != 6:
                raise RuntimeError(f"vLLM returned {len(depth1_nodes)} of 6 depth-1 states for {prompt_id}")
            expander_stats.append({
                "prompt_id": prompt_id,
                "seed": root_seed,
                "depth1_request_seconds": elapsed,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            })
            parents.extend([root, *depth1_nodes])
        else:
            parents.append(root)

    def proxy_fn(parent: Mapping[str, Any]) -> Mapping[str, Any]:
        return estimate_proxy(
            parent,
            base_url=base_url,
            model=model,
            tokenizer=tokenizer,
            seed=int(parent.get("request_seed", seed)),
        )

    generation_calls: list[dict[str, Any]] = []

    def continuation_fn(parent: Mapping[str, Any], n: int, request_seed: int) -> Sequence[Mapping[str, Any]]:
        prefix = str(parent["prefix"])
        started = time.perf_counter()
        result = sync_client.complete(
            prefix,
            n=n,
            model=model,
            max_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            seed=request_seed,
        )
        elapsed = time.perf_counter() - started
        usage = result.get("usage", {})
        generation_calls.append({
            "node_id": str(parent["node_id"]),
            "n": n,
            "seed": request_seed,
            "seconds": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        })
        samples = []
        for choice in result.get("choices", []):
            text = choice.get("text", "")
            samples.append({
                "text": text,
                "full_text": prefix + text,
                "finish_reason": choice.get("finish_reason"),
                "generated_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                "request_seed": request_seed,
                "request_seconds": elapsed,
            })
        return samples

    def reward_fn(parent: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> Sequence[float]:
        row = row_by_id[str(parent["prompt_id"])]
        original_prompt = str(parent["original_prompt"])
        rewards = []
        for sample in samples:
            full_text = str(sample.get("full_text", ""))
            response = full_text[len(original_prompt):] if full_text.startswith(original_prompt) else str(sample.get("text", ""))
            rewards.append(_task_reward(task, response, row))
        return rewards

    progress_path = Path(str(output_path) + ".progress.json")

    def progress_callback(progress: Mapping[str, Any]) -> None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=progress_path.name, dir=progress_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(dict(progress), stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, progress_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        print(json.dumps(dict(progress), ensure_ascii=False, sort_keys=True), flush=True)

    result = collect_node_bank(
        parents,
        output_path,
        proxy_fn=proxy_fn,
        continuation_fn=continuation_fn,
        reward_fn=reward_fn,
        mc_per_half=mc_per_half,
        child_mc=child_mc,
        seed=seed,
        progress_callback=progress_callback,
    )
    summary = {
        **result,
        "prompt_count": prompt_count,
        "parent_count": len(parents),
        "include_depth1": include_depth1,
        "fixed_prompt_manifest": str(manifest_path),
        "prompt_ids": [item["prompt_id"] for item in manifest["prompts"][:prompt_count]],
        "depth1_generation": expander_stats,
        "continuation_requests": len(generation_calls),
        "continuation_completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in generation_calls),
        "continuation_seconds": sum(float(row["seconds"]) for row in generation_calls),
        "protocol": {
            "checkpoint": "/workspace/src/vdra_iter_0100/checkpoints/ckpt--iter_0100--epoch_1.00--step_0400/hf_pretrained",
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
            "branch_factor": 6,
            "segment_tokens": segment_tokens,
            "mc_per_half": mc_per_half,
            "child_mc": child_mc,
            "source_reward": "GEAR snapshot treetune.tasks.math.MATH answer extraction and grading; answer_prefix=None; unfinished response reward=0",
            "tokenizer_fix_mistral_regex": True,
        },
    }
    summary_path = Path(output_path).with_suffix(Path(output_path).suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="iter100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-count", type=int, default=16)
    parser.add_argument("--segment-tokens", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--mc-per-half", type=int, default=32)
    parser.add_argument("--child-mc", type=int, default=16)
    parser.add_argument("--root-only", action="store_true", help="collect only depth-0 parent nodes for an integration smoke")
    args = parser.parse_args(argv)
    result = run_probe(
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        base_url=args.base_url,
        model=args.model,
        seed=args.seed,
        prompt_count=args.prompt_count,
        segment_tokens=args.segment_tokens,
        max_new_tokens=args.max_new_tokens,
        mc_per_half=args.mc_per_half,
        child_mc=args.child_mc,
        include_depth1=not args.root_only,
    )
    print(json.dumps({
        key: result[key]
        for key in ("new_nodes", "completed_nodes", "elapsed_seconds", "prompt_count", "parent_count", "continuation_requests", "continuation_completion_tokens")
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
