"""Resumable frozen-policy node-bank collection primitives."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests


class ResumableBank:
    """Append one gzip member per node and checkpoint its durable byte offset."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.offset_path = self.path.with_suffix(self.path.suffix + ".offset")
        if self.path.exists() and self.offset_path.exists():
            durable = int(self.offset_path.read_text(encoding="ascii").strip())
            actual = self.path.stat().st_size
            if actual < durable:
                raise RuntimeError("bank is shorter than its durable offset marker")
            if actual > durable:
                with self.path.open("r+b") as stream:
                    stream.truncate(durable)
        self.completed_ids: set[str] = set()
        if self.path.exists():
            with gzip.open(self.path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        item = json.loads(line)
                        if "node_id" in item:
                            self.completed_ids.add(str(item["node_id"]))
        self._write_offset(self.path.stat().st_size if self.path.exists() else 0)

    def _write_offset(self, offset: int) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=self.offset_path.name, dir=self.offset_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="ascii") as stream:
                stream.write(f"{offset}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.offset_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def append(self, record: Mapping[str, Any]) -> bool:
        if "node_id" not in record:
            raise KeyError("bank records require node_id")
        node_id = str(record["node_id"])
        if node_id in self.completed_ids:
            return False
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("ab") as raw:
            with gzip.GzipFile(fileobj=raw, mode="ab", mtime=0) as stream:
                stream.write(payload.encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        self.completed_ids.add(node_id)
        self._write_offset(self.path.stat().st_size)
        return True


class VLLMClient:
    """Small OpenAI-compatible client for a local vLLM completions endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", *, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def complete(
        self,
        prompt: str,
        *,
        n: int,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        logprobs: int | None = None,
        echo: bool = False,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "n": int(n),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "seed": int(seed),
                "logprobs": logprobs,
                "echo": bool(echo),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def _stable_node_seed(seed: int, node_id: str) -> int:
    digest = hashlib.sha256(str(node_id).encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:4], "big")) % (2**30)


def collect_node_bank(
    parents: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    proxy_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    continuation_fn: Callable[[Mapping[str, Any], int, int], Sequence[Mapping[str, Any]]],
    reward_fn: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Sequence[float]],
    mc_per_half: int = 32,
    child_mc: int = 16,
    seed: int = 0,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, float | int]:
    """Run injected proxy/generation/verifier functions and persist resumably.

    Callbacks let a caller use the exact source snapshot's estimator and reward
    function. The helper does not silently substitute a different verifier.
    """
    if mc_per_half < 1 or child_mc < 1:
        raise ValueError("MC sample counts must be positive")
    bank = ResumableBank(output_path)
    started = time.perf_counter()
    attempted = 0
    for parent in parents:
        node_id = str(parent["node_id"])
        if node_id in bank.completed_ids:
            continue
        attempted += 1
        proxy = dict(proxy_fn(parent))
        node_seed = _stable_node_seed(seed, node_id)
        seed_a = node_seed
        seed_b = node_seed + 1
        samples_a = list(continuation_fn(parent, mc_per_half, seed_a))
        samples_b = list(continuation_fn(parent, mc_per_half, seed_b))
        if len(samples_a) != mc_per_half or len(samples_b) != mc_per_half:
            raise ValueError(f"continuation callback returned the wrong sample count for {node_id}")
        rewards_a = list(reward_fn(parent, samples_a))
        rewards_b = list(reward_fn(parent, samples_b))
        if len(rewards_a) != mc_per_half or len(rewards_b) != mc_per_half:
            raise ValueError(f"reward callback returned the wrong score count for {node_id}")
        child_values = []
        candidates = proxy.get("candidates", parent.get("candidates", []))
        for child_index, child in enumerate(candidates):
            child_parent = dict(parent)
            child_parent["prefix"] = (
                child.get("full_text", child.get("prefix", child))
                if isinstance(child, Mapping)
                else child
            )
            child_parent["full_text"] = child_parent["prefix"]
            child_parent["node_id"] = f"{node_id}:candidate:{child_index}"
            child_seed = (node_seed + 1_000_000 + child_index) % (2**31)
            child_samples = list(continuation_fn(child_parent, child_mc, child_seed))
            child_rewards = list(reward_fn(child_parent, child_samples))
            if len(child_rewards) != len(child_samples) or not child_rewards:
                raise ValueError(f"invalid child reward callback output for {node_id}:{child_index}")
            child_values.append({
                "candidate_index": child_index,
                "seed": child_seed,
                "samples": child_samples,
                "rewards": child_rewards,
            })
        bank.append(
            {
                "node_id": node_id,
                "prompt_id": parent.get("prompt_id"),
                "depth": parent.get("depth"),
                "prefix": parent.get("prefix", ""),
                "parent_metadata": {
                    key: parent.get(key)
                    for key in (
                        "request_seed", "parent_generation_seed", "parent_node_id",
                        "finish_reason",
                    )
                    if key in parent
                },
                "proxy": proxy,
                "R_a": {"samples": samples_a, "rewards": rewards_a, "seed": seed_a},
                "R_b": {"samples": samples_b, "rewards": rewards_b, "seed": seed_b},
                "child_values": child_values,
            }
        )
        if progress_callback is not None:
            progress_callback({
                "completed_nodes": len(bank.completed_ids),
                "new_nodes": attempted,
                "last_node_id": node_id,
                "elapsed_seconds": time.perf_counter() - started,
                "bank_bytes": bank.path.stat().st_size,
            })
    elapsed = time.perf_counter() - started
    return {"new_nodes": attempted, "completed_nodes": len(bank.completed_ids), "elapsed_seconds": elapsed}
