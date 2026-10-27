"""Benchmark TreeRL's MCTS repeated-pattern check against a collision-safe hash path."""

from __future__ import annotations

import json
import platform
import random
import statistics
import string
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np


def legacy_find_repeated_patterns(text: str, pattern_length: int = 50, threshold: int = 20) -> dict[str, int]:
    """Mirror the current TreeRL implementation, including its full result dict."""
    ngrams = [text[i : i + pattern_length] for i in range(len(text) - pattern_length + 1)]
    ngram_counts = Counter(ngrams)
    return {gram: count for gram, count in ngram_counts.items() if count > threshold}


@lru_cache(maxsize=16)
def hash_weights(pattern_length: int) -> np.ndarray:
    base = 1_000_003
    mask = (1 << 64) - 1
    weights = np.empty(pattern_length, dtype=np.uint64)
    power = 1
    for index in range(pattern_length - 1, -1, -1):
        weights[index] = power
        power = (power * base) & mask
    return weights


def has_repeated_patterns_fast(
    text: str, pattern_length: int = 50, threshold: int = 20, numpy_cutoff: int = 768
) -> bool:
    """Return the legacy boolean using vectorized hashes for longer strings.

    64-bit hash collisions cannot change the answer: every hash bucket that
    crosses the threshold is checked against the original exact substrings.
    """
    if pattern_length <= 0:
        return bool(legacy_find_repeated_patterns(text, pattern_length, threshold))
    window_count = len(text) - pattern_length + 1
    if window_count <= 0:
        return False
    if window_count <= threshold:
        return False
    if len(text) < numpy_cutoff:
        return bool(legacy_find_repeated_patterns(text, pattern_length, threshold))

    codepoints = np.fromiter(map(ord, text), dtype=np.uint64, count=len(text))
    windows = np.lib.stride_tricks.sliding_window_view(codepoints, pattern_length)
    hashes = windows @ hash_weights(pattern_length)
    unique_hashes, counts = np.unique(hashes, return_counts=True)
    for hash_value in unique_hashes[counts > threshold]:
        positions = np.flatnonzero(hashes == hash_value)
        exact_counts: dict[str, int] = {}
        for position in positions:
            gram = text[int(position) : int(position) + pattern_length]
            count = exact_counts.get(gram, 0) + 1
            if count > threshold:
                return True
            exact_counts[gram] = count
    return False


def median_ms(fn, text: str, repeats: int) -> float:
    samples = []
    for _ in range(3):
        fn(text)
    for _ in range(repeats):
        started = time.perf_counter()
        fn(text)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def measure_workload(fn, texts: list[str], repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for text in texts:
            fn(text)
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def main() -> None:
    rng = random.Random(183)
    alphabet = string.ascii_letters + string.digits + " .,\\{}[]"
    length_results = []
    parity_cases = []
    for length in (49, 50, 69, 70, 256, 512, 768, 1024, 4096, 8192):
        random_text = "".join(rng.choices(alphabet, k=length))
        repeated_text = ("We can solve this step by repeated algebraic expansion. " * (length // 55 + 1))[:length]
        for text in (random_text, repeated_text):
            expected = bool(legacy_find_repeated_patterns(text))
            actual = has_repeated_patterns_fast(text)
            if actual != expected:
                raise RuntimeError(f"parity failure for length={length}: expected={expected}, got={actual}")
            parity_cases.append({"length": len(text), "expected": expected})
        old_ms = median_ms(lambda value: bool(legacy_find_repeated_patterns(value)), random_text, repeats=30)
        fast_ms = median_ms(has_repeated_patterns_fast, random_text, repeats=30)
        length_results.append(
            {
                "length": length,
                "legacy_median_ms": old_ms,
                "fast_median_ms": fast_ms,
                "speedup_ratio": old_ms / fast_ms,
                "result": bool(legacy_find_repeated_patterns(random_text)),
            }
        )
    for text in ("A" * 69, "A" * 70, "Ω" * 70, "😀" * 70):
        expected = bool(legacy_find_repeated_patterns(text))
        actual = has_repeated_patterns_fast(text)
        if actual != expected:
            raise RuntimeError(f"edge-case parity failure for length={len(text)}")
        parity_cases.append({"length": len(text), "expected": expected, "kind": "overlap-or-unicode"})

    # Approximate one 256-node MCTS expansion pass with growing aggregate answers.
    base = "We compare the candidate terms and then simplify the resulting expression. "
    workload = [(base * 80)[: 128 + (index * 3971 // 255)] for index in range(256)]
    if any(bool(legacy_find_repeated_patterns(text)) != has_repeated_patterns_fast(text) for text in workload):
        raise RuntimeError("MCTS workload parity failure")
    legacy_workload = measure_workload(lambda value: bool(legacy_find_repeated_patterns(value)), workload, repeats=21)
    fast_workload = measure_workload(has_repeated_patterns_fast, workload, repeats=21)

    result = {
        "device": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "CPU-only detector microbenchmark; no model generation or reward scoring.",
        "numpy_cutoff_characters": 768,
        "parity_cases": parity_cases,
        "random_text_by_length": length_results,
        "synthetic_256_node_workload": {
            "nodes": len(workload),
            "min_length": min(map(len, workload)),
            "max_length": max(map(len, workload)),
            "legacy_median_ms": statistics.median(legacy_workload),
            "fast_median_ms": statistics.median(fast_workload),
            "speedup_ratio": statistics.median(legacy_workload) / statistics.median(fast_workload),
            "legacy_samples_ms": legacy_workload,
            "fast_samples_ms": fast_workload,
        },
    }
    rendered = json.dumps(result, indent=2)
    output = Path(__file__).resolve().parent / "phase_b_results" / "local_treerl_repeat_detector_20261026.json"
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"saved={output.as_posix()}")


if __name__ == "__main__":
    main()
