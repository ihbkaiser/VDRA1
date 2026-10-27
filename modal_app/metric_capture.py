"""Optional canonical per-update metric capture for the Phase D equivalence run."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


_FRAMEWORKS = {"treetune", "verl", "openrlhf"}


def _scalar(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def write_training_metrics(
    framework: str,
    update: int,
    *,
    rollout_count: int,
    generated_tokens: int,
    mean_reward: Any,
    loss: Any,
    grad_norm: Any,
) -> bool:
    """Append one normalized update record when ``VDRA_METRICS_JSONL`` is set.

    The unset case deliberately does no validation or I/O, so ordinary training
    runs retain their existing behavior and overhead.
    """
    output = os.environ.get("VDRA_METRICS_JSONL")
    if not output:
        return False
    if framework not in _FRAMEWORKS:
        raise ValueError(f"framework must be one of {sorted(_FRAMEWORKS)}")
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ValueError("update must be a positive integer")

    row = {
        "framework": framework,
        "update": update,
        "rollout_count": _count(rollout_count, "rollout_count"),
        "generated_tokens": _count(generated_tokens, "generated_tokens"),
        "mean_reward": _scalar(mean_reward, "mean_reward"),
        "loss": _scalar(loss, "loss"),
        "grad_norm": _scalar(grad_norm, "grad_norm"),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    return True
