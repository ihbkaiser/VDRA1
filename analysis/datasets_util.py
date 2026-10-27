"""Load evaluation / training problems from HF-on-disk, jsonl, json or parquet
into a uniform list of {"id", "problem", "answer"} dicts."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdra_core.verifier import _last_boxed  # noqa: E402


def _gold(r: Dict) -> str:
    for k in ("answer", "final_answer", "gt_answer", "label"):
        v = r.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, list):
            if not v:
                continue
            v = v[0]
        if v is None:
            continue
        answer = str(v).strip()
        if answer:
            return answer
    reward_model = r.get("reward_model")
    if isinstance(reward_model, dict):
        value = reward_model.get("ground_truth")
        if isinstance(value, list):
            value = value[0] if value else None
        if value is not None and str(value).strip():
            return str(value).strip()
    solution_answer = _last_boxed(str(r.get("solution", "")))
    if solution_answer and solution_answer.strip():
        return solution_answer.strip()
    raise KeyError(f"no gold answer field or boxed solution in {sorted(r)}")


def _problem(r: Dict) -> str:
    for k in ("problem", "question", "query", "text"):
        if r.get(k):
            return str(r[k])
    extra = r.get("extra_info")
    if isinstance(extra, dict) and extra.get("problem"):
        return str(extra["problem"])
    raise KeyError(f"no problem field in {sorted(r)}")


def _id(r: Dict, index: int) -> str:
    for key in ("id", "unique_id", "index"):
        value = r.get(key)
        if value is not None:
            return str(value)
    extra = r.get("extra_info")
    if isinstance(extra, dict) and extra.get("index") is not None:
        return str(extra["index"])
    return str(index)


def load_problems(spec: str) -> List[Dict]:
    """spec = PATH or PATH:SPLIT."""
    path, split = (spec.rsplit(":", 1) + [None])[:2] if ":" in spec and not os.path.exists(spec) else (spec, None)
    rows: List[Dict] = []
    if os.path.isdir(path):
        from datasets import load_from_disk

        d = load_from_disk(path)
        if hasattr(d, "keys"):
            split = split or ("test" if "test" in d else list(d.keys())[0])
            d = d[split]
        rows = [dict(r) for r in d]
    elif path.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(path) if l.strip()]
    elif path.endswith(".json"):
        rows = json.load(open(path))
    elif path.endswith(".parquet"):
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
    else:
        raise ValueError(f"unsupported dataset spec: {spec}")
    out = []
    for i, r in enumerate(rows):
        out.append({"id": _id(r, i), "problem": _problem(r), "answer": _gold(r)})
    return out
