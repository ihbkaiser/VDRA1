#!/usr/bin/env python3
"""Carve the diagnostic split out of the training corpus and export the
remaining training problems for all three frameworks.

  python analysis/prepare_data.py --math_dict $IGPO/data/math --out data_vdra --n_diag 256 --seed 2026

Writes:
  data_vdra/math-nodiag/         HF DatasetDict for treetune (train split filtered)
  data_vdra/diag256.jsonl        diagnostic problems (id, problem, answer, level)
  data_vdra/train.parquet        verl / TreePO format
  data_vdra/train_treerl.jsonl   OpenRLHF / TreeRL format (text, label, data_type)
  data_vdra/split_manifest.json  ids, counts, hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdra_core.verifier import _last_boxed  # noqa: E402

INSTRUCTION = (
    "Solve the following math problem efficiently and clearly.  The last line of your response should be "
    "of the following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$. I hope it is correct' "
    "(without quotes) where ANSWER is just the final number or expression that solves the problem. "
    "Think step by step before answering.\n\n"
)


def gold_of(row) -> str:
    for k in ("answer", "final_answer", "gt_answer"):
        v = row.get(k)
        if v:
            return v[0] if isinstance(v, list) else str(v)
    return _last_boxed(row.get("solution", "")) or ""


def key_of(problem: str) -> str:
    return hashlib.sha1(" ".join(problem.split()).encode()).hexdigest()


def training_indices(
    n_rows: int,
    diagnostic_indices: list[int],
    excluded_indices: list[int] | tuple[int, ...] = (),
) -> list[int]:
    """Return source indices for training, preserving the diagnostic selection."""
    diagnostic = set(diagnostic_indices)
    excluded = set(excluded_indices)
    invalid = sorted(index for index in diagnostic | excluded if index < 0 or index >= n_rows)
    if invalid:
        raise ValueError(f"source indices out of range for {n_rows} rows: {invalid}")
    overlap = sorted(diagnostic & excluded)
    if overlap:
        raise ValueError(f"cannot exclude diagnostic rows without changing the diagnostic split: {overlap}")
    return [index for index in range(n_rows) if index not in diagnostic and index not in excluded]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math_dict", required=True, help="HF DatasetDict on disk used by treetune (data/math)")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_diag", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--exclude_train_indices",
        type=int,
        nargs="*",
        default=[],
        help="optional original source-row indices to omit from every training export",
    )
    a = ap.parse_args()
    from datasets import DatasetDict, load_from_disk

    dd = load_from_disk(a.math_dict)
    train = dd[a.train_split]
    rows = [dict(r) for r in train]
    rng = random.Random(a.seed)
    by_level = {}
    for i, r in enumerate(rows):
        by_level.setdefault(str(r.get("level", "all")), []).append(i)
    for v in by_level.values():
        rng.shuffle(v)
    diag = []
    levels = sorted(by_level)
    while len(diag) < a.n_diag:
        progressed = False
        for lv in levels:
            if by_level[lv] and len(diag) < a.n_diag:
                diag.append(by_level[lv].pop())
                progressed = True
        if not progressed:
            break
    diag_set = set(diag)
    keep = training_indices(len(rows), diag, a.exclude_train_indices)

    os.makedirs(a.out, exist_ok=True)
    new = DatasetDict({k: (train.select(keep) if k == a.train_split else v) for k, v in dd.items()})
    new.save_to_disk(os.path.join(a.out, "math-nodiag"))

    with open(os.path.join(a.out, "diag256.jsonl"), "w") as f:
        for i in diag:
            r = rows[i]
            f.write(json.dumps({"id": f"math-train-{i}", "problem": r["problem"], "answer": gold_of(r),
                                "level": r.get("level")}) + "\n")

    import pandas as pd

    verl_rows, treerl_rows = [], []
    for i in keep:
        r = rows[i]
        content = INSTRUCTION + r["problem"]
        verl_rows.append({
            "data_source": "math_vdra",
            "prompt": [{"role": "user", "content": content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gold_of(r)},
            "extra_info": {"index": i, "problem": r["problem"], "split": "train"},
        })
        treerl_rows.append({"text": content, "label": gold_of(r), "data_type": "math", "index": i})
    pd.DataFrame(verl_rows).to_parquet(os.path.join(a.out, "train.parquet"))
    with open(os.path.join(a.out, "train_treerl.jsonl"), "w") as f:
        for r in treerl_rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"n_train_total": len(rows), "n_train_kept": len(keep), "n_diag": len(diag),
                "diag_ids": [f"math-train-{i}" for i in diag], "seed": a.seed,
                "diag_hashes": [key_of(rows[i]["problem"]) for i in diag]}
    if a.exclude_train_indices:
        manifest["excluded_train_indices"] = sorted(set(a.exclude_train_indices))
    json.dump(manifest, open(os.path.join(a.out, "split_manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
