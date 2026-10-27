#!/usr/bin/env python3
"""Train/eval leakage check: normalized-text hash and 13-gram overlap.

  python analysis/check_leakage.py --train data_vdra/train.parquet \
      --eval math500=$IGPO/data/math:test aime24=$IGPO/data/aime24 ... --out results/leakage.json
Exit code 1 if any exact duplicate is found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets_util import load_problems  # noqa: E402


def norm(t: str) -> str:
    """Normalize surface formatting without discarding mathematical meaning.

    Keep command names, signs, and grouping. The previous normalizer erased all
    TeX commands and punctuation, which made sin/tan, plus/minus, and grouped
    expressions collide.
    """
    # These commands affect layout only. Preserve semantic commands such as
    # \sin, \frac, and \cdot as tokens below.
    t = re.sub(r"\\(?:left|right|displaystyle|textstyle|scriptstyle)\b", " ", t)
    t = re.sub(r"\\(?:ldots|cdots|dots)\b", " dots ", t)
    t = re.sub(r"\\(?:\[|\])", " ", t)
    t = t.replace("$", " ")

    # Canonicalize simple fractions and braces around a single exponent or
    # subscript token while retaining operator order and expression grouping.
    t = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\(([A-Za-z0-9]+)/([A-Za-z0-9]+)\)", r"\1/\2", t)
    t = re.sub(r"\(([A-Za-z0-9]+)\)/\(([A-Za-z0-9]+)\)", r"\1/\2", t)
    t = re.sub(r"\\([a-zA-Z]+)", r" \1 ", t)
    t = re.sub(r"([_^])\s*\{\s*([^{}]+?)\s*\}", r"\1\2", t)

    # Sentence punctuation is ignored, but math operators and parentheses are
    # retained so algebraically different expressions cannot collide.
    t = re.sub(r"[^A-Za-z0-9+\-=*/^_<>!()\[\]]", " ", t)
    t = re.sub(r"\s*([+\-=*/^_<>!])\s*", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\[\s+", "[", t)
    t = re.sub(r"\s+\]", "]", t)
    return " ".join(t.split())


def grams(t: str, n: int = 13):
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="name=spec")
    ap.add_argument("--diag", default=None, help="diag jsonl: must not overlap train")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--allow-exact-overlaps",
        action="store_true",
        help="write all overlap evidence but exit successfully; no training rows are removed",
    )
    a = ap.parse_args()
    train = [p for s in a.train for p in load_problems(s)]
    train_by_norm = {}
    for p in train:
        train_by_norm.setdefault(norm(p["problem"]), []).append(p["id"])
    tg = set()
    for p in train:
        tg |= grams(norm(p["problem"]))
    report, bad = {}, False
    evals = [e.split("=", 1) for e in a.eval]
    if a.diag:
        evals.append(("diag", a.diag))
    for name, spec in evals:
        probs = load_problems(spec)
        exact_matches = [
            {"eval_id": p["id"], "train_ids": train_by_norm[norm(p["problem"])]}
            for p in probs
            if norm(p["problem"]) in train_by_norm
        ]
        exact = [match["eval_id"] for match in exact_matches]
        ngram = [p["id"] for p in probs if grams(norm(p["problem"])) & tg]
        report[name] = {
            "n": len(probs),
            "exact_duplicates": exact,
            "exact_duplicate_matches": exact_matches,
            "ngram13_overlap": ngram,
        }
        bad |= bool(exact)
        print(f"[leakage] {name}: n={len(probs)} exact={len(exact)} 13gram={len(ngram)}")
    report["_summary"] = {
        "exact_overlap_evaluation_sets": sorted(
            name for name, result in report.items() if name != "_summary" and result["exact_duplicates"]
        ),
        "exact_overlaps_allowed": bool(a.allow_exact_overlaps),
        "training_rows_removed": 0,
        "status": (
            "overlaps_accepted"
            if bad and a.allow_exact_overlaps
            else "failed"
            if bad
            else "passed"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    sys.exit(1 if bad and not a.allow_exact_overlaps else 0)


if __name__ == "__main__":
    main()
