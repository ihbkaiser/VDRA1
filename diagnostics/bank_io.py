"""Node-bank format shared by the builder, Exp 1 and Exp 2.

A bank is a gzip JSONL file, one root (prompt) per line:

{
  "prompt_id": "...", "level": "Level 3",
  "proxy": {C_v, C_raw, k_v, capped, pilot_nll, pilot_diversity, tokens_*...},
  "pool": [child, ...]            # K i.i.d. depth-1 children of the root
  "expanded": [                   # subset of non-terminal pool children with their own pools
      {"pool_index": i, "proxy": {...}, "pool": [child, ...]}, ...
  ]
}
child = {"finished": bool, "reward": float | null, "refs": [R binary rewards],
         "p_true": float (synthetic banks only)}

For a finished child, refs are its own reward repeated (a terminal segment has
a deterministic value).  refs[:R/2] is the "draw half" used to build
estimates; refs[R/2:] is the independent "reference half" used as target.
A bank header line {"_header": {...}} records model, step, seed, config.
"""

from __future__ import annotations

import gzip
import json
from typing import Any, Dict, Iterator, List, Tuple


def save_bank(path: str, header: Dict[str, Any], roots: List[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"_header": header}) + "\n")
        for r in roots:
            f.write(json.dumps(r) + "\n")


def iter_bank(path: str) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_bank(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    header, roots = {}, []
    for row in iter_bank(path):
        if "_header" in row:
            header = row["_header"]
        else:
            roots.append(row)
    return header, roots
