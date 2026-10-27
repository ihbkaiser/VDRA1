#!/usr/bin/env python3
"""Budget, headroom and null-test audit from VDRA / SPO JSONL logs.

  # pilot / training audit
  python analysis/budget_audit.py --logs runs/R2_s42 --out results/audit/R2_s42.json
  # null test: constant-proxy VDRA vs SPO (same seed) must match branch counts
  python analysis/budget_audit.py --null runs/null_vdra_const runs/null_spo --out results/audit/null_igpo.json

Checks
  * every dispatched queue spends its budget (sum n_v == B_q unless all nodes are capped);
  * allocation headroom: coefficient of variation of n_v (gate G1 requires >= 0.3);
  * share of nodes with n_v == 1 (their edges carry zero advantage) and n_v > nominal;
  * main-rollout tokens per tree (VDRA vs SPO, reported, not forced equal);
  * null test: identical children-per-node histograms by depth.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np


def _finite_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _valid_count(value) -> bool:
    return type(value) is int and value >= 0


def _valid_allocation_vector(n_v) -> bool:
    return isinstance(n_v, list) and bool(n_v) and all(_valid_count(value) for value in n_v)


def _valid_budget(value) -> bool:
    return _valid_count(value)


def rows(base: str, pattern: str):
    for p in glob.glob(os.path.join(base, "**", pattern), recursive=True):
        for line in open(p):
            if line.strip():
                yield json.loads(line)


def _valid_capped_vector(queue: dict) -> bool:
    if "capped" not in queue:
        return True
    n_v, capped = queue.get("n_v"), queue.get("capped")
    return (
        _valid_allocation_vector(n_v)
        and isinstance(capped, list)
        and len(capped) == len(n_v)
        and all(type(flag) is bool for flag in capped)
    )


def audit(base: str) -> dict:
    qs = list(rows(base, "vdra_queues*.jsonl"))
    nodes = list(rows(base, "vdra_nodes*.jsonl"))
    trees = list(rows(base, "vdra_trees.jsonl"))
    allocation_vectors_valid = bool(qs) and all(
        _valid_allocation_vector(q.get("n_v")) for q in qs
    )
    n_v_values = [n for q in qs for n in q["n_v"]] if allocation_vectors_valid else []
    n_v = np.array(n_v_values, dtype=float)
    viol = []
    for q in qs:
        b, nv = q.get("budget"), q.get("n_v", [])
        if not _valid_budget(b):
            continue
        if not _valid_allocation_vector(nv):
            viol.append({"budget": b, "allocated": None, "reason": "invalid_n_v"})
            continue
        capped = q.get("capped") if "capped" in q else [False] * len(nv)
        capped_vector_valid = _valid_capped_vector(q)
        allocated = sum(nv)
        all_capped = bool(nv) and capped_vector_valid and all(capped)
        if b is not None and (allocated > b or (allocated != b and not all_capped)):
            viol.append({"budget": b, "allocated": allocated})
    budget_evidence_complete = bool(qs) and all(
        _valid_budget(q.get("budget"))
        and _valid_allocation_vector(q.get("n_v"))
        and _valid_capped_vector(q)
        for q in qs
    )
    cv_n_v = _finite_float(n_v.std() / n_v.mean()) if n_v.size and n_v.mean() > 0 else None
    capped_flags_valid = allocation_vectors_valid and all(_valid_capped_vector(q) for q in qs)
    capped_flags = (
        [
            c
            for q in qs
            for c in (q.get("capped") if "capped" in q else [False] * len(q.get("n_v", [])))
        ]
        if capped_flags_valid
        else []
    )
    res = {
        "n_queues": len(qs),
        "n_nodes": sum(len(q["n_v"]) for q in qs if isinstance(q.get("n_v"), list)),
        "budget_violations": len(viol), "violation_examples": viol[:5],
        "budget_evidence_complete": budget_evidence_complete,
        "cv_n_v": cv_n_v,
        "share_n_v_eq_1": _finite_float(np.mean(n_v == 1)) if n_v.size else None,
        "n_v_hist": dict(Counter(n_v_values)),
        "share_capped": _finite_float(np.mean(capped_flags)) if capped_flags else None,
        "mean_queue_size": (
            _finite_float(np.mean([len(q["n_v"]) for q in qs]))
            if allocation_vectors_valid else None
        ),
        "mean_queue_wait_s": _finite_float(np.mean([q.get("waited_s", 0) for q in qs])) if qs else None,
    }
    if nodes:
        nominal = np.array([r.get("nominal", 6) for r in nodes])
        nv = np.array([r.get("n_v", 0) for r in nodes])
        res["share_n_v_above_nominal"] = _finite_float(np.mean(nv > nominal))
        res["mean_terminal_pilots_reused"] = _finite_float(np.mean([r.get("n_reused_terminal_pilots", 0) for r in nodes]))
        by_depth: dict[int, list[int]] = {}
        for r in nodes:
            by_depth.setdefault(int(r.get("depth", -1)), []).append(r.get("n_v", 0))
        res["mean_n_v_by_depth"] = {k: _finite_float(np.mean(v)) for k, v in sorted(by_depth.items())}
    if trees:
        res["main_rollout_tokens_per_tree"] = _finite_float(np.mean([t["main_rollout_tokens"] for t in trees]))
        aux = [t.get("vdra_aux_tokens") or {} for t in trees]
        res["aux_pilot_tokens_per_tree"] = _finite_float(np.mean([a.get("gen_pilot", 0) for a in aux]))
        res["aux_support_tokens_per_tree"] = _finite_float(np.mean([a.get("gen_support", 0) for a in aux]))
        res["score_requests_per_tree"] = _finite_float(np.mean([a.get("score_requests", 0) for a in aux]))
        res["edges_per_tree"] = _finite_float(np.mean([t["n_edges"] for t in trees]))
    res["gate_G1_headroom_ok"] = bool(cv_n_v >= 0.3) if cv_n_v is not None else None
    res["gate_G1_budget_ok"] = budget_evidence_complete and res["budget_violations"] == 0
    return res


def null_test(a: str, b: str) -> dict:
    def tree_shapes(base):
        shapes = Counter()
        for t in rows(base, "vdra_trees.jsonl"):
            counts_by_depth = {
                str(int(d)): [int(c) for c in counts]
                for d, counts in (t.get("children_per_node_by_depth") or {}).items()
            }
            signature = json.dumps(counts_by_depth, sort_keys=True, separators=(",", ":"))
            shapes[(str(t.get("problem_id")), signature)] += 1
        return shapes

    def hist(shapes):
        h = Counter()
        for (_, signature), n_trees in shapes.items():
            for d, counts in json.loads(signature).items():
                for c in counts:
                    h[(int(d), int(c))] += n_trees
        return h

    va, vb = tree_shapes(a), tree_shapes(b)
    ha, hb = hist(va), hist(vb)
    bad_a = {f"{d}:{c}": n for (d, c), n in ha.items() if c != 6 and d < 2}
    bad_b = {f"{d}:{c}": n for (d, c), n in hb.items() if c != 6 and d < 2}
    allocation_match = va == vb
    return {"vdra_constant_nonnominal": bad_a, "spo_nonnominal": bad_b,
            "allocation_match": allocation_match,
            "vdra_only_tree_signatures": [[pid, sig, n] for (pid, sig), n in sorted((va - vb).items())[:5]],
            "spo_only_tree_signatures": [[pid, sig, n] for (pid, sig), n in sorted((vb - va).items())[:5]],
            "pass": bool(va) and allocation_match,
            "note": "children counts can be < 6 only when the expander returns fewer samples (context exhausted); "
                    "such cases must appear equally in both runs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=None)
    ap.add_argument("--null", nargs=2, default=None, metavar=("VDRA_CONST_DIR", "SPO_DIR"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-budget-evidence", action="store_true")
    a = ap.parse_args()
    if a.require_budget_evidence and a.logs is None:
        ap.error("--require-budget-evidence can only be used with --logs")
    res = null_test(*a.null) if a.null else audit(a.logs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, allow_nan=False)
    print(json.dumps(res, indent=1))
    if a.null and not res["pass"]:
        sys.exit(1)
    if a.require_budget_evidence and res["gate_G1_budget_ok"] is not True:
        sys.exit(1)


if __name__ == "__main__":
    main()
