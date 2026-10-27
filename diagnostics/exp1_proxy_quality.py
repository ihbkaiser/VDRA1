#!/usr/bin/env python3
"""Experiment 1: TV proxy vs cheap allocation heuristics on frozen node banks.

For every bank (= one policy checkpoint) and every allocation method:
  J(n) = sum_v sigma^2_MC(v) / n_v      (lower is better)
reported as % reduction vs Uniform and as oracle-gap recovery
  (J_uniform - J_method) / (J_uniform - J_oracle),
with prompt-clustered bootstrap intervals and a fresh random queue partition
in every replicate.  Nodes = roots + expanded depth-1 nodes.

  python diagnostics/exp1_proxy_quality.py --banks banks/*.jsonl.gz --out results/exp1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_io import load_bank  # noqa: E402
from common import allocate_in_queues, oracle_dispersion, quantiles, spearman  # noqa: E402

METHODS = ["uniform", "random_perm", "pilot_nll", "pilot_diversity", "raw_tv", "vdra_nocap", "vdra", "oracle"]


def flatten(roots: List[Dict], use_true: bool = False) -> List[Dict]:
    """One record per eligible node with its oracle dispersion and proxies."""
    half = "true" if use_true else "all"
    nodes = []
    for ri, r in enumerate(roots):
        recs = [(r["proxy"], r["pool"], 0)] + [(e["proxy"], e["pool"], 1) for e in r.get("expanded", [])]
        for proxy, pool, depth in recs:
            nodes.append({
                "root": ri, "depth": depth,
                "sigma2": oracle_dispersion(pool, half),
                "C_v": float(proxy.get("C_v") or 0.0),
                "C_raw": float(proxy.get("C_raw") or 0.0),
                "k_v": int(proxy.get("k_v") or 6),
                "capped": bool(proxy.get("capped")),
                "pilot_nll": float(proxy.get("pilot_nll") if proxy.get("pilot_nll") is not None else float("nan")),
                "pilot_diversity": float(proxy.get("pilot_diversity") if proxy.get("pilot_diversity") is not None else float("nan")),
                "cost_pilot": int(proxy.get("tokens_gen_pilot") or 0),
                "cost_support": int(proxy.get("tokens_gen_support") or 0),
                "cost_scored": int(proxy.get("tokens_scored_prompt") or 0),
                "pool_size": len(pool),
            })
    return nodes


def coefficients(nodes: List[Dict], method: str) -> np.ndarray:
    def col(k):
        v = np.array([n[k] for n in nodes], dtype=float)
        return np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(v).any() else 1.0)

    if method == "uniform":
        return np.ones(len(nodes))
    if method == "pilot_nll":
        return np.maximum(col("pilot_nll"), 0.0)
    if method == "pilot_diversity":
        return np.maximum(col("pilot_diversity"), 0.0)
    if method == "raw_tv":
        return col("C_raw")
    if method in ("vdra", "vdra_nocap", "random_perm"):
        return col("C_v")
    if method == "oracle":
        return col("sigma2")
    raise ValueError(method)


def cost_of(method: str, nodes: List[Dict]) -> Dict[str, float]:
    """Mean auxiliary tokens per node that each method needs before allocation."""
    pilot = float(np.mean([n["cost_pilot"] for n in nodes]))
    support = float(np.mean([n["cost_support"] for n in nodes]))
    scored = float(np.mean([n["cost_scored"] for n in nodes]))
    if method in ("vdra", "vdra_nocap", "raw_tv", "random_perm"):
        return {"aux_generated_tokens": pilot + support, "aux_scored_tokens": scored}
    if method in ("pilot_nll", "pilot_diversity"):
        return {"aux_generated_tokens": pilot, "aux_scored_tokens": 0.0}
    if method == "oracle":
        return {"aux_generated_tokens": float("nan"), "aux_scored_tokens": float("nan")}
    return {"aux_generated_tokens": 0.0, "aux_scored_tokens": 0.0}


def run_once(nodes: List[Dict], rng: np.random.Generator, capacity: int, nominal: int) -> Dict[str, float]:
    order = rng.permutation(len(nodes))
    sig = np.array([n["sigma2"] for n in nodes])
    k = [n["k_v"] for n in nodes]
    hard = [n["pool_size"] for n in nodes]
    out = {}
    allocs = {}
    for m in METHODS:
        if m == "random_perm":
            continue
        capped = [n["capped"] for n in nodes] if m == "vdra" else [False] * len(nodes)
        allocs[m] = allocate_in_queues(coefficients(nodes, m), k, capped, nominal, order, capacity, hard_cap=hard)
    # control: VDRA's counts shuffled within each queue (same histogram, no information)
    perm = allocs["vdra"].copy()
    for s in range(0, len(nodes), capacity):
        idx = order[s:s + capacity]
        perm[idx] = rng.permutation(perm[idx])
    allocs["random_perm"] = perm
    for m, a in allocs.items():
        out[m] = float(np.sum(sig / np.maximum(a, 1)))
    out["_allocs"] = allocs
    return out


def analyse_bank(path: str, reps: int, capacity: int, nominal: int, seed: int, use_true: bool) -> Dict:
    header, roots = load_bank(path)
    nodes = flatten(roots, use_true)
    rng = np.random.default_rng(seed)
    by_root: Dict[int, List[int]] = {}
    for i, n in enumerate(nodes):
        by_root.setdefault(n["root"], []).append(i)
    root_ids = sorted(by_root)

    point = run_once(nodes, rng, capacity, nominal)
    rec = {m: [] for m in METHODS}
    red = {m: [] for m in METHODS}
    for _ in range(reps):
        pick = rng.choice(root_ids, size=len(root_ids), replace=True)
        sub = [dict(nodes[i], root=j) for j, r in enumerate(pick) for i in by_root[r]]
        res = run_once(sub, rng, capacity, nominal)
        ju, jo = res["uniform"], res["oracle"]
        for m in METHODS:
            red[m].append(100.0 * (ju - res[m]) / ju if ju > 0 else float("nan"))
            rec[m].append((ju - res[m]) / (ju - jo) if ju - jo > 1e-12 else float("nan"))

    sig = [n["sigma2"] for n in nodes]
    rows = []
    for m in METHODS:
        coef = coefficients(nodes, m) if m != "random_perm" else np.full(len(nodes), np.nan)
        a = point["_allocs"][m]
        top = np.array(sig) >= np.quantile(sig, 0.75)
        rows.append({
            "bank": os.path.basename(path), "label": header.get("label", ""), "method": m,
            "J": point[m],
            "reduction_pct": quantiles(red[m]), "recovery": quantiles(rec[m]),
            "spearman_coef_vs_sigma2": spearman(coef, sig) if m != "random_perm" else float("nan"),
            "spearman_n_vs_oracle_n": spearman(a, point["_allocs"]["oracle"]),
            "budget_share_top_quartile": float(a[top].sum() / max(a.sum(), 1)),
            **cost_of(m, nodes),
            "n_nodes": len(nodes),
        })
    by_depth = {}
    for d in (0, 1):
        idx = [i for i, n in enumerate(nodes) if n["depth"] == d]
        by_depth[d] = {m: spearman(coefficients([nodes[i] for i in idx], m), [sig[i] for i in idx])
                       for m in METHODS if m != "random_perm"}
    return {"rows": rows, "by_depth": by_depth, "header": header}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--capacity", type=int, default=16)
    ap.add_argument("--nominal", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_true_values", action="store_true", help="synthetic banks only")
    a = ap.parse_args()
    paths = sorted({p for g in a.banks for p in glob.glob(g)})
    os.makedirs(a.out, exist_ok=True)
    flat = []
    for p in paths:
        res = analyse_bank(p, a.reps, a.capacity, a.nominal, a.seed, a.use_true_values)
        with open(os.path.join(a.out, os.path.basename(p).replace(".jsonl.gz", ".exp1.json")), "w") as f:
            json.dump(res, f, indent=1, default=float)
        for r in res["rows"]:
            flat.append({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()})
        print(f"[exp1] {os.path.basename(p)}: " + ", ".join(
            f"{r['method']}={r['recovery']['median']:.2f}" for r in res["rows"]))
    import csv

    with open(os.path.join(a.out, "exp1_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)


if __name__ == "__main__":
    main()
