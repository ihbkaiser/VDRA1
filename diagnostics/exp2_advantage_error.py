#!/usr/bin/env python3
"""Experiment 2: measure the edge-averaged advantage error (Eq. 5) directly.

Training trees (depth 3, uniform final depth) are rebuilt by resampling from
a node bank:
  root r --(n_r, adaptive)--> expanded depth-1 nodes v --(n_v, adaptive)-->
  depth-2 children u from v's pool --(6 leaves, uniform)--> leaf rewards
then values are backed up exactly as in training (means of children) and the
advantage A = V_child - V_parent is compared with its target
A* = V*(child) - V*(parent) built from the bank's independent reference half.

Methods: uniform, vdra (Eq. 7), vdra_lambda (incoming-edge weights),
oracle, oracle_lambda.  Each draw resamples prompts with replacement
(cluster bootstrap) and uses common random numbers across methods.

Outputs per bank and method (median and 95% interval over draws):
  own-tree L_adv by edge type, trained-edge L_adv (A != 0, as the trainer's
  only_adv_greater_than_zero filter keeps), node-value MSE by depth, the
  shared-frontier node-value MSE, allocation descriptors, and the Prop. 1-2
  prediction for depth-1 -> depth-2 -> leaf edges next to the measured value.

  python diagnostics/exp2_advantage_error.py --banks banks/*.jsonl.gz --out results/exp2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_io import load_bank  # noqa: E402
from common import allocate_in_queues, child_mean, child_noise_var, oracle_dispersion, quantiles  # noqa: E402

METHODS = ["uniform", "vdra", "vdra_lambda", "oracle", "oracle_lambda"]
LEAVES = 6


def prepare(roots: List[Dict], use_true: bool) -> List[Dict]:
    """Precompute targets and oracle quantities per root and expanded node."""
    tgt = "true" if use_true else "ref"
    out = []
    for r in roots:
        exp = []
        for e in r.get("expanded", []):
            pool = e["pool"]
            vstar_u = np.array([child_mean(c, tgt) for c in pool])
            noise_u = np.array([0.0 if use_true else child_noise_var(c, "ref") for c in pool])
            p_u = vstar_u if use_true else np.array([child_mean(c, None) for c in pool])
            # unbiased Bernoulli variance of a leaf reward, used by the prediction
            nref = np.array([max(len(c["refs"]), 2) for c in pool])
            leaf_var = np.where([c.get("finished") for c in pool], 0.0,
                                p_u * (1 - p_u) * (1.0 if use_true else nref / (nref - 1)))
            exp.append({
                "pool": pool,
                "vstar": float(np.mean(vstar_u)),
                "noise": float(np.mean(noise_u) / len(pool)),
                "vstar_u": vstar_u, "noise_u": noise_u, "leaf_var": leaf_var,
                "sigma2": oracle_dispersion(pool, "true" if use_true else "all"),
                "sigma2_pop": float(np.var(vstar_u)) if use_true else
                              float(max(np.var(vstar_u) - np.mean(noise_u), 0.0)),
                "C_v": float(e["proxy"].get("C_v") or 0.0),
                "k_v": int(e["proxy"].get("k_v") or 6),
                "capped": bool(e["proxy"].get("capped")),
            })
        rpool = r["pool"]
        out.append({
            "vstar": float(np.mean([child_mean(c, tgt) for c in rpool])),
            "noise": 0.0 if use_true else float(np.mean([child_noise_var(c, "ref") for c in rpool]) / len(rpool)),
            "sigma2": oracle_dispersion(rpool, "true" if use_true else "all"),
            "C_v": float(r["proxy"].get("C_v") or 0.0),
            "k_v": int(r["proxy"].get("k_v") or 6),
            "capped": bool(r["proxy"].get("capped")),
            "expanded": exp,
        })
    return [r for r in out if r["expanded"]]


def coef(method: str, node: Dict, lam: float) -> float:
    if method == "uniform":
        return 1.0
    if method == "vdra":
        return node["C_v"]
    if method == "vdra_lambda":
        return lam * node["C_v"]
    if method == "oracle":
        return node["sigma2"]
    if method == "oracle_lambda":
        return lam * node["sigma2"]
    raise ValueError(method)


def leaf_estimate(child: Dict, rng: np.random.Generator) -> float:
    if child.get("finished"):
        return float(child["reward"] if child.get("reward") is not None else child["refs"][0])
    draw = child["refs"][: len(child["refs"]) // 2]
    k = min(LEAVES, len(draw))
    return float(np.mean(rng.choice(draw, size=k, replace=False)))


def one_tree_set(roots: List[Dict], method: str, rng: np.random.Generator, capacity: int, nominal: int) -> Dict:
    use_caps = method.startswith("vdra")
    # ---- root level
    R = len(roots)
    rc = [coef(method, r, 1.0) for r in roots]
    n_r = allocate_in_queues(rc, [r["k_v"] for r in roots], [r["capped"] and use_caps for r in roots],
                             nominal, rng.permutation(R), capacity, hard_cap=[len(r["expanded"]) for r in roots])
    drawn = []
    for ri, r in enumerate(roots):
        for ei in rng.choice(len(r["expanded"]), size=int(n_r[ri]), replace=False):
            drawn.append((ri, int(ei)))
    # ---- depth-1 level (queues mix prompts)
    nodes = [roots[ri]["expanded"][ei] for ri, ei in drawn]
    lam = [(n_r[ri] - 1) / n_r[ri] for ri, _ in drawn]
    vc = [coef(method, v, l) for v, l in zip(nodes, lam)]
    n_v = allocate_in_queues(vc, [v["k_v"] for v in nodes], [v["capped"] and use_caps for v in nodes],
                             nominal, rng.permutation(len(nodes)), capacity, hard_cap=[len(v["pool"]) for v in nodes])

    e01, e12, e2l, trained, mse0, mse1, mse2 = [], [], [], [], [], [], []
    pred_terms, pred_edges, meas_12l = 0.0, 0, []
    vhat_v_all = []
    for (ri, ei), v, nv in zip(drawn, nodes, n_v):
        kids = rng.choice(len(v["pool"]), size=int(nv), replace=False)
        vhat_u = np.array([leaf_estimate(v["pool"][k], rng) for k in kids])
        vhat_v = float(np.mean(vhat_u))
        vhat_v_all.append((ri, vhat_v))
        mse1.append((vhat_v - v["vstar"]) ** 2 - v["noise"])
        for k, vu in zip(kids, vhat_u):
            a_hat = vu - vhat_v
            a_star = v["vstar_u"][k] - v["vstar"]
            noise = v["noise_u"][k] + v["noise"] - 2 * v["noise_u"][k] / len(v["pool"])
            err = (a_hat - a_star) ** 2 - noise
            e12.append(err)
            meas_12l.append(err)
            if abs(a_hat) > 1e-12:
                trained.append(err)
            mse2.append((vu - v["vstar_u"][k]) ** 2 - v["noise_u"][k])
            if not v["pool"][k].get("finished"):
                # leaf edges: A_hat - A* = V*(u) - V_hat(u), six identical-error edges
                le = (v["vstar_u"][k] - vu) ** 2 - v["noise_u"][k]
                e2l.extend([le] * LEAVES)
                meas_12l.extend([le] * LEAVES)
                trained.extend([le] * LEAVES)   # leaf rewards differ from V_hat(u) almost surely
        # Prop. 1-2 prediction for this parent: (n-1) mean tau^2 + n Var(mean of drawn child values)
        N = len(v["pool"])
        tau2 = float(np.mean(v["leaf_var"])) / LEAVES
        pred_terms += (nv - 1) * tau2 + v["sigma2_pop"] * (N - nv) / max(N - 1, 1)
        pred_terms += float(np.sum(v["leaf_var"][kids]))          # leaf edges: sigma^2_u each
        pred_edges += int(nv) + LEAVES * int(sum(1 for k in kids if not v["pool"][k].get("finished")))
    by_root: Dict[int, List[float]] = {}
    for ri, vv in vhat_v_all:
        by_root.setdefault(ri, []).append(vv)
    # root edges
    vhat_v_iter = iter(vhat_v_all)
    for (ri, ei), v in zip(drawn, nodes):
        _, vv = next(vhat_v_iter)
        vr = float(np.mean(by_root[ri]))
        err = ((vv - vr) - (v["vstar"] - roots[ri]["vstar"])) ** 2 - (v["noise"] + roots[ri]["noise"])
        e01.append(err)
        if abs(vv - vr) > 1e-12:
            trained.append(err)
    for ri, r in enumerate(roots):
        if ri in by_root:
            mse0.append((float(np.mean(by_root[ri])) - r["vstar"]) ** 2 - r["noise"])
    all_e = e01 + e12 + e2l
    nonleaf_n = np.concatenate([n_r, n_v])
    return {
        "L_own_all": float(np.mean(all_e)), "L_own_root_d1": float(np.mean(e01)),
        "L_own_d1_d2": float(np.mean(e12)), "L_own_d2_leaf": float(np.mean(e2l)) if e2l else float("nan"),
        "L_trained": float(np.mean(trained)) if trained else float("nan"),
        "mse_d0": float(np.mean(mse0)), "mse_d1": float(np.mean(mse1)), "mse_d2": float(np.mean(mse2)),
        "pred_d1_d2_leaf": pred_terms / max(pred_edges, 1), "meas_d1_d2_leaf": float(np.mean(meas_12l)),
        "frac_n1": float(np.mean(nonleaf_n == 1)), "n_edges": len(all_e),
        "mean_lambda": float(np.mean(lam)),
    }


def shared_frontier(roots: List[Dict], method: str, rng: np.random.Generator, capacity: int, nominal: int) -> float:
    nodes = [e for r in roots for e in r["expanded"]]
    lam = (nominal - 1) / nominal
    vc = [coef(method, v, lam) for v in nodes]
    n_v = allocate_in_queues(vc, [v["k_v"] for v in nodes], [v["capped"] and method.startswith("vdra") for v in nodes],
                             nominal, rng.permutation(len(nodes)), capacity, hard_cap=[len(v["pool"]) for v in nodes])
    errs = []
    for v, nv in zip(nodes, n_v):
        kids = rng.choice(len(v["pool"]), size=int(nv), replace=False)
        vh = float(np.mean([leaf_estimate(v["pool"][k], rng) for k in kids]))
        errs.append((vh - v["vstar"]) ** 2 - v["noise"])
    return float(np.mean(errs))


def analyse(path: str, draws: int, capacity: int, nominal: int, seed: int, use_true: bool) -> Dict:
    header, raw = load_bank(path)
    roots = prepare(raw, use_true)
    rng0 = np.random.default_rng(seed)
    per = {m: [] for m in METHODS}
    sf = {m: [] for m in METHODS}
    for d in range(draws):
        pick = rng0.choice(len(roots), size=len(roots), replace=True)
        sub = [roots[i] for i in pick]
        for m in METHODS:
            rng = np.random.default_rng([seed, d])          # common random numbers across methods
            per[m].append(one_tree_set(sub, m, rng, capacity, nominal))
            sf[m].append(shared_frontier(sub, m, np.random.default_rng([seed, d, 1]), capacity, nominal))
    rows = []
    for m in METHODS:
        row = {"bank": os.path.basename(path), "label": header.get("label", ""), "method": m}
        for k in per[m][0]:
            row[k] = quantiles([x[k] for x in per[m]])
        row["sf_mse_d1"] = quantiles(sf[m])
        # paired difference vs uniform (same draws)
        row["L_own_all_minus_uniform"] = quantiles(
            [x["L_own_all"] - u["L_own_all"] for x, u in zip(per[m], per["uniform"])])
        row["L_trained_minus_uniform"] = quantiles(
            [x["L_trained"] - u["L_trained"] for x, u in zip(per[m], per["uniform"])])
        rows.append(row)
    return {"header": header, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--capacity", type=int, default=16)
    ap.add_argument("--nominal", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_true_values", action="store_true", help="synthetic banks only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    flat = []
    for p in sorted({p for g in a.banks for p in glob.glob(g)}):
        res = analyse(p, a.draws, a.capacity, a.nominal, a.seed, a.use_true_values)
        with open(os.path.join(a.out, os.path.basename(p).replace(".jsonl.gz", ".exp2.json")), "w") as f:
            json.dump(res, f, indent=1, default=float)
        for r in res["rows"]:
            flat.append({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()})
        print(f"[exp2] {os.path.basename(p)}: " + ", ".join(
            f"{r['method']} L={r['L_own_all']['median']:.4f}" for r in res["rows"]))
    with open(os.path.join(a.out, "exp2_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)


if __name__ == "__main__":
    main()
