#!/usr/bin/env python3
"""Primary statistics for Experiments 3 and 4.

Inputs
  --eval   problem-level JSONL files written by eval_checkpoints.py
  --runs   CSV manifest: run_id,pair,role,seed   (role = baseline | hybrid)
Outputs (in --out)
  curves.csv          per run x step x benchmark accuracy and macro-average (greedy)
  comparisons.csv     per pair x regime (matched_updates | matched_gpu_hours):
                      hierarchical-bootstrap difference in macro-average (pp) with 95% CI,
                      bootstrap p-value, Holm-adjusted p, seeds won, mixed-model log-odds
  auc.csv             area under Pass@1-vs-step and Pass@1-vs-GPU-hours (common range)
  avgk.csv            avg@k paired differences (if sampled eval exists)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


def load_eval(paths: List[str]) -> List[Dict]:
    """Load problem-level rows; duplicates of (run, step, benchmark, mode, problem)
    are dropped (last wins), so overlapping evaluation outputs never double count."""
    rows = {}
    for g in paths:
        for p in sorted(glob.glob(g)):
            if ".tmp" in os.path.basename(p):
                continue
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    rows[(r["run_id"], int(r["step"]), r["benchmark"], r["mode"], str(r["problem_id"]))] = r
    return list(rows.values())


def load_runs(path: str) -> Dict[str, Dict]:
    with open(path) as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def table(rows, mode="greedy"):
    """acc[run][step][bench][problem] = correct; gh[run][step] = gpu_hours."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    gh = defaultdict(dict)
    for r in rows:
        if r["mode"] != mode:
            continue
        val = r["correct"] if mode == "greedy" else r["k_correct"] / r["k"]
        acc[r["run_id"]][int(r["step"])][r["benchmark"]][r["problem_id"]] = float(val)
        if r.get("gpu_hours") is not None:
            gh[r["run_id"]][int(r["step"])] = float(r["gpu_hours"])
    return acc, gh


def macro(bench_map) -> float:
    return float(np.mean([np.mean(list(p.values())) for p in bench_map.values()])) if bench_map else float("nan")


def hier_bootstrap(pairs, reps=10000, seed=0):
    """pairs: list over seeds of (hyb_bench_map, base_bench_map) with identical
    problem sets.  Resample seeds, then problems within each benchmark."""
    rng = np.random.default_rng(seed)
    benches = sorted(pairs[0][0].keys())
    probs = {b: sorted(pairs[0][0][b].keys()) for b in benches}
    # arrays [seed, problem] per benchmark
    H = {b: np.array([[h[b].get(p, np.nan) for p in probs[b]] for h, _ in pairs]) for b in benches}
    B = {b: np.array([[s[b].get(p, np.nan) for p in probs[b]] for _, s in pairs]) for b in benches}
    S = len(pairs)
    diffs = np.empty(reps)
    for k in range(reps):
        si = rng.integers(0, S, S)
        per_b = []
        for b in benches:
            n = H[b].shape[1]
            pi = rng.integers(0, n, n)
            per_b.append(np.nanmean(H[b][np.ix_(si, pi)]) - np.nanmean(B[b][np.ix_(si, pi)]))
        diffs[k] = np.mean(per_b)
    point = float(np.mean([macro(h) - macro(s) for h, s in pairs]))
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"diff_pp": 100 * point, "lo_pp": 100 * float(np.quantile(diffs, 0.025)),
            "hi_pp": 100 * float(np.quantile(diffs, 0.975)), "p_boot": float(min(1.0, p))}


def mixed_model(pairs):
    """Bayesian mixed logistic regression: correct ~ hybrid + benchmark + (1|problem) + (1|seed)."""
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as exc:  # pragma: no cover
        return {"mm_error": repr(exc)}
    recs = []
    for s, (h, b) in enumerate(pairs):
        for role, mp in (("hyb", h), ("base", b)):
            for bench, probs in mp.items():
                for pid, c in probs.items():
                    recs.append({"y": int(c), "hybrid": int(role == "hyb"), "bench": bench,
                                 "prob": f"{bench}/{pid}", "seed": str(s)})
    df = pd.DataFrame(recs)
    try:
        m = BinomialBayesMixedGLM.from_formula(
            "y ~ hybrid + C(bench)", {"prob": "0 + C(prob)", "seed": "0 + C(seed)"}, df)
        r = m.fit_vb()
        i = list(r.model.exog_names).index("hybrid")
        mu, sd = float(r.fe_mean[i]), float(r.fe_sd[i])
        return {"mm_logodds": mu, "mm_lo": mu - 1.96 * sd, "mm_hi": mu + 1.96 * sd}
    except Exception as exc:
        return {"mm_error": repr(exc)}


def holm(ps: List[float]) -> List[float]:
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def nearest_step(gh_run: Dict[int, float], target: float) -> int:
    return min(gh_run, key=lambda s: abs(gh_run[s] - target))


def auc(xs, ys, x_max):
    pts = sorted((x, y) for x, y in zip(xs, ys) if x <= x_max + 1e-9)
    if len(pts) < 2:
        return float("nan")
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    if x[-1] <= x[0]:
        return float("nan")
    area = np.sum((y[:-1] + y[1:]) * np.diff(x) / 2.0)
    return float(area / (x[-1] - x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=220)
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs)
    rows = load_eval(a.eval)
    acc, gh = table(rows, "greedy")

    with open(os.path.join(a.out, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "pair", "role", "seed", "step", "gpu_hours", "benchmark", "accuracy"])
        for rid, steps in acc.items():
            meta = runs.get(rid, {})
            for st, bm in sorted(steps.items()):
                for b, probs in bm.items():
                    w.writerow([rid, meta.get("pair"), meta.get("role"), meta.get("seed"), st, gh[rid].get(st), b,
                                np.mean(list(probs.values()))])
                w.writerow([rid, meta.get("pair"), meta.get("role"), meta.get("seed"), st, gh[rid].get(st), "macro", macro(bm)])

    by_pair = defaultdict(lambda: defaultdict(dict))
    for rid, m in runs.items():
        by_pair[m["pair"]][m["seed"]][m["role"]] = rid

    comps, aucs = [], []
    for pair, seeds in sorted(by_pair.items()):
        full = {s: r for s, r in seeds.items() if "baseline" in r and "hybrid" in r}
        for regime in ("matched_updates", "matched_gpu_hours"):
            pairs, wins, notes = [], 0, []
            for s, r in sorted(full.items()):
                h, b = r["hybrid"], r["baseline"]
                if a.step not in acc[h]:
                    notes.append(f"{h} missing step {a.step}")
                    continue
                if regime == "matched_updates":
                    bs = a.step
                else:
                    if a.step not in gh[h] or not gh[b]:
                        notes.append(f"no gpu_hours for {h} or {b}")
                        continue
                    bs = nearest_step(gh[b], gh[h][a.step])
                    notes.append(f"seed {s}: hybrid {gh[h][a.step]:.1f} GPU-h vs baseline step {bs} ({gh[b][bs]:.1f})")
                if bs not in acc[b]:
                    notes.append(f"{b} missing step {bs}")
                    continue
                pairs.append((acc[h][a.step], acc[b][bs]))
                wins += int(macro(acc[h][a.step]) > macro(acc[b][bs]))
            if not pairs:
                continue
            row = {"pair": pair, "regime": regime, "n_seeds": len(pairs), "seeds_won": wins,
                   **hier_bootstrap(pairs, a.reps), **mixed_model(pairs), "notes": " | ".join(notes)}
            comps.append(row)
        # areas under the curves
        for s, r in sorted(full.items()):
            h, b = r["hybrid"], r["baseline"]
            steps_h = sorted(acc[h]); steps_b = sorted(acc[b])
            if not steps_h or not steps_b:
                continue
            x_max = min(max(steps_h), max(steps_b), a.step)
            row = {"pair": pair, "seed": s,
                   "auc_step_hybrid": auc(steps_h, [macro(acc[h][t]) for t in steps_h], x_max),
                   "auc_step_baseline": auc(steps_b, [macro(acc[b][t]) for t in steps_b], x_max)}
            if gh[h] and gh[b]:
                gx = min(max(gh[h].values()), max(gh[b].values()))
                row["auc_gpuh_hybrid"] = auc([gh[h][t] for t in steps_h if t in gh[h]],
                                             [macro(acc[h][t]) for t in steps_h if t in gh[h]], gx)
                row["auc_gpuh_baseline"] = auc([gh[b][t] for t in steps_b if t in gh[b]],
                                               [macro(acc[b][t]) for t in steps_b if t in gh[b]], gx)
            aucs.append(row)
    if comps:
        adj = holm([c["p_boot"] for c in comps])
        for c, p in zip(comps, adj):
            c["p_holm"] = p
        keys = sorted({k for c in comps for k in c})
        with open(os.path.join(a.out, "comparisons.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(comps)
    if aucs:
        keys = sorted({k for c in aucs for k in c})
        with open(os.path.join(a.out, "auc.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(aucs)

    # avg@k (sampled) paired differences at the shared steps
    acck, _ = table(rows, mode=next((r["mode"] for r in rows if r["mode"].startswith("avg@")), "none"))
    if acck:
        out = []
        for pair, seeds in sorted(by_pair.items()):
            for s, r in sorted(seeds.items()):
                if "baseline" in r and "hybrid" in r:
                    h, b = r["hybrid"], r["baseline"]
                    for st in sorted(set(acck[h]) & set(acck[b])):
                        out.append({"pair": pair, "seed": s, "step": st,
                                    "hybrid": macro(acck[h][st]), "baseline": macro(acck[b][st])})
        with open(os.path.join(a.out, "avgk.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pair", "seed", "step", "hybrid", "baseline"])
            w.writeheader()
            w.writerows(out)
    for c in comps:
        print(f"[stats] {c['pair']:>8} {c['regime']:<18} diff={c['diff_pp']:+.2f}pp "
              f"[{c['lo_pp']:+.2f},{c['hi_pp']:+.2f}] p_holm={c['p_holm']:.3f} seeds_won={c['seeds_won']}/{c['n_seeds']}")


if __name__ == "__main__":
    main()
