#!/usr/bin/env python3
"""Figures E1-E4 from the CSV/JSON outputs.

  python analysis/figures.py --results results --out results/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAIR_TITLES = {"spo": "SPO vs SPO + VDRA", "treepo": "TreePO vs TreePO + VDRA", "treerl": "TreeRL vs TreeRL + VDRA"}
COLORS = {"baseline": "#7a7a7a", "hybrid": "#1f5fbf"}


def step_from_label(label: str) -> int:
    m = re.search(r"step(\d+)", label)
    return int(m.group(1)) if m else -1


def fig_curves(curves: pd.DataFrame, x: str, out: str):
    pairs = [p for p in PAIR_TITLES if p in set(curves["pair"].dropna())]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 3.6), squeeze=False)
    m = curves[curves["benchmark"] == "macro"].dropna(subset=[x])
    for ax, pair in zip(axes[0], pairs):
        for role in ("baseline", "hybrid"):
            d = m[(m["pair"] == pair) & (m["role"] == role)]
            if d.empty:
                continue
            if x == "step":
                g = d.groupby("step")["accuracy"].agg(["mean", "std"]).reset_index()
                ax.plot(g["step"], 100 * g["mean"], color=COLORS[role], label=role)
                ax.fill_between(g["step"], 100 * (g["mean"] - g["std"].fillna(0)), 100 * (g["mean"] + g["std"].fillna(0)),
                                color=COLORS[role], alpha=0.15, lw=0)
            else:
                for _, dd in d.groupby("seed"):
                    dd = dd.sort_values("gpu_hours")
                    ax.plot(dd["gpu_hours"], 100 * dd["accuracy"], color=COLORS[role], alpha=0.7,
                            label=role if _ == d["seed"].iloc[0] else None)
        ax.set_title(PAIR_TITLES[pair], fontsize=10)
        ax.set_xlabel("policy updates" if x == "step" else "cumulative GPU-hours")
        ax.set_ylabel("macro Pass@1 (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_exp1(exp1_dir: str, out: str):
    rows = []
    for p in glob.glob(os.path.join(exp1_dir, "*.exp1.json")):
        res = json.load(open(p))
        for r in res["rows"]:
            rows.append({"label": r["label"], "method": r["method"], "recovery": r["recovery"]["median"],
                         "lo": r["recovery"]["lo"], "hi": r["recovery"]["hi"],
                         "aux": r.get("aux_generated_tokens", np.nan)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["step"] = df["label"].map(step_from_label)
    df["policy"] = df["label"].str.extract(r"^(R\d)")[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for (method, pol), d in df.groupby(["method", "policy"]):
        if method in ("uniform", "oracle"):
            continue
        d = d.sort_values("step")
        a1.plot(d["step"], d["recovery"], ls="-" if pol == "R2" else "--", marker="o", ms=3,
                label=f"{method} ({pol})")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xlabel("training step of frozen policy")
    a1.set_ylabel("oracle-gap recovery")
    a1.legend(fontsize=7, ncol=2)
    a1.grid(alpha=0.3)
    last = df[df["step"] == df["step"].max()]
    for _, r in last.iterrows():
        if r["method"] in ("uniform", "oracle") or not np.isfinite(r["aux"]):
            continue
        a2.errorbar(max(r["aux"], 1), r["recovery"], yerr=[[r["recovery"] - r["lo"]], [r["hi"] - r["recovery"]]],
                    fmt="o", capsize=3)
        a2.annotate(f"{r['method']} ({r['policy']})", (max(r["aux"], 1), r["recovery"]), fontsize=7)
    a2.set_xscale("log")
    a2.set_xlabel("auxiliary generated tokens per node")
    a2.set_ylabel("oracle-gap recovery")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_exp2(exp2_dir: str, out: str):
    rows = []
    for p in glob.glob(os.path.join(exp2_dir, "*.exp2.json")):
        res = json.load(open(p))
        for r in res["rows"]:
            rows.append({"label": r["label"], "method": r["method"],
                         "L_own": r["L_own_all"]["median"], "L_trained": r["L_trained"]["median"],
                         "pred": r["pred_d1_d2_leaf"]["median"], "meas": r["meas_d1_d2_leaf"]["median"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["step"] = df["label"].map(step_from_label)
    df["policy"] = df["label"].str.extract(r"^(R\d)")[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for (method, pol), d in df.groupby(["method", "policy"]):
        d = d.sort_values("step")
        a1.plot(d["step"], d["L_own"], ls="-" if pol == "R2" else "--", marker="o", ms=3, label=f"{method} ({pol})")
    a1.set_xlabel("training step of frozen policy")
    a1.set_ylabel("edge-averaged advantage MSE")
    a1.legend(fontsize=7, ncol=2)
    a1.grid(alpha=0.3)
    a2.scatter(df["pred"], df["meas"], s=12)
    lim = [min(df["pred"].min(), df["meas"].min()), max(df["pred"].max(), df["meas"].max())]
    a2.plot(lim, lim, color="k", lw=0.8)
    a2.set_xlabel("predicted (Prop. 1-2)")
    a2.set_ylabel("measured")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cpath = os.path.join(a.results, "stats", "curves.csv")
    if os.path.exists(cpath):
        curves = pd.read_csv(cpath)
        fig_curves(curves, "step", os.path.join(a.out, "E3_curves_matched_updates.png"))
        fig_curves(curves, "gpu_hours", os.path.join(a.out, "E4_curves_gpu_hours.png"))
    fig_exp1(os.path.join(a.results, "exp1"), os.path.join(a.out, "E1_proxy_quality.png"))
    fig_exp2(os.path.join(a.results, "exp2"), os.path.join(a.out, "E2_advantage_error.png"))
    print(f"[figures] written to {a.out}")


if __name__ == "__main__":
    main()
