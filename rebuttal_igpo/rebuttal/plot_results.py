"""Reproducible, publication-style figures for the compact A2--A5 results."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import spearman


PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#8C8C8C",
    "black": "#333333",
}


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def _save(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=300)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_a3_proxy_variance(bank_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    with gzip.open(bank_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rewards = [float(value) for value in row["R_b"]["rewards"]]
            if len(rewards) < 2:
                continue
            prompt_id = str(row["prompt_id"])
            records.append({
                "prompt_hash": hashlib.sha256(prompt_id.encode("utf-8")).hexdigest(),
                "depth": int(row["depth"]),
                "proxy_sigma2": float(row["proxy"]["sigma2"]),
                "rb_sample_variance": statistics.variance(rewards),
            })
    if not records:
        raise ValueError(f"no complete A3 records in {bank_path}")

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_prompt[row["prompt_hash"]].append(row)
    prompt_hashes = sorted(by_prompt)
    point = spearman(
        [row["proxy_sigma2"] for row in records],
        [row["rb_sample_variance"] for row in records],
    )
    rng = random.Random(42)
    bootstrap = []
    for _ in range(1000):
        sampled = [prompt_hashes[rng.randrange(len(prompt_hashes))] for _ in prompt_hashes]
        sample_rows = [row for prompt_hash in sampled for row in by_prompt[prompt_hash]]
        value = spearman(
            [row["proxy_sigma2"] for row in sample_rows],
            [row["rb_sample_variance"] for row in sample_rows],
        )
        if math.isfinite(value):
            bootstrap.append(value)
    bootstrap.sort()
    ci_low = bootstrap[max(0, int(0.025 * len(bootstrap)))] if bootstrap else float("nan")
    ci_high = bootstrap[min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))] if bootstrap else float("nan")
    stats = {
        "n_parents": len(records),
        "n_prompt_clusters": len(prompt_hashes),
        "spearman_proxy_vs_Rb_variance": point,
        "spearman_prompt_bootstrap_ci_low": ci_low,
        "spearman_prompt_bootstrap_ci_high": ci_high,
        "bootstrap_replicates": len(bootstrap),
        "note": "R_b sample variance is a finite-MC reference; bootstrap resamples prompts.",
    }
    return records, stats


def _plot_a2(path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(_read_csv(path), key=lambda row: float(row["noise_sigma"]))
    x = [float(row["noise_sigma"]) for row in rows]
    j = [float(row["mean_heldout_J_Rb"]) for row in rows]
    j_low = [value - float(row["heldout_J_ci_low"]) for value, row in zip(j, rows)]
    j_high = [float(row["heldout_J_ci_high"]) - value for value, row in zip(j, rows)]
    churn = [100 * float(row["mean_rollout_share_reassigned"]) for row in rows]
    c_low = [100 * (float(row["mean_rollout_share_reassigned"]) - float(row["reassigned_ci_low"])) for row in rows]
    c_high = [100 * (float(row["reassigned_ci_high"]) - float(row["mean_rollout_share_reassigned"])) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7), constrained_layout=True)
    axes[0].errorbar(x, j, yerr=[j_low, j_high], color=PALETTE["blue"], marker="o", capsize=3, linewidth=1.5)
    axes[0].set(xlabel=r"Proxy noise scale $\sigma$", ylabel=r"Held-out MC objective $J_{R_b}$", title="(a) Outcome objective")
    axes[1].errorbar(x, churn, yerr=[c_low, c_high], color=PALETTE["orange"], marker="s", capsize=3, linewidth=1.5)
    axes[1].set(xlabel=r"Proxy noise scale $\sigma$", ylabel="Rollout share reassigned (%)", title="(b) Allocation change")
    for ax in axes:
        ax.set_xticks(x)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    _save(fig, output_dir, "fig_A2")
    plt.close(fig)


def _plot_a3(records: list[dict[str, Any]], stats: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    styles = {
        0: (PALETTE["blue"], "o", "Root"),
        1: (PALETTE["orange"], "^", "Depth 1"),
    }
    for depth, (color, marker, label) in styles.items():
        selected = [row for row in records if row["depth"] == depth]
        if selected:
            ax.scatter(
                [row["proxy_sigma2"] for row in selected],
                [row["rb_sample_variance"] for row in selected],
                s=22, color=color, marker=marker, alpha=0.72,
                edgecolors="white", linewidths=0.35, label=label,
            )
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel(r"GEAR proxy $\hat{\sigma}^2$")
    ax.set_ylabel(r"Independent $R_b$ reward variance")
    ax.text(
        0.98, 0.97,
        "Spearman %.2f\n95%% prompt CI [%.2f, %.2f]" % (
            stats["spearman_proxy_vs_Rb_variance"],
            stats["spearman_prompt_bootstrap_ci_low"],
            stats["spearman_prompt_bootstrap_ci_high"],
        ),
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(True, which="major", alpha=0.18)
    ax.grid(False, which="minor")
    _save(fig, output_dir, "fig_A3")
    plt.close(fig)


def _plot_a4(pairs_path: Path, summary_path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(pairs_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    x = [float(row["tv"]) for row in rows]
    y = [float(row["absolute_value_gap"]) for row in rows]
    bins = summary["calibration_bins"]

    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    ax.scatter(x, y, color=PALETTE["gray"], s=10, alpha=0.24, edgecolors="none", label="Candidate pairs")
    ax.plot(
        [float(row["mean_proxy"]) for row in bins],
        [float(row["mean_target"]) for row in bins],
        color=PALETTE["orange"], marker="o", markersize=4, linewidth=1.5,
        label="TV-rank decile means",
    )
    ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_xlabel("GEAR pairwise total variation")
    ax.set_ylabel("Absolute gap in candidate reward means")
    ax.text(
        0.98, 0.97,
        "Spearman %.2f\nTop-quartile overlap %.1f%%" % (
            summary["spearman_tv_vs_absolute_value_gap"],
            100 * summary["top_quartile_overlap"],
        ),
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(True, which="major", alpha=0.18)
    ax.grid(False, which="minor")
    _save(fig, output_dir, "fig_A4")
    plt.close(fig)


def _plot_a5(path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(path)
    selected = [
        ("uniform", "Uniform", PALETTE["gray"], "o"),
        ("eq7_exact", "Eq. (7) exact", PALETTE["blue"], "s"),
        ("gear_online_snapshot_defaults", "GEAR snapshot defaults", PALETTE["orange"], "^"),
        ("oracle", "Oracle (R_a)", PALETTE["green"], "D"),
    ]
    regimes = ["trained", "capped", "capped_k"]
    titles = ["Uncapped", "Cap at 6", "Candidate-bounded"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.25), constrained_layout=True)
    metrics = [
        ("J_Rb", r"Held-out MC objective $J_{R_b}$"),
        ("L_node", r"Node-value error proxy $L_{node}$"),
    ]
    offsets = [-0.27, -0.09, 0.09, 0.27]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for x_index, regime in enumerate(regimes):
            for (strategy, label, color, marker), offset in zip(selected, offsets):
                row = next(r for r in rows if r["bound_regime"] == regime and r["strategy"] == strategy)
                estimate = float(row[f"{metric}_estimate"])
                low = float(row[f"{metric}_ci_low"])
                high = float(row[f"{metric}_ci_high"])
                ax.errorbar(
                    x_index + offset, estimate,
                    yerr=[[max(0.0, estimate - low)], [max(0.0, high - estimate)]],
                    color=color, marker=marker, markersize=4, capsize=2.5,
                    linewidth=1.0, linestyle="none", label=label if x_index == 0 else None,
                )
        ax.set_xticks(range(len(regimes)), titles, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    axes[1].legend(frameon=False, loc="upper right", fontsize=7.2)
    _save(fig, output_dir, "fig_A5")
    plt.close(fig)


def make_figures(*, results_root: str | Path, bank_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create A2--A5 figures and the text-free numeric A3 scatter source."""
    _style()
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    a3_records, a3_stats = _load_a3_proxy_variance(Path(bank_path))
    _write_csv(results_root / "a3_iter100" / "analysis" / "a3_proxy_variance.csv", a3_records)
    (results_root / "a3_iter100" / "analysis" / "a3_proxy_variance_summary.json").write_text(
        json.dumps(a3_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_a2(results_root / "a3_iter100" / "analysis" / "a2_noise_sensitivity.csv", output_dir)
    _plot_a3(a3_records, a3_stats, output_dir)
    _plot_a4(
        results_root / "a3_iter100" / "analysis" / "a4_tv_value_pairs.csv",
        results_root / "a3_iter100" / "analysis" / "a4_tv_value_summary.json",
        output_dir,
    )
    _plot_a5(results_root / "a3_iter100" / "analysis" / "a5_allocation_summary.csv", output_dir)
    return {"a3": a3_stats, "figures": [f"fig_A{i}" for i in range(2, 6)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(Path(__file__).parents[1] / "results"))
    parser.add_argument("--bank", default="/workspace/rebuttal_out/a3_iter100/bank.jsonl.gz")
    parser.add_argument("--output-dir", default=str(Path(__file__).parents[1] / "results" / "figures"))
    args = parser.parse_args(argv)
    print(json.dumps(make_figures(results_root=args.results_root, bank_path=args.bank, output_dir=args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
