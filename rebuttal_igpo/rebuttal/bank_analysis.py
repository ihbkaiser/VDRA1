"""CPU analyses for the frozen-policy node bank (A2/A4/A5)."""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .alloc_replay import allocate, allocate_gear_as_trained, repair_counts
from .metrics import (
    allocation_summary,
    calibration_bins,
    cluster_bootstrap,
    j_objective,
    spearman,
    top_quartile_overlap,
)


def iter_bank(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield complete gzip JSONL records; call only after collection finishes."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or "node_id" not in row:
                    raise ValueError(f"invalid bank record at line {line_number}")
                yield row


def _sample_variance(values: Sequence[Any]) -> float:
    xs = [float(value) for value in values]
    if len(xs) < 2:
        return float("nan")
    return statistics.variance(xs)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _prompt_groups(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("prompt_id", "unknown"))].append(record)
    return dict(sorted(groups.items()))


def a2_noise_sensitivity(
    records: Iterable[Mapping[str, Any]],
    *,
    noise_levels: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    repetitions: int = 50,
    seed: int = 42,
    bootstrap_repetitions: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Perturb proxy C lognormally and evaluate allocation on independent R_b.

    Allocations use equal per-prompt budget 6 * parent_count, lower bound 1,
    and no cap. C is the source estimator's proxy; the objective is evaluated
    on unbiased sample variances from R_b. Each prompt is the bootstrap cluster.
    """
    if repetitions < 1 or bootstrap_repetitions < 1:
        raise ValueError("repetitions must be positive")
    groups = _prompt_groups(records)
    if not groups:
        raise ValueError("bank is empty")
    rng = random.Random(seed)
    per_prompt: list[dict[str, Any]] = []
    for prompt_id, group in groups.items():
        group = sorted(group, key=lambda row: (int(row.get("depth", 0)), str(row["node_id"])))
        proxy = [max(float(row.get("proxy", {}).get("sigma2", 0.0) or 0.0), 0.0) for row in group]
        target = [_sample_variance(row.get("R_b", {}).get("rewards", [])) for row in group]
        if any(not math.isfinite(value) for value in target):
            raise ValueError(f"R_b needs at least two rewards per node ({prompt_id})")
        budget = 6 * len(group)
        baseline = allocate(proxy, lower=1, upper=None, budget=budget, rule="eq7_exact")
        for sigma in noise_levels:
            if sigma < 0:
                raise ValueError("noise levels must be nonnegative")
            distances: list[float] = []
            objectives: list[float] = []
            for _ in range(repetitions):
                noisy = [c * math.exp(rng.gauss(0.0, float(sigma))) for c in proxy]
                counts = allocate(noisy, lower=1, upper=None, budget=budget, rule="eq7_exact")
                distances.append(sum(abs(a - b) for a, b in zip(baseline, counts)) / (2 * budget))
                objectives.append(sum(c / n for c, n in zip(target, counts)))
            per_prompt.append({
                "prompt_id": prompt_id,
                "noise_sigma": float(sigma),
                "mean_rollout_share_reassigned": statistics.mean(distances),
                "mean_heldout_J_Rb": statistics.mean(objectives),
                "repetitions": repetitions,
                "budget": budget,
                "parent_count": len(group),
            })
    summary: list[dict[str, Any]] = []
    for sigma in noise_levels:
        rows = [row for row in per_prompt if row["noise_sigma"] == float(sigma)]
        distance_ci = cluster_bootstrap(
            rows, value_key="mean_rollout_share_reassigned", replicates=bootstrap_repetitions,
            seed=seed + int(float(sigma) * 1000),
        )
        objective_ci = cluster_bootstrap(
            rows, value_key="mean_heldout_J_Rb", replicates=bootstrap_repetitions,
            seed=seed + 10_000 + int(float(sigma) * 1000),
        )
        summary.append({
            "noise_sigma": float(sigma),
            "mean_rollout_share_reassigned": distance_ci["estimate"],
            "reassigned_ci_low": distance_ci["ci_low"],
            "reassigned_ci_high": distance_ci["ci_high"],
            "mean_heldout_J_Rb": objective_ci["estimate"],
            "heldout_J_ci_low": objective_ci["ci_low"],
            "heldout_J_ci_high": objective_ci["ci_high"],
            "n_prompt_clusters": distance_ci["n_clusters"],
            "repetitions_per_prompt": repetitions,
            "budget_rule": "6 * nodes per prompt; lower=1; uncapped",
            "allocation_rule": "eq7_exact on noisy proxy; evaluated on R_b sample variance",
        })
    return per_prompt, summary


def a4_tv_value_calibration(
    records: Iterable[Mapping[str, Any]], *, bins: int = 10
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare source proxy pair TV with absolute held-out child-value gaps."""
    pairs: list[dict[str, Any]] = []
    for record in records:
        proxy = record.get("proxy", {})
        children = record.get("child_values", [])
        means = [
            statistics.mean(map(float, child["rewards"])) if child.get("rewards") else None
            for child in children
        ]
        for item in proxy.get("pair_tvs", []):
            i, j = int(item["i"]), int(item["j"])
            if i >= len(means) or j >= len(means) or means[i] is None or means[j] is None:
                continue
            rewards_i = list(map(float, children[i]["rewards"]))
            rewards_j = list(map(float, children[j]["rewards"]))
            se_gap = math.sqrt(
                _sample_variance(rewards_i) / len(rewards_i)
                + _sample_variance(rewards_j) / len(rewards_j)
            )
            pairs.append({
                "prompt_id": record.get("prompt_id"),
                "node_id": record.get("node_id"),
                "i": i,
                "j": j,
                "tv": float(item["tv"]),
                "absolute_value_gap": abs(means[i] - means[j]),
                "signed_value_gap": means[i] - means[j],
                "child_mean_i": means[i],
                "child_mean_j": means[j],
                "child_rewards_i": len(children[i].get("rewards", [])),
                "child_rewards_j": len(children[j].get("rewards", [])),
                "absolute_gap_se_normal_approx": se_gap,
                "gap_noise_band_low": max(0.0, abs(means[i] - means[j]) - 1.96 * se_gap),
                "gap_noise_band_high": abs(means[i] - means[j]) + 1.96 * se_gap,
            })
    if not pairs:
        raise ValueError("bank contains no comparable proxy/child pairs")
    x = [row["tv"] for row in pairs]
    y = [row["absolute_value_gap"] for row in pairs]
    summary = {
        "n_pairs": len(pairs),
        "spearman_tv_vs_absolute_value_gap": spearman(x, y) if len(pairs) >= 2 else float("nan"),
        "top_quartile_overlap": top_quartile_overlap(x, y),
        "calibration_bins": calibration_bins(x, y, bins=bins),
        "target_definition": "absolute difference of two candidate child means from their fixed 16-completion reward samples",
        "independence_note": "child sample values are reused for this descriptive calibration; this is not an independent test set",
    }
    return pairs, summary


def _mean(values: Sequence[Any]) -> float:
    return statistics.mean(map(float, values))


def _heldout_edge_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Squared discrepancy between disjoint child and parent MC halves.

    This fixed-edge diagnostic does not use allocation counts, so it is the
    same across allocation rules. It measures split-half MC disagreement,
    not error against a known ground truth.
    """
    ra = list(map(float, record.get("R_a", {}).get("rewards", [])))
    rb = list(map(float, record.get("R_b", {}).get("rewards", [])))
    if not ra or not rb:
        return []
    parent_a, parent_b = _mean(ra), _mean(rb)
    rows = []
    for index, child in enumerate(record.get("child_values", [])):
        values = list(map(float, child.get("rewards", [])))
        left, right = values[::2], values[1::2]
        if not left or not right:
            continue
        advantage_hat = _mean(left) - parent_b
        advantage_ref = _mean(right) - parent_a
        rows.append({
            "prompt_id": record.get("prompt_id"),
            "node_id": record.get("node_id"),
            "candidate_index": index,
            "child_mean_hat": _mean(left),
            "child_mean_reference": _mean(right),
            "advantage_hat": advantage_hat,
            "advantage_reference": advantage_ref,
            "squared_error": (advantage_hat - advantage_ref) ** 2,
            "child_samples_per_half": min(len(left), len(right)),
        })
    return rows


def _prompt_allocations(group: Sequence[Mapping[str, Any]]) -> tuple[int, list[int], dict[str, list[int]]]:
    ordered = sorted(group, key=lambda row: (int(row.get("depth", 0)), str(row["node_id"])))
    n = len(ordered)
    candidate_counts = [len(row.get("proxy", {}).get("candidates", [])) for row in ordered]
    caps_k = [min(6, max(int(k), 0)) for k in candidate_counts]
    lower = [1 if cap > 0 else 0 for cap in caps_k]
    budget = min(6 * n, sum(caps_k))
    if budget < sum(lower):
        raise ValueError("candidate-bounded prompt budget cannot cover all represented nodes")

    proxy = [max(float(row.get("proxy", {}).get("sigma2", 0.0) or 0.0), 0.0) for row in ordered]
    raw_tv = [
        statistics.mean(float(pair["tv"]) for pair in row.get("proxy", {}).get("pair_tvs", []))
        if row.get("proxy", {}).get("pair_tvs") else 0.0
        for row in ordered
    ]
    entropy = [float(row.get("proxy", {}).get("pilot_entropy_mean_top5") or 0.0) for row in ordered]
    nll = [float(row.get("proxy", {}).get("pilot_nll_mean") or 0.0) for row in ordered]
    oracle = [_sample_variance(row.get("R_a", {}).get("rewards", [])) for row in ordered]
    oracle = [0.0 if not math.isfinite(x) else max(x, 0.0) for x in oracle]
    parent_counts = [1 if int(row.get("depth", 0)) == 0 else 6 for row in ordered]
    vdra_lambda = [c * (p - 1) / p for c, p in zip(proxy, parent_counts)]
    regime_caps: dict[str, list[int | None]] = {
        "trained": [None] * n,
        "capped": [6] * n,
        "capped_k": caps_k,
    }
    allocations: dict[str, list[int]] = {}
    source_defaults = {"lambda_": 0.005, "n_min": 2, "distribute_remainder": True, "fallback_uniform": True}
    for regime, upper in regime_caps.items():
        for strategy, coeffs in (
            ("uniform", [1.0] * n),
            ("fixed_branch", proxy),
            ("eq7_exact", proxy),
            ("vdra_lambda", vdra_lambda),
            ("oracle", oracle),
            ("entropy", entropy),
            ("nll", nll),
            ("raw_tv", raw_tv),
        ):
            allocation = allocate(coeffs, lower=lower, upper=upper, budget=budget, rule=strategy)
            allocations[f"{regime}:{strategy}"] = allocation
        source_counts = allocate_gear_as_trained(
            proxy,
            budget,
            node_ids=[str(row["node_id"]) for row in ordered],
            **source_defaults,
        )
        source_counts, _ = repair_counts(source_counts, lower, upper, budget)
        allocations[f"{regime}:gear_online_snapshot_defaults"] = source_counts
    return budget, lower, allocations


def a5_allocation_errors(
    records: Iterable[Mapping[str, Any]], *, bootstrap_repetitions: int = 1000, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute A5 variance/error diagnostics from disjoint R_a/R_b bank halves.

    The source-default GEAR comparator is explicitly not a recovered run
    allocation: the archived checkpoint's resolved allocator config is absent.
    """
    groups = _prompt_groups(records)
    if not groups:
        raise ValueError("bank is empty")
    per_prompt: list[dict[str, Any]] = []
    for prompt_id, group in groups.items():
        ordered = sorted(group, key=lambda row: (int(row.get("depth", 0)), str(row["node_id"])))
        budget, _, allocations = _prompt_allocations(ordered)
        target_variances = [_sample_variance(row.get("R_b", {}).get("rewards", [])) for row in ordered]
        means_a = [_mean(row.get("R_a", {}).get("rewards", [])) for row in ordered]
        means_b = [_mean(row.get("R_b", {}).get("rewards", [])) for row in ordered]
        if any(not math.isfinite(value) for value in target_variances):
            raise ValueError(f"R_b needs at least two rewards per node ({prompt_id})")
        heldout_rows = [edge for row in ordered for edge in _heldout_edge_rows(row)]
        heldout_errors = [float(edge["squared_error"]) for edge in heldout_rows]
        fixed_heldout = statistics.mean(heldout_errors) if heldout_errors else float("nan")
        proxy_coeffs = [max(float(row.get("proxy", {}).get("sigma2", 0.0) or 0.0), 0.0) for row in ordered]
        for label, counts in allocations.items():
            regime, strategy = label.split(":", 1)
            uniform_counts = allocations[f"{regime}:uniform"]
            oracle_counts = allocations[f"{regime}:oracle"]
            node_errors = [
                (variance / count + (mean_b - mean_a) ** 2) if count > 0 else float("inf")
                for variance, count, mean_a, mean_b in zip(target_variances, counts, means_a, means_b)
            ]
            allocated = sum(counts)
            edge_loss = (
                sum(count * error for count, error in zip(counts, node_errors)) / allocated
                if allocated else float("inf")
            )
            proxy_j = sum(
                (variance / count if count else (float("inf") if variance > 0 else 0.0))
                for variance, count in zip(target_variances, counts)
            )
            comparison = allocation_summary(target_variances, counts, uniform_counts, oracle_counts)
            per_prompt.append({
                "prompt_id": prompt_id,
                "bound_regime": regime,
                "strategy": strategy,
                "budget": budget,
                "allocated": allocated,
                "node_count": len(ordered),
                "zero_allocation_nodes": sum(count == 0 for count in counts),
                "L_node": statistics.mean(node_errors),
                "L_edge": edge_loss,
                "L_heldout": fixed_heldout,
                "J_Rb": proxy_j,
                "percent_vs_uniform": comparison["percent_vs_uniform"],
                "oracle_gap_recovery": comparison["oracle_gap_recovery"],
                "oracle_regret": comparison["oracle_regret"],
                "rollout_share_reassigned_vs_uniform": sum(
                    abs(a - b) for a, b in zip(counts, uniform_counts)
                ) / (2 * max(sum(counts), sum(uniform_counts), 1)),
                "spearman_proxy_vs_Rb_variance": spearman(proxy_coeffs, target_variances) if len(proxy_coeffs) >= 2 else float("nan"),
                "allocations_json": json.dumps(counts, separators=(",", ":")),
                "L_heldout_note": "fixed child-edge split-half disagreement; independent of allocation; not ground truth",
                "gear_comparator_note": "source online defaults lambda=0.005,n_min=2,remainder+uniform fallback; not recovered run settings",
            })
    summary: list[dict[str, Any]] = []
    keys = sorted({(row["bound_regime"], row["strategy"]) for row in per_prompt})
    for regime, strategy in keys:
        rows = [row for row in per_prompt if row["bound_regime"] == regime and row["strategy"] == strategy]
        result: dict[str, Any] = {
            "bound_regime": regime,
            "strategy": strategy,
            "n_prompt_clusters": len(groups),
            "bootstrap_repetitions": bootstrap_repetitions,
        }
        for metric in (
            "L_node", "L_edge", "L_heldout", "J_Rb", "percent_vs_uniform",
            "oracle_gap_recovery", "oracle_regret", "rollout_share_reassigned_vs_uniform",
            "spearman_proxy_vs_Rb_variance",
        ):
            ci = cluster_bootstrap(
                rows,
                value_key=metric,
                replicates=bootstrap_repetitions,
                seed=seed + len(summary) * 17 + len(metric),
            )
            result[f"{metric}_estimate"] = ci["estimate"]
            result[f"{metric}_ci_low"] = ci["ci_low"]
            result[f"{metric}_ci_high"] = ci["ci_high"]
        summary.append(result)
    return per_prompt, summary


def run_a2(bank_path: str | Path, output_dir: str | Path, *, seed: int = 42) -> dict[str, Any]:
    out = Path(output_dir)
    per_prompt, summary = a2_noise_sensitivity(iter_bank(bank_path), seed=seed)
    _write_csv(out / "a2_noise_per_prompt.csv", per_prompt)
    _write_csv(out / "a2_noise_sensitivity.csv", summary)
    return {"per_prompt_rows": len(per_prompt), "summary_rows": len(summary)}


def run_a4(bank_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    pairs, summary = a4_tv_value_calibration(iter_bank(bank_path))
    _write_csv(out / "a4_tv_value_pairs.csv", pairs)
    (out / "a4_tv_value_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"pair_rows": len(pairs)}


def run_a5(bank_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    records = list(iter_bank(bank_path))
    per_prompt, summary = a5_allocation_errors(records)
    edge_rows = [edge for record in records for edge in _heldout_edge_rows(record)]
    _write_csv(out / "a5_allocation_per_prompt.csv", per_prompt)
    _write_csv(out / "a5_allocation_summary.csv", summary)
    _write_csv(out / "a5_heldout_edges.csv", edge_rows)
    return {"per_prompt_rows": len(per_prompt), "summary_rows": len(summary), "heldout_edge_rows": len(edge_rows)}
