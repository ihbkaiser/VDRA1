"""Paired prompt bootstrap for saved IGPO MATH evaluation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import cluster_bootstrap


def _load_prediction_dataset(root: str | Path):
    from datasets import Dataset

    root = Path(root)
    candidates = [root] if (root / "state.json").is_file() else []
    candidates.extend(path.parent for path in root.rglob("state.json"))
    datasets = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            dataset = Dataset.load_from_disk(str(resolved))
        except Exception:
            continue
        if "_treetune__candidate_answers" in dataset.column_names:
            datasets.append((resolved, dataset))
    if len(datasets) != 1:
        found = [str(path) for path, _ in datasets]
        raise ValueError(
            f"expected exactly one saved prediction dataset under {root}, found {found}"
        )
    return datasets[0][1]


def _verifier(evaluator_root: str | Path):
    evaluator_root = str(Path(evaluator_root).resolve())
    if evaluator_root not in sys.path:
        sys.path.insert(0, evaluator_root)
    from treetune.tasks.math import MATH

    # These settings match the three selected benchmark task configs, all of
    # which inherit math_inplace_no_answer_prefix.jsonnet.
    task = object.__new__(MATH)
    task.use_minerva_few_shot_prompt = False
    task.answer_prefix = None
    task.use_dataset_answer = True
    return task


def _prompt_id(reference: Mapping[str, Any]) -> str:
    problem = str(reference.get("problem", reference.get("question", "")))
    answer = reference.get("answer", "")
    payload = json.dumps([problem, answer], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_dataset(dataset, *, evaluator_root: str | Path, samples: int = 16) -> list[dict[str, Any]]:
    if "_treetune__candidate_answers" not in dataset.column_names:
        raise ValueError("saved results lack _treetune__candidate_answers")
    task = _verifier(evaluator_root)
    output = []
    for index, reference in enumerate(dataset):
        predictions = reference["_treetune__candidate_answers"]
        if len(predictions) != samples:
            raise ValueError(
                f"row {index} has {len(predictions)} samples; expected {samples}"
            )
        metrics = task.evaluate_predictions(predictions=[predictions], references=[reference])
        output.append({
            "prompt_id": _prompt_id(reference),
            "row_index": index,
            "avg": float(metrics[f"avg@{samples}"]),
            "pass": float(metrics[f"pass@{samples}"]),
        })
    ids = [row["prompt_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt identities are not unique within a dataset")
    return output


def paired_bootstrap(
    policy_a: Sequence[Mapping[str, Any]],
    policy_b: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    policy_a_name: str = "spo",
    policy_b_name: str = "vdra",
    samples: int = 16,
    replicates: int = 1000,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_a = {row["prompt_id"]: row for row in policy_a}
    by_b = {row["prompt_id"]: row for row in policy_b}
    if not by_a or by_a.keys() != by_b.keys():
        raise ValueError("paired policies must contain the same nonempty prompt identities")
    rows = []
    for prompt_id in sorted(by_a):
        a, b = by_a[prompt_id], by_b[prompt_id]
        rows.append({
            "dataset": dataset_name,
            "prompt_id": prompt_id,
            "policy_a": policy_a_name,
            "policy_b": policy_b_name,
            "avg_a": float(a["avg"]),
            "avg_b": float(b["avg"]),
            "avg_diff_b_minus_a": float(b["avg"]) - float(a["avg"]),
            "pass_a": float(a["pass"]),
            "pass_b": float(b["pass"]),
            "pass_diff_b_minus_a": float(b["pass"]) - float(a["pass"]),
        })

    summaries = []
    for metric in ("avg", "pass"):
        for side, name in (("a", policy_a_name), ("b", policy_b_name)):
            interval = cluster_bootstrap(
                rows,
                value_key=f"{metric}_{side}",
                replicates=replicates,
                seed=seed,
            )
            summaries.append({
                "dataset": dataset_name,
                "metric": f"{metric}@{samples}",
                "policy": name,
                "estimate": interval["estimate"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "n_prompts": interval["n_clusters"],
                "bootstrap_replicates": replicates,
                "seed": seed,
            })
        diff = cluster_bootstrap(
            rows,
            value_key=f"{metric}_diff_b_minus_a",
            replicates=replicates,
            seed=seed,
        )
        summaries.append({
            "dataset": dataset_name,
            "metric": f"{metric}@{samples}",
            "policy": f"{policy_b_name}-{policy_a_name}",
            "estimate": diff["estimate"],
            "ci_low": diff["ci_low"],
            "ci_high": diff["ci_high"],
            "n_prompts": diff["n_clusters"],
            "bootstrap_replicates": replicates,
            "seed": seed,
        })
    return rows, summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    spo_root: str | Path,
    vdra_root: str | Path,
    evaluator_root: str | Path,
    output_dir: str | Path,
    datasets: Sequence[str] = ("aime24", "aime25", "amc23"),
    samples: int = 16,
    replicates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    all_rows = []
    all_summaries = []
    for name in datasets:
        spo = score_dataset(
            _load_prediction_dataset(Path(spo_root) / name),
            evaluator_root=evaluator_root,
            samples=samples,
        )
        vdra = score_dataset(
            _load_prediction_dataset(Path(vdra_root) / name),
            evaluator_root=evaluator_root,
            samples=samples,
        )
        rows, summaries = paired_bootstrap(
            spo,
            vdra,
            dataset_name=name,
            samples=samples,
            replicates=replicates,
            seed=seed,
        )
        all_rows.extend(rows)
        all_summaries.extend(summaries)
    _write_csv(output_dir / "a7_per_prompt.csv", all_rows)
    _write_csv(output_dir / "a7_paired_bootstrap.csv", all_summaries)
    return {
        "datasets": list(datasets),
        "prompt_count": len(all_rows),
        "samples": samples,
        "bootstrap_replicates": replicates,
        "seed": seed,
        "per_prompt_csv": str(output_dir / "a7_per_prompt.csv"),
        "summary_csv": str(output_dir / "a7_paired_bootstrap.csv"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spo-root", required=True)
    parser.add_argument("--vdra-root", required=True)
    parser.add_argument("--evaluator-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="aime24,aime25,amc23")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = run(
        spo_root=args.spo_root,
        vdra_root=args.vdra_root,
        evaluator_root=args.evaluator_root,
        output_dir=args.output_dir,
        datasets=args.datasets.split(","),
        samples=args.samples,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
