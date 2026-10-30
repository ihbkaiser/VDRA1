"""Streaming readers for GEAR, InGPO and VDRA run artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


_FIELD_ALIASES = {
    "segment_id": ("gear_segment_id", "ingpo_segment_id", "vdra_segment_id", "segment_id", "id"),
    "reward_variance": (
        "gear_reward_variance", "gear_sigma2", "ingpo_reward_variance", "ingpo_sigma2",
        "vdra_reward_variance", "vdra_sigma2",
    ),
    "allocated_branch_factor": (
        "gear_allocated_branch_factor", "ingpo_allocated_branch_factor", "vdra_allocated_branch_factor",
    ),
    "requested_branch_factor": (
        "gear_default_branch_factor", "ingpo_default_branch_factor", "vdra_default_branch_factor",
        "default_branch_factor", "branch_factor",
    ),
    "budget_weight": ("gear_budget_weight", "ingpo_budget_weight", "vdra_budget_weight"),
    "queue_id": ("gear_budget_queue_id", "ingpo_budget_queue_id", "vdra_budget_queue_id"),
    "candidates": ("gear_budget_candidates", "ingpo_budget_candidates", "vdra_budget_candidates"),
    "discarded_candidates": (
        "gear_discarded_budget_candidates", "ingpo_discarded_budget_candidates",
        "vdra_discarded_budget_candidates",
    ),
}
_MATRIX_KEYS = {"ingpo_tv_logp_matrix", "gear_tv_logp_matrix", "vdra_tv_logp_matrix"}


def _decode_tree(value: Any) -> Any:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"reasoning tree must be a mapping, got {type(value).__name__}")
    return value


def _tree_from_row(row: Any) -> Any | None:
    if isinstance(row, Mapping):
        value = row.get("_treetune__reasoning_tree", row.get("reasoning_tree"))
        if value is None and "children" in row:
            value = row
        if value is None:
            return None
        return _decode_tree(value)
    if isinstance(row, str):
        return _decode_tree(row)
    return None


def _iter_json_rows(path: Path) -> Iterator[Any]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, list):
            yield from payload
        else:
            yield payload


def _iter_dataset_rows(path: Path) -> Iterator[Any]:
    """Iterate rows from a Dataset/DatasetDict saved with HF datasets."""
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("reading an HF saved dataset requires `datasets`") from exc
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        for split in sorted(dataset):
            yield from dataset[split]
    else:
        yield from dataset


def _iter_records(source: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(source)
    if path.is_dir():
        if (path / "state.json").exists() or (path / "dataset_dict.json").exists():
            yield from _iter_dataset_rows(path)
            return
        files = sorted(p for p in path.rglob("*") if p.is_file())
        for file_path in files:
            if file_path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                for row in _iter_json_rows(file_path):
                    if isinstance(row, dict):
                        yield row
        return
    if path.suffix.lower() in {".arrow", ".parquet"}:
        yield from _iter_dataset_rows(path)
        return
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            yield from csv.DictReader(stream)
        return
    for row in _iter_json_rows(path):
        if isinstance(row, dict):
            yield row


def _iteration_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    root = run_dir / "temp_episodes" if (run_dir / "temp_episodes").is_dir() else run_dir
    found = []
    for path in root.glob("iteration__*"):
        match = re.search(r"iteration__(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda item: (item[0], item[1].as_posix()))


def iter_trees(
    run_dir: str | Path,
    iters: Iterable[int] | None = None,
    max_per_iter: int | None = None,
) -> Iterator[tuple[int, str, int, dict[str, Any]]]:
    """Yield `(iteration, shard, row_index, tree)` without retaining a run in RAM."""
    wanted = set(iters) if iters is not None else None
    for iteration, iter_dir in _iteration_dirs(Path(run_dir)):
        if wanted is not None and iteration not in wanted:
            continue
        emitted = 0
        infer_root = iter_dir / "infer_results"
        if not infer_root.exists():
            continue
        dataset_paths = sorted(infer_root.rglob("traj_results_ds"))
        seen_files: set[Path] = set()
        for dataset_path in dataset_paths:
            for idx, row in enumerate(_iter_dataset_rows(dataset_path)):
                tree = _tree_from_row(row)
                if tree is not None:
                    yield iteration, dataset_path.relative_to(iter_dir).as_posix(), idx, dict(tree)
                    emitted += 1
                    if max_per_iter is not None and emitted >= max_per_iter:
                        break
            if max_per_iter is not None and emitted >= max_per_iter:
                break
        if max_per_iter is not None and emitted >= max_per_iter:
            continue
        for file_path in sorted(infer_root.rglob("*.json")) + sorted(infer_root.rglob("*.jsonl")):
            if any(parent.name == "traj_results_ds" for parent in file_path.parents):
                continue
            if file_path in seen_files:
                continue
            seen_files.add(file_path)
            for idx, row in enumerate(_iter_json_rows(file_path)):
                tree = _tree_from_row(row)
                if tree is not None:
                    yield iteration, file_path.relative_to(iter_dir).as_posix(), idx, dict(tree)
                    emitted += 1
                    if max_per_iter is not None and emitted >= max_per_iter:
                        break
            if max_per_iter is not None and emitted >= max_per_iter:
                break


def _first_present(node: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in node:
            return node[key]
    return None


def walk(tree: Mapping[str, Any] | str) -> Iterator[dict[str, Any]]:
    """Flatten a reasoning tree, retaining scalar algorithm fields but not logp matrices."""
    root = _decode_tree(tree)
    stack: list[tuple[Mapping[str, Any], str | None, int, str]] = [(root, None, 0, "0")]
    while stack:
        node, parent_id, depth, fallback_id = stack.pop()
        children = node.get("children") or []
        own_id = str(_first_present(node, _FIELD_ALIASES["segment_id"]) or fallback_id)
        row: dict[str, Any] = {
            "segment_id": own_id,
            "parent_id": parent_id,
            "depth": depth,
            "n_children": len(children),
            "reward": node.get("reward"),
        }
        for canonical, aliases in _FIELD_ALIASES.items():
            value = _first_present(node, aliases)
            if canonical != "segment_id" and value is not None:
                row[canonical] = value
        for key, value in node.items():
            if key in {"children", "parent", "_treetune__reasoning_tree"} or key in _MATRIX_KEYS:
                continue
            if key.startswith(("ingpo_", "gear_", "vdra_")) and isinstance(
                value, (str, int, float, bool, type(None), list)
            ) and key not in _MATRIX_KEYS:
                row[key] = value
        yield row
        for index in range(len(children) - 1, -1, -1):
            child = children[index]
            if isinstance(child, Mapping):
                stack.append((child, own_id, depth + 1, f"{fallback_id}.{index}"))


def load_demos(source: str | Path) -> Iterator[dict[str, Any]]:
    return _iter_records(source)


def load_timing(source: str | Path) -> Iterator[dict[str, Any]]:
    return _iter_records(source)


def load_edges(source: str | Path) -> Iterator[dict[str, Any]]:
    return _iter_records(source)


def load_episodes(source: str | Path) -> Iterator[dict[str, Any]]:
    return _iter_records(source)
