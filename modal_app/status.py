"""Read-only Modal status view for progress, GPU-hours, restarts and spend."""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import modal

from modal_app.phase_c import APP_NAME, IMAGES, WORK_VOLUME


app = modal.App(f"{APP_NAME}-status")
WORK = Path("/work")
GPU_RATE_USD_PER_SECOND = 0.001736
FRAMEWORKS = {"R1": "treetune", "R2": "treetune", "R3": "verl", "R4": "verl", "R5": "openrlhf", "R6": "openrlhf"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _active_session_costs(
    session_files: list[str], now: float | None = None
) -> tuple[int, dict[str, float], set[str], float]:
    """Allocate live session spend without counting its startup benchmark twice."""
    now = time.time() if now is None else now
    by_run: dict[str, float] = {}
    active_call_ids: set[str] = set()
    unallocated = 0.0
    count = 0
    for filename in session_files:
        try:
            session = json.loads(Path(filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if session.get("status") != "running":
            continue
        count += 1
        call_id = session.get("call_id")
        if call_id:
            active_call_ids.add(str(call_id))
        elapsed = max(0.0, now - float(session.get("started_at", now)))
        cost = elapsed * GPU_RATE_USD_PER_SECOND
        if session.get("run_id"):
            run_ids = [str(session["run_id"])]
        elif session.get("experiment") and session.get("seeds"):
            run_ids = [f"{session['experiment']}_s{seed}" for seed in session["seeds"]]
        elif session.get("baseline") and session.get("seeds"):
            run_ids = [f"{session['baseline']}_s{seed}" for seed in session["seeds"]]
        else:
            run_ids = []
        if run_ids:
            allocation = cost / len(run_ids)
            for run_id in run_ids:
                by_run[run_id] = by_run.get(run_id, 0.0) + allocation
        else:
            unallocated += cost
    return count, by_run, active_call_ids, unallocated


def _run_row(
    run_dir: Path,
    analysis: Any,
    gpu_hours: Any,
    median_benchmark: float | None,
    active_call_ids: set[str],
    active_spend: float,
    results_root: Path | None = None,
) -> dict[str, Any]:
    run_id = run_dir.name
    exp = run_id.split("_", 1)[0]
    framework = FRAMEWORKS.get(exp, "unknown")
    meta = _read_jsonl(run_dir / "run_meta.jsonl")
    current = analysis.status(str(run_dir), framework if framework != "unknown" else None)
    step = int(current.get("resumable_update", 0))
    target = int(current.get("stop", 0) or 220)
    results_root = results_root or WORK / "results"
    hours_path = results_root / "gpu_hours" / f"{run_id}.json"
    hours = 0.0
    try:
        if hours_path.is_file():
            series = {int(k): float(v) for k, v in json.loads(hours_path.read_text(encoding="utf-8")).items()}
        elif framework == "treetune":
            series = gpu_hours.treetune_hours(str(run_dir), 1)
        elif framework in {"verl", "openrlhf"}:
            pattern, step_re = (("ckpts/global_step_*", r"global_step_(\d+)") if framework == "verl" else
                                ("ckpt/_actor_global_step*", r"global_step(\d+)$"))
            series = gpu_hours.ckpt_hours(str(run_dir), 1, pattern, step_re)
        else:
            series = {}
        available = [s for s in series if s <= step]
        hours = float(series[max(available)]) if available else 0.0
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass

    benchmark_rows = [
        r for r in meta
        if r.get("event") in {"container_start", "container_profile_reused"}
        and r.get("benchmark_iterations_per_second")
    ]
    benchmark_rate = float(benchmark_rows[-1]["benchmark_iterations_per_second"]) if benchmark_rows else None
    outlier = bool(median_benchmark and benchmark_rate and benchmark_rate < 0.9 * median_benchmark)
    spend = 0.0
    starts = set()
    attempts_by_chunk: dict[int, int] = {}
    errors = []
    completed_training_calls = {
        str(row.get("container_call_id"))
        for row in meta
        if row.get("event") in {"modal_container_end", "modal_continuation_end"} and row.get("container_call_id")
    }
    for row in meta:
        event = row.get("event")
        if event in {"container_start", "container_profile_reused"}:
            call_id = str(row.get("container_call_id"))
            if call_id not in completed_training_calls and call_id not in active_call_ids:
                spend += float(row.get("allocated_benchmark_cost_usd", 0.0) or 0.0)
            container = row.get("container_id") or row.get("container_call_id")
            if container:
                starts.add(str(container))
        elif event == "eval_container_start":
            call_id = str(row.get("container_call_id"))
            if call_id not in active_call_ids:
                spend += float(row.get("allocated_benchmark_cost_usd", 0.0) or 0.0)
            container = row.get("container_id") or row.get("container_call_id")
            if container:
                starts.add(str(container))
        elif event in {"modal_container_end", "modal_continuation_end"}:
            spend += float(row.get("allocated_approx_cost_usd", 0.0) or 0.0)
        elif event == "eval_container_end":
            spend += float(row.get("approx_cost_usd", 0.0) or 0.0)
        elif event == "start":
            chunk = int(row.get("chunk_stop", row.get("stop", 0)) or 0)
            attempts_by_chunk[chunk] = max(attempts_by_chunk.get(chunk, 0), int(row.get("attempt", 1) or 1))
        if event == "end" and int(row.get("rc", 0) or 0) != 0:
            errors.append(f"trainer exited {row.get('rc')}")
        if row.get("status") == "failed" or row.get("error"):
            errors.append(str(row.get("error") or row.get("status")))
    restarts = max(0, len(starts) - 1) + sum(max(0, attempts - 1) for attempts in attempts_by_chunk.values())

    return {
        "run_id": run_id,
        "framework": framework,
        "update": step,
        "target": target,
        "gpu_hours": hours,
        "heartbeat_age_seconds": current.get("heartbeat_age_s"),
        "status": current.get("status", "unknown"),
        "restarts": restarts,
        "last_error": errors[-1] if errors else "",
        "spend_usd": spend + active_spend,
        "benchmark_iterations_per_second": benchmark_rate,
        "benchmark_outlier": outlier,
    }


@app.function(
    image=IMAGES["treepo"],
    cpu=4,
    memory=8192,
    timeout=300,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def collect_status() -> dict[str, Any]:
    sys.path.insert(0, "/opt/vdra/analysis")
    import gpu_hours
    import run_status

    run_dirs = sorted(Path("/work/runs").glob("R[1-6]_s*"))
    phase_d_run_items = [
        (path, Path("/work/mock/results"))
        for path in sorted(Path("/work/mock/runs").glob("R[1-6]_s*"))
    ]
    phase_d_run_items.extend(
        (path, Path("/work/mock/resume/results"))
        for path in sorted(Path("/work/mock/resume/runs").glob("R[1-6]_s*"))
    )
    all_run_dirs = run_dirs + [path for path, _ in phase_d_run_items]
    bench_rates = []
    all_meta = {path.name: _read_jsonl(path / "run_meta.jsonl") for path in all_run_dirs}
    for rows in all_meta.values():
        bench_rates.extend(
            float(row["benchmark_iterations_per_second"])
            for row in rows
            if row.get("event") == "container_start"
            and not row.get("benchmark_reused")
            and row.get("benchmark_iterations_per_second")
        )
    median_benchmark = statistics.median(bench_rates) if bench_rates else None
    session_files = (
        glob.glob("/work/results/modal_sessions/*.json")
        + glob.glob("/work/mock/results/modal_sessions/*.json")
        + glob.glob("/work/mock/resume/results/modal_sessions/*.json")
    )
    active_sessions, active_by_run, active_call_ids, unallocated_active_spend = (
        _active_session_costs(session_files)
    )
    runs = [
        _run_row(
            path,
            run_status,
            gpu_hours,
            median_benchmark,
            active_call_ids,
            active_by_run.get(path.name, 0.0),
        )
        for path in run_dirs
    ]
    phase_d_runs = [
        _run_row(
            path,
            run_status,
            gpu_hours,
            median_benchmark,
            active_call_ids,
            active_by_run.get(path.name, 0.0),
            results_root,
        )
        for path, results_root in phase_d_run_items
    ]

    total_spend = (
        sum(float(row["spend_usd"]) for row in runs + phase_d_runs)
        + unallocated_active_spend
    )

    for path in glob.glob("/work/probes/*/run_meta.jsonl"):
        total_spend += sum(
            float(row.get("approx_cost_usd", 0.0) or 0.0)
            for row in _read_jsonl(Path(path))
            if row.get("event") == "probe_end"
        )

    return {
        "generated_at": time.time(),
        "median_benchmark_iterations_per_second": median_benchmark,
        "benchmark_outliers": [row["run_id"] for row in runs if row["benchmark_outlier"]],
        "active_modal_sessions": active_sessions,
        "total_approx_spend_usd": total_spend,
        "runs": runs,
        "phase_d_runs": phase_d_runs,
        "phase_d_approx_spend_usd": sum(float(row["spend_usd"]) for row in phase_d_runs),
    }


@app.local_entrypoint()
def main() -> None:
    report = collect_status.remote()
    for title, rows in (("Production runs", report["runs"]), ("Phase D mock/resume runs", report["phase_d_runs"])):
        print(title)
        print("run       update      GPU-h  heartbeat  restarts  spend    status      error")
        for row in rows:
            heartbeat = "-" if row["heartbeat_age_seconds"] is None else f"{row['heartbeat_age_seconds']:.0f}s"
            error = row["last_error"].replace("\n", " ")[:36]
            print(
                f"{row['run_id']:<9} {row['update']:>3}/{row['target']:<3} "
                f"{row['gpu_hours']:>7.2f} {heartbeat:>10} {row['restarts']:>8} "
                f"${row['spend_usd']:>6.2f} {row['status']:<11} {error}"
            )
    if report["benchmark_outliers"]:
        print("benchmark runs more than 10% below the median: " + ", ".join(report["benchmark_outliers"]))
    print(
        f"Phase D mock/resume spend: ${report['phase_d_approx_spend_usd']:.2f}; "
        f"Modal GPU sessions: {report['active_modal_sessions']} active; "
        f"estimated total project spend ${report['total_approx_spend_usd']:.2f}"
    )
