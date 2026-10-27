"""Tiny paired B200 smoke runs for the three framework images."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import modal

from modal_app.phase_c import (
    APP_NAME,
    EXPERIMENTS,
    GPU_RATE_USD_PER_SECOND,
    IMAGES,
    WORK_VOLUME,
    _checkpoint_commit_watcher,
    _commit_volume_serialized,
    _record_container_start,
    _run_training,
    _runtime_manifest,
)
from modal_app.phase_d_budget import (
    MOCK_TIMEOUT_SECONDS,
    IGPO_PAIR_TIMEOUT_SECONDS,
    IGPO_PROFILE_TIMEOUT_SECONDS,
    IGPO_R1_REFERENCE_TIMEOUT_SECONDS,
    PHASE_D_LIVE_HARD_STOP_USD,
    PHASE_D_NON_GPU_RESERVE_USD,
    PHASE_D_STOP_USD,
    check_live_phase_d_budget as _check_live_phase_d_budget,
)
from modal_app.telemetry import append_jsonl


MOCK_PAIRS = {
    "igpo": ("R1", "R2"),
    "treepo": ("R3", "R4"),
    "treerl": ("R5", "R6"),
}
INTERRUPT_DEADLINE_SECONDS = 9 * 60
PROJECT_SPEND_BEFORE_PROFILE_SWITCH_USD = 25.48359444
app = modal.App(f"{APP_NAME}-phase-d")
FRAMEWORK_NAMES = {"igpo": "treetune", "treepo": "verl", "treerl": "openrlhf"}
PHASE_D_ROOT = Path("/work/mock/phase_d_v6")
PHASE_D_EVAL_BENCHMARKS = ("math500", "aime24", "aime25", "amc23", "olympiadbench")
PHASE_D_EVAL_LIMIT = 16


def _is_monotonic_gpu_hours(series: dict[str, float]) -> bool:
    values = [float(value) for _, value in sorted(series.items(), key=lambda item: int(item[0]))]
    return all(right >= left for left, right in zip(values, values[1:]))


def _all_pairs_complete(results: dict[str, dict[str, Any]]) -> bool:
    return bool(results) and all(result.get("status") == "complete" for result in results.values())


def _phase_d_cost_so_far(
    mock_results_root: Path | None = None,
    resume_results_root: Path | None = None,
) -> float:
    """Include recorded session costs across isolated Phase D volume roots."""
    total = PROJECT_SPEND_BEFORE_PROFILE_SWITCH_USD
    if mock_results_root is None and resume_results_root is None:
        # Run artifacts use versioned roots so failed evidence stays intact. Keep
        # their spend in the cumulative gate when the active root advances.
        phase_roots = set(PHASE_D_ROOT.parent.glob("phase_d_v*")) | {PHASE_D_ROOT}
        results_roots = []
        for phase_root in sorted(phase_roots):
            results_roots.extend(
                [phase_root / "results", phase_root / "resume" / "results"]
            )
            results_roots.extend(phase_root.glob("r1_reference_runs/*/results"))
    else:
        results_roots = [
            mock_results_root or PHASE_D_ROOT / "results",
            resume_results_root or PHASE_D_ROOT / "resume" / "results",
        ]
    seen: set[Path] = set()
    for root in results_roots:
        if root in seen:
            continue
        seen.add(root)
        for path in (root / "modal_sessions").glob("*.json"):
            session = json.loads(path.read_text(encoding="utf-8"))
            total += float(session.get("approx_cost_usd", 0.0))
    return total


def _check_phase_d_budget(cost_so_far: float, reserve_seconds: int = MOCK_TIMEOUT_SECONDS) -> None:
    reserve = reserve_seconds * GPU_RATE_USD_PER_SECOND
    if cost_so_far + reserve + PHASE_D_NON_GPU_RESERVE_USD > PHASE_D_LIVE_HARD_STOP_USD:
        raise RuntimeError(
            f"Phase D budget gate: ${cost_so_far:.2f} spent plus ${reserve:.2f} reserved "
            f"and ${PHASE_D_NON_GPU_RESERVE_USD:.2f} non-GPU headroom "
            f"would exceed the ${PHASE_D_LIVE_HARD_STOP_USD:.2f} local hard stop "
            f"for the ${PHASE_D_STOP_USD:.2f} project cap"
        )


def _is_checkpoint_for_interrupt(progress: int, target: int) -> bool:
    return progress == target


def _is_safe_interrupt_checkpoint(state: dict[str, Any], target: int) -> bool:
    launch = state.get("last_launch") or {}
    try:
        chunk_stop = int(launch.get("chunk_stop", -1))
    except (TypeError, ValueError):
        return False
    return (
        _is_checkpoint_for_interrupt(int(state.get("resumable_update", 0)), target)
        and chunk_stop == target
        and state.get("heartbeat_age_s") is None
    )


def _validate_igpo_max_num_seqs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 128:
        raise ValueError("INGPO_VLLM_MAX_NUM_SEQS must be an integer between 1 and 128")
    return value


def phase_d_run_root(
    run_tag: str = "",
    *,
    igpo_max_num_seqs: int = 64,
) -> Path:
    """Return the standard Phase D root or a fresh, isolated IGPO sweep root."""
    igpo_max_num_seqs = _validate_igpo_max_num_seqs(igpo_max_num_seqs)
    if igpo_max_num_seqs != 64 and not run_tag:
        raise ValueError("a nondefault IGPO sequence limit requires an isolated run_tag")
    if not run_tag:
        return PHASE_D_ROOT
    if not run_tag.isalnum():
        raise ValueError("Phase D run_tag must be alphanumeric")
    return PHASE_D_ROOT.parent / f"phase_d_v7_{run_tag}"


def mock_environment(
    root: Path = PHASE_D_ROOT,
    *,
    igpo_max_num_seqs: int = 64,
) -> dict[str, str]:
    """Return the frozen tiny settings specified for Phase D."""
    root = Path(root)
    igpo_max_num_seqs = _validate_igpo_max_num_seqs(igpo_max_num_seqs)
    return {
        "STOP_STEP": "4",
        "EVAL_EVERY_N": "2",
        "EVAL_AT_END": "1",
        "EVAL_STEP0": "1",
        "RESUME_EVERY": "2",
        "KEEP_EVERY": "2",
        "SEEDS": "42",
        "SAMPLE_K": "2",
        "VDRA_EVAL_LIMIT": "16",
        "COMPUTE_MATCHED": "0",
        "MAX_RESTARTS": "0",
        "WORK": "/work",
        "RUNS_DIR": str(root / "runs"),
        "RESULTS_DIR": str(root / "results"),
        "BANKS_DIR": str(root / "banks"),
        "DATA_DIR": "/work/data_vdra",
        "MATH_DICT": "/work/source_data/math",
        "EVAL_SETS": (
            "math500=/work/source_data/math:test "
            "aime24=/work/source_data/aime24 "
            "aime25=/work/source_data/aime25 "
            "amc23=/work/source_data/amc23 "
            "olympiadbench=/work/source_data/olympiadbench_hf"
        ),
        "VDRA_HOME": "/opt/vdra",
        "IGPO_DIR": "/opt/work/igpo",
        "TREEPO_DIR": "/opt/work/TreePO",
        "TREERL_DIR": "/opt/work/TreeRL",
        "VDRA_B200": "1",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # Tiny, shared IGPO smoke settings. Limit vLLM capture/concurrency and
        # episode volume so checkpoint/resume checks fit the Phase D budget.
        # Keep vLLM memory below the training allocation; the B200 container
        # reports retained vLLM memory under its PID 1 after server shutdown.
        "INGPO_VLLM_GPU_MEMORY_UTILIZATION": "0.25",
        "INGPO_VLLM_MIN_AVAILABLE_GPU_MEMORY_MB": str(20 * 1024),
        "INGPO_WAIT_UNTIL_MEMORY_RELEASE": "false",
        "INGPO_DATASET_NUM_SAMPLES_PER_ITERATION": "4",
        "INGPO_NUM_EPISODES_PER_ITERATION": "64",
        "INGPO_TARGET_TRAIN_BATCH_SIZE": "32",
        "INGPO_VLLM_MAX_NUM_SEQS": str(igpo_max_num_seqs),
        "INGPO_MAX_CONCURRENT_PROGRAMS": "16",
        "INGPO_MAX_CONCURRENT_GENERATIONS": "64",
        # Log duration and padded sequence length for the first three IGPO
        # logprob batches to identify unusually expensive cold batches.
        "VDRA_LOGPROB_PROFILE": "1",
        # Run step-0/2/4 evaluation through the shared Phase D evaluator;
        # IGPO's built-in pass uses repo-relative dataset paths and duplicates it.
        "INGPO_DISABLE_INTERNAL_EVALUATION": "1",
    }


def _igpo_eval_config_preflight(
    config_root: Path = Path("/opt/work/igpo/configs"),
    model_path: str = "/work/hf_cache/model",
) -> dict[str, Any]:
    """Compile R1/R2 and ensure every evaluation tokenizer follows the shared model override."""
    from modal_app.igpo_config_preflight import preflight_igpo_configs

    return preflight_igpo_configs(config_root, model_path)


def resume_environment(root: Path = PHASE_D_ROOT) -> dict[str, str]:
    """Keep the Phase D settings while isolating interrupted-run artifacts."""
    root = Path(root) / "resume"
    env = mock_environment(root.parent)
    env.update(
        {
            "RUNS_DIR": str(root / "runs"),
            "RESULTS_DIR": str(root / "results"),
            "BANKS_DIR": str(root / "banks"),
            "EVAL_STEP0": "0",
        }
    )
    return env


def _validate_pair(
    framework: str,
    runs_root: Path = PHASE_D_ROOT / "runs",
    results_root: Path = PHASE_D_ROOT / "results",
    *,
    model_path: str,
    dataset_specs: dict[str, str],
) -> dict[str, Any]:
    runs_root = Path(runs_root)
    results_root = Path(results_root)
    summary: dict[str, Any] = {}
    for exp in MOCK_PAIRS[framework]:
        run_id = f"{exp}_s42"
        run_dir = runs_root / run_id
        metric_path = run_dir / "phase_d_metrics.jsonl"
        reference_path = results_root / "equivalence" / f"reference_{run_id}.json"
        if not metric_path.is_file() or not reference_path.is_file():
            raise RuntimeError(f"{run_id}: fixed-seed metric reference is missing")
        state = json.loads(
            subprocess.check_output(
                [sys.executable, "/opt/vdra/analysis/run_status.py", str(run_dir),
                 "--framework", FRAMEWORK_NAMES[framework], "--json"],
                text=True,
            )
        )
        if state.get("resumable_update") != 4:
            raise RuntimeError(f"{run_id}: expected resumable update 4, got {state.get('resumable_update')}")
        checkpoint_steps = _checkpoint_updates(run_dir, FRAMEWORK_NAMES[framework])
        missing_checkpoints = [step for step in (2, 4) if step not in checkpoint_steps]
        if missing_checkpoints:
            raise RuntimeError(f"{run_id}: missing checkpoint steps {missing_checkpoints}; found {checkpoint_steps}")

        gpu_hours_path = results_root / "gpu_hours" / f"{run_id}.json"
        subprocess.run(
            [sys.executable, "/opt/vdra/analysis/gpu_hours.py", "--run_dir", str(run_dir),
             "--framework", FRAMEWORK_NAMES[framework], "--gpus", "1", "--out", str(gpu_hours_path)],
            check=True,
        )
        gpu_hours = json.loads(gpu_hours_path.read_text(encoding="utf-8"))
        if not _is_monotonic_gpu_hours(gpu_hours):
            raise RuntimeError(f"{run_id}: GPU-hour series is not monotonic")

        eval_problems = _phase_d_load_eval_problems(dataset_specs, PHASE_D_EVAL_LIMIT)
        checkpoint_paths = _phase_d_checkpoint_paths(run_dir, model_path)
        eval_files = {}
        for step in (0, 2, 4):
            expected_signatures = _phase_d_expected_eval_signatures(
                run_id,
                step,
                checkpoint_paths[step],
                model_path,
                dataset_specs,
                limit=PHASE_D_EVAL_LIMIT,
                problems_by_benchmark=eval_problems,
            )
            rows = _evaluation_rows(
                results_root, run_id, step, expected_signatures=expected_signatures
            )
            eval_files[str(step)] = len(rows)

        audit_path = results_root / "audit" / f"{run_id}.json"
        audit_command = [sys.executable, "/opt/vdra/analysis/budget_audit.py"]
        if exp in {"R2", "R4", "R6"}:
            audit_command.append("--require-budget-evidence")
        audit_command.extend(["--logs", str(run_dir), "--out", str(audit_path)])
        subprocess.run(
            audit_command,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("budget_violations") != 0:
            raise RuntimeError(f"{run_id}: budget audit found {audit.get('budget_violations')} violations")

        vdra_files = sorted(path for path in run_dir.rglob("vdra_*.jsonl") if path.stat().st_size)
        if exp in {"R2", "R4", "R6"} and not vdra_files:
            raise RuntimeError(f"{run_id}: no non-empty VDRA logs found")
        summary[run_id] = {
            "update": state["resumable_update"],
            "checkpoint_steps": checkpoint_steps,
            "gpu_hours": gpu_hours,
            "evaluation_file_counts": eval_files,
            "budget_violations": audit["budget_violations"],
            "vdra_log_files": len(vdra_files),
            "reference_metrics": str(reference_path),
        }
        if framework == "treerl":
            sync_path = results_root / "equivalence" / f"treerl_sync_{run_id}.json"
            if not sync_path.is_file():
                raise RuntimeError(f"{run_id}: TreeRL actor/vLLM sync report is missing")
            sync_report = json.loads(sync_path.read_text(encoding="utf-8"))
            if sync_report.get("matched") is not True:
                raise RuntimeError(f"{run_id}: TreeRL actor/vLLM sync did not match: {sync_report}")
            summary[run_id]["sync_verification"] = sync_report
    return summary


def _run_status(run_dir: Path, framework: str) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output(
            [sys.executable, "/opt/vdra/analysis/run_status.py", str(run_dir),
             "--framework", framework, "--json"],
            text=True,
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


_METRIC_FIELDS = (
    "framework",
    "update",
    "rollout_count",
    "generated_tokens",
    "mean_reward",
    "loss",
    "grad_norm",
)


def _read_metric_events(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Phase D metric file is missing: {path}")
    events: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid metric JSON at {path}:{line_number}") from exc
        missing = [field for field in _METRIC_FIELDS if field not in row]
        if missing:
            raise RuntimeError(f"metric row at {path}:{line_number} is missing {missing}")
        update = row["update"]
        if isinstance(update, bool) or not isinstance(update, int) or update < 1:
            raise RuntimeError(f"metric row at {path}:{line_number} has invalid update {update!r}")
        if update in events:
            raise RuntimeError(f"duplicate metric event for update {update} in {path}")
        for field in ("rollout_count", "generated_tokens"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"metric row at {path}:{line_number} has invalid {field}")
        for field in ("mean_reward", "loss", "grad_norm"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"metric row at {path}:{line_number} has invalid {field}")
        events[update] = {field: row[field] for field in _METRIC_FIELDS}
    return events


def _build_reference_artifact(
    exp: str,
    framework: str,
    metrics_path: Path,
    results_root: Path = PHASE_D_ROOT / "results",
) -> dict[str, Any]:
    """Save the fixed-seed canonical metrics for updates 1 and 2."""
    run_id = f"{exp}_s42"
    events = _read_metric_events(Path(metrics_path))
    missing = [update for update in (1, 2) if update not in events]
    if missing:
        raise RuntimeError(f"{run_id}: missing updates {missing} in Phase D metric file")
    wrong_framework = [update for update in (1, 2) if events[update]["framework"] != framework]
    if wrong_framework:
        raise RuntimeError(f"{run_id}: metrics have the wrong framework at updates {wrong_framework}")
    report = {
        "run_id": run_id,
        "seed": 42,
        "framework": framework,
        "updates": {str(update): events[update] for update in (1, 2)},
    }
    path = Path(results_root) / "equivalence" / f"reference_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _compare_resume_metric_events(
    reference_file: Path,
    resumed_file: Path,
    run_id: str,
) -> dict[str, Any]:
    """Compare rollout and generated-token counts at resumed updates 3 and 4."""
    reference = _read_metric_events(Path(reference_file))
    resumed = _read_metric_events(Path(resumed_file))
    missing_reference = [update for update in (3, 4) if update not in reference]
    missing_resumed = [update for update in (3, 4) if update not in resumed]
    if missing_reference or missing_resumed:
        raise RuntimeError(
            f"{run_id}: missing resume metric updates; "
            f"reference={missing_reference}, resumed={missing_resumed}"
        )
    mismatches = []
    for update in (3, 4):
        ref, got = reference[update], resumed[update]
        if (ref["rollout_count"], ref["generated_tokens"]) != (got["rollout_count"], got["generated_tokens"]):
            mismatches.append(update)
    if mismatches:
        raise RuntimeError(f"{run_id}: rollout/token counts differ at updates {mismatches}")
    return {"run_id": run_id, "matched": True, "compared_updates": [3, 4]}


def _assert_gpu_hours_pause_excluded(before: dict[str, float], after: dict[str, float], step: str = "2") -> None:
    if step not in before or step not in after:
        raise RuntimeError(f"GPU-hour series is missing interruption checkpoint {step}")
    if not math.isclose(float(before[step]), float(after[step]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"GPU-hours changed during interruption pause at update {step}: "
            f"{before[step]} -> {after[step]}"
        )


def _parse_gpu_memory_output(output: str) -> dict[str, int]:
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        try:
            used, total = int(fields[0]), int(fields[1])
        except (IndexError, ValueError):
            break
        if used < 0 or total <= 0 or used > total:
            break
        return {"used_mib": used, "total_mib": total}
    raise RuntimeError("could not parse nvidia-smi memory output")


def _checkpoint_updates(run_dir: Path, framework: str) -> list[int]:
    run_dir = Path(run_dir)
    if framework == "treetune":
        candidates = run_dir.glob("checkpoints/ckpt--iter_*")
        pattern = re.compile(r"ckpt--iter_([0-9.]+)")
    elif framework == "verl":
        candidates = run_dir.glob("ckpts/global_step_*")
        pattern = re.compile(r"global_step_(\d+)")
    elif framework == "openrlhf":
        candidates = run_dir.glob("ckpt/_actor_resume/update*")
        pattern = re.compile(r"update(\d+)")
    else:
        raise ValueError(f"unknown checkpoint framework: {framework}")
    updates = set()
    for candidate in candidates:
        match = pattern.search(candidate.name)
        if match:
            try:
                updates.add(int(float(match.group(1))))
            except ValueError:
                continue
    return sorted(updates)


class _GpuMemorySampler:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.thread: threading.Thread | None = None

    def _capture(self) -> None:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            self.samples.append({"time": time.time(), **_parse_gpu_memory_output(output)})
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self._capture()

    def start(self) -> None:
        self._capture()
        if not self.samples:
            raise RuntimeError(f"could not sample TreeRL GPU memory: {self.errors[-1] if self.errors else 'unknown error'}")
        self.thread = threading.Thread(target=self._run, name="vdra-phase-d-gpu-memory", daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        if not self.samples:
            raise RuntimeError("TreeRL GPU memory sampler captured no samples")
        total_values = {sample["total_mib"] for sample in self.samples}
        if len(total_values) != 1:
            raise RuntimeError(f"TreeRL GPU total memory changed while sampling: {sorted(total_values)}")
        peak = max(sample["used_mib"] for sample in self.samples)
        total = next(iter(total_values))
        return {
            "framework": "treerl",
            "sample_count": len(self.samples),
            "peak_used_mib": peak,
            "total_mib": total,
            "free_at_peak_mib": total - peak,
            "fits": peak < total,
            "sampling_errors": self.errors,
        }


def _phase_d_dataset_specs(env: dict[str, str] | None = None) -> dict[str, str]:
    values = os.environ.copy()
    values.update(env or {})
    specs_text = values.get("EVAL_SETS", "").strip()
    if not specs_text:
        work = values.get("WORK", "/work")
        math_dict = values.get("MATH_DICT", f"{work}/source_data/math")
        specs_text = (
            f"math500={math_dict}:test "
            f"aime24={work}/source_data/aime24 "
            f"aime25={work}/source_data/aime25 "
            f"amc23={work}/source_data/amc23 "
            f"olympiadbench={work}/source_data/olympiadbench_hf"
        )
    specs: dict[str, str] = {}
    for item in specs_text.split():
        if "=" not in item:
            raise ValueError(f"invalid Phase D EVAL_SETS item: {item!r}")
        name, spec = item.split("=", 1)
        if not name or not spec or name in specs:
            raise ValueError(f"invalid or duplicate Phase D evaluation benchmark: {item!r}")
        specs[name] = spec
    if set(specs) != set(PHASE_D_EVAL_BENCHMARKS):
        raise ValueError(
            f"Phase D requires EVAL_SETS for {PHASE_D_EVAL_BENCHMARKS}, got {tuple(specs)}"
        )
    return specs


def _phase_d_analysis_imports():
    """Import the same shared loader, signature builder and verifier as eval_pending."""
    repo_root = Path(__file__).resolve().parents[1]
    igpo_dir = os.environ.get("IGPO_DIR", "/opt/work/igpo")
    vdra_home = os.environ.get("VDRA_HOME", str(repo_root))
    analysis_dir = str(repo_root / "analysis")
    for entry in (analysis_dir, vdra_home, igpo_dir):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path[:0] = [igpo_dir, vdra_home, analysis_dir]
    from datasets_util import load_problems
    from eval_checkpoints import evaluation_signature, limit_problems
    from vdra_core import verifier

    return load_problems, evaluation_signature, limit_problems, verifier


def _phase_d_load_eval_problems(
    dataset_specs: dict[str, str], limit: int = PHASE_D_EVAL_LIMIT
) -> dict[str, list[dict[str, str]]]:
    load_problems, _, limit_problems, _ = _phase_d_analysis_imports()
    problems = {
        name: limit_problems(load_problems(spec), limit)
        for name, spec in dataset_specs.items()
    }
    empty = [name for name, rows in problems.items() if not rows]
    if empty:
        raise RuntimeError(f"Phase D evaluation datasets are empty: {empty}")
    return problems


def _phase_d_expected_eval_signatures(
    run_id: str,
    step: int,
    checkpoint_path: str,
    tokenizer: str,
    dataset_specs: dict[str, str],
    *,
    limit: int = PHASE_D_EVAL_LIMIT,
    problems_by_benchmark: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, str]:
    _, evaluation_signature, limit_problems, verifier = _phase_d_analysis_imports()
    problem_sets = problems_by_benchmark or _phase_d_load_eval_problems(dataset_specs, limit)
    verifier._BACKEND = None
    verifier_backend = verifier.backend_name()
    verifier_fingerprint = verifier.backend_fingerprint()
    expected = {}
    for benchmark, spec in dataset_specs.items():
        problems = limit_problems(problem_sets[benchmark], limit)
        expected[benchmark] = evaluation_signature(
            run_id=run_id,
            step=step,
            checkpoint_path=checkpoint_path,
            benchmark=benchmark,
            dataset_spec=spec,
            problems=problems,
            mode="greedy",
            max_model_len=2048,
            tokenizer=tokenizer,
            seed=42,
            temperature=0.0,
            sample_k=1,
            verifier_backend=verifier_backend,
            verifier_fingerprint=verifier_fingerprint,
            gpu_hours=None,
            limit=limit,
        )
    return expected


def _phase_d_checkpoint_paths(run_dir: Path, model_path: str) -> dict[int, str]:
    repo_root = Path(__file__).resolve().parents[1]
    script = Path(os.environ.get("VDRA_HOME", str(repo_root))) / "analysis" / "list_ckpts.py"
    rows = subprocess.check_output(
        [sys.executable, str(script), str(run_dir)], text=True
    )
    checkpoints = {0: model_path}
    for line in rows.splitlines():
        if "\t" not in line:
            continue
        step_text, path = line.split("\t", 1)
        step = int(step_text)
        if step > 0:
            checkpoints[step] = path
    missing = [step for step in (2, 4) if step not in checkpoints]
    if missing:
        raise RuntimeError(f"{run_dir}: missing evaluation checkpoints {missing}")
    return checkpoints


def _evaluation_rows(
    results_root: Path,
    run_id: str,
    step: int,
    *,
    expected_signatures: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    eval_dir = results_root / "eval"
    files = sorted(eval_dir.glob(f"{run_id}.step{step}.*.jsonl"))
    expected_names = {
        f"{run_id}.step{step}.{benchmark}.greedy.jsonl"
        for benchmark in PHASE_D_EVAL_BENCHMARKS
    }
    found_names = {path.name for path in files}
    if found_names != expected_names:
        raise RuntimeError(
            f"{run_id}: step-{step} expected evaluation set {sorted(expected_names)}, "
            f"found {sorted(found_names)}"
        )
    if set(expected_signatures) != set(PHASE_D_EVAL_BENCHMARKS):
        raise RuntimeError(
            f"{run_id}: current evaluation signatures do not cover every benchmark"
        )

    validated: dict[str, list[dict[str, Any]]] = {}
    for path in files:
        benchmark = path.name.split(f".step{step}.", 1)[1].removesuffix(".greedy.jsonl")
        try:
            result_bytes = path.read_bytes()
            metadata = json.loads(Path(f"{path}.meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{run_id}: evaluation result/sidecar is unreadable for {path.name}") from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != 1
            or not isinstance(metadata.get("signature"), str)
            or re.fullmatch(r"[0-9a-f]{64}", metadata["signature"]) is None
        ):
            raise RuntimeError(f"{run_id}: evaluation metadata is invalid for {path.name}")
        if metadata["signature"] != expected_signatures[benchmark]:
            raise RuntimeError(
                f"{run_id}: evaluation signature does not match current inputs for {path.name}"
            )
        result_lines = result_bytes.splitlines()
        if len(result_lines) != PHASE_D_EVAL_LIMIT or any(not line.strip() for line in result_lines):
            raise RuntimeError(f"{run_id}: {path.name} contains blank or extra result lines")
        rows = _read_jsonl(path)
        if metadata.get("row_count") != PHASE_D_EVAL_LIMIT or len(rows) != PHASE_D_EVAL_LIMIT:
            raise RuntimeError(
                f"{run_id}: {path.name} must contain exactly {PHASE_D_EVAL_LIMIT} rows"
            )
        digest = hashlib.sha256(result_bytes).hexdigest()
        if metadata.get("result_sha256") != digest:
            raise RuntimeError(f"{run_id}: evaluation sidecar digest does not match {path.name}")
        problem_ids = set()
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("run_id") != run_id
                or row.get("step") != step
                or row.get("benchmark") != benchmark
                or row.get("mode") != "greedy"
                or type(row.get("correct")) not in (int, float)
                or row.get("correct") not in (0, 1)
                or row.get("problem_id") is None
            ):
                raise RuntimeError(f"{run_id}: invalid evaluation row in {path.name}")
            problem_id = str(row["problem_id"])
            if not problem_id or problem_id in problem_ids:
                raise RuntimeError(f"{run_id}: missing or duplicate problem ID in {path.name}")
            problem_ids.add(problem_id)
        validated[path.name] = [
            {key: value for key, value in row.items() if key != "gpu_hours"}
            for row in rows
        ]
    return validated


def _validate_resume_equivalence(
    exp: str,
    reference_root: Path = PHASE_D_ROOT / "results",
    resumed_root: Path = PHASE_D_ROOT / "resume" / "results",
    *,
    reference_expected_signatures: dict[str, str],
    resumed_expected_signatures: dict[str, str],
) -> dict[str, Any]:
    """Compare deterministic step-4 eval results from the ordinary and resumed runs."""
    run_id = f"{exp}_s42"
    reference = _evaluation_rows(
        reference_root, run_id, 4, expected_signatures=reference_expected_signatures
    )
    resumed = _evaluation_rows(
        resumed_root, run_id, 4, expected_signatures=resumed_expected_signatures
    )
    if reference.keys() != resumed.keys():
        raise RuntimeError(
            f"{run_id}: step-4 evaluation file sets differ: "
            f"reference={sorted(reference)}, resumed={sorted(resumed)}"
        )
    mismatches = [name for name in reference if reference[name] != resumed[name]]
    report = {
        "run_id": run_id,
        "step": 4,
        "compared_files": sorted(reference),
        "matched": not mismatches,
        "mismatched_files": mismatches,
        "reference": reference,
        "resumed": resumed,
    }
    report_path = reference_root / "equivalence" / f"eval_step4_{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mismatches:
        raise RuntimeError(f"{run_id}: resumed step-4 evaluations differ in {mismatches}")
    return report


def _interrupt_and_resume(
    exp: str,
    framework: str,
    model_path: str,
    platform_call_id: str,
    root: Path = PHASE_D_ROOT,
) -> dict[str, Any]:
    """Kill one real trainer after update 2, then resume it to update 4."""
    root = Path(root)
    env = resume_environment(root)
    env["MODEL_PATH"] = model_path
    run_id = f"{exp}_s42"
    env["VDRA_METRICS_JSONL"] = f"{env['RUNS_DIR']}/{run_id}/phase_d_metrics.jsonl"
    if framework == "openrlhf":
        env["VDRA_TREERL_VERIFY_SYNC"] = "1"
        env["VDRA_TREERL_SYNC_REPORT"] = f"{env['RESULTS_DIR']}/equivalence/treerl_sync_{run_id}.json"
    run_dir = Path(env["RUNS_DIR"]) / run_id
    results_root = Path(env["RESULTS_DIR"])
    call_id = f"{platform_call_id}-{exp}-resume"
    session_dir = results_root / "modal_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / f"{call_id}.json"
    started = time.time()
    session = {
        "call_id": call_id,
        "experiment": exp,
        "seeds": ["42"],
        "gpu": "B200",
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "started_at": started,
        "approx_cost_usd_per_second": GPU_RATE_USD_PER_SECOND,
        "status": "running",
    }
    session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

    previous_call_id = os.environ.get("MODAL_FUNCTION_CALL_ID")
    os.environ["MODAL_FUNCTION_CALL_ID"] = call_id
    stop_watcher = threading.Event()
    process = None
    interrupted = False
    completed = False
    try:
        _record_container_start(exp, ["42"], os.environ.get("MODAL_IMAGE_ID"), Path(env["RUNS_DIR"]))
        watcher = threading.Thread(
            target=_checkpoint_commit_watcher,
            args=(exp, ["42"], stop_watcher, Path(env["RUNS_DIR"])),
            daemon=True,
        )
        watcher_started = False
        watcher.start()
        watcher_started = True

        process_env = os.environ.copy()
        process_env.update(env)
        process_env.update(
            {
                "GPU": "0",
                "GPUS": "0",
                "GPUS_PER_RUN": "1",
                "GPU_GROUPS": "0",
                "VDRA_SINGLE_RANK_COLLECTIVES": "1" if exp in {"R1", "R2", "R5", "R6"} else "0",
            }
        )
        log_dir = Path(env["RUNS_DIR"]) / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{exp}_resume_gpu0.log"
        command = ["bash", f"/opt/vdra/scripts/exp/{EXPERIMENTS[exp][0]}"]
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                env=process_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + INTERRUPT_DEADLINE_SECONDS
            while time.monotonic() < deadline:
                state = _run_status(run_dir, framework)
                progress = int(state.get("resumable_update", 0))
                if _is_safe_interrupt_checkpoint(state, 2):
                    # The trainer chunk has ended at update 2; flush its finished
                    # checkpoint before killing the launcher during its eval pause.
                    _commit_volume_serialized()
                    stable_state = _run_status(run_dir, framework)
                    if _is_safe_interrupt_checkpoint(stable_state, 2) and process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=30)
                        interrupted = True
                        break
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{run_id}: trainer exited with code {process.returncode} before interruption at update 2"
                    )
                time.sleep(1.0)
            if not interrupted:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                raise TimeoutError(
                    f"{run_id}: did not reach update 2 within {INTERRUPT_DEADLINE_SECONDS // 60} minutes"
                )

            interrupted_state = _run_status(run_dir, framework)
            if interrupted_state.get("resumable_update") != 2:
                raise RuntimeError(f"{run_id}: expected checkpoint 2 after kill, got {interrupted_state}")
            if 2 not in _checkpoint_updates(run_dir, framework):
                raise RuntimeError(f"{run_id}: checkpoint 2 is missing after interruption")
            hours_before_pause_path = results_root / "gpu_hours" / f"{run_id}.interruption_before.json"
            hours_after_pause_path = results_root / "gpu_hours" / f"{run_id}.interruption_after.json"
            subprocess.run(
                [sys.executable, "/opt/vdra/analysis/gpu_hours.py", "--run_dir", str(run_dir),
                 "--framework", framework, "--gpus", "1", "--out", str(hours_before_pause_path)],
                check=True,
            )
            hours_before_pause = json.loads(hours_before_pause_path.read_text(encoding="utf-8"))
            pause_started = time.monotonic()
            time.sleep(5.0)
            interruption_pause_seconds = time.monotonic() - pause_started
            subprocess.run(
                [sys.executable, "/opt/vdra/analysis/gpu_hours.py", "--run_dir", str(run_dir),
                 "--framework", framework, "--gpus", "1", "--out", str(hours_after_pause_path)],
                check=True,
            )
            hours_after_pause = json.loads(hours_after_pause_path.read_text(encoding="utf-8"))
            _assert_gpu_hours_pause_excluded(hours_before_pause, hours_after_pause)
            subprocess.run(command, env=process_env, stdout=log_handle, stderr=subprocess.STDOUT, check=True)

        final_state = _run_status(run_dir, framework)
        if final_state.get("resumable_update") != 4:
            raise RuntimeError(f"{run_id}: expected update 4 after resume, got {final_state}")
        if 4 not in _checkpoint_updates(run_dir, framework):
            raise RuntimeError(f"{run_id}: checkpoint 4 is missing after resume")
        resume_from = int((final_state.get("last_launch") or {}).get("resume_from", -1))
        if resume_from != 2:
            raise RuntimeError(f"{run_id}: expected resume_from=2, got {resume_from}")

        gpu_hours_path = results_root / "gpu_hours" / f"{run_id}.json"
        subprocess.run(
            [sys.executable, "/opt/vdra/analysis/gpu_hours.py", "--run_dir", str(run_dir),
             "--framework", framework, "--gpus", "1", "--out", str(gpu_hours_path)],
            check=True,
        )
        gpu_hours = json.loads(gpu_hours_path.read_text(encoding="utf-8"))
        if not _is_monotonic_gpu_hours(gpu_hours):
            raise RuntimeError(f"{run_id}: resumed GPU-hour series is not monotonic")
        audit_path = results_root / "audit" / f"{run_id}.json"
        audit_command = [sys.executable, "/opt/vdra/analysis/budget_audit.py"]
        if exp in {"R2", "R4", "R6"}:
            audit_command.append("--require-budget-evidence")
        audit_command.extend(["--logs", str(run_dir), "--out", str(audit_path)])
        subprocess.run(
            audit_command,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("budget_violations") != 0:
            raise RuntimeError(f"{run_id}: resumed budget audit found {audit.get('budget_violations')} violations")
        vdra_files = sorted(path for path in run_dir.rglob("vdra_*.jsonl") if path.stat().st_size)
        if exp in {"R2", "R4", "R6"} and not vdra_files:
            raise RuntimeError(f"{run_id}: resumed run has no non-empty VDRA logs")
        dataset_specs = _phase_d_dataset_specs(env)
        eval_problems = _phase_d_load_eval_problems(dataset_specs, PHASE_D_EVAL_LIMIT)
        reference_checkpoint = _phase_d_checkpoint_paths(
            root / "runs" / run_id, model_path
        )[4]
        resumed_checkpoint = _phase_d_checkpoint_paths(run_dir, model_path)[4]
        reference_expected = _phase_d_expected_eval_signatures(
            run_id,
            4,
            reference_checkpoint,
            model_path,
            dataset_specs,
            problems_by_benchmark=eval_problems,
        )
        resumed_expected = _phase_d_expected_eval_signatures(
            run_id,
            4,
            resumed_checkpoint,
            model_path,
            dataset_specs,
            problems_by_benchmark=eval_problems,
        )
        equivalence = _validate_resume_equivalence(
            exp,
            root / "results",
            results_root,
            reference_expected_signatures=reference_expected,
            resumed_expected_signatures=resumed_expected,
        )
        metric_equivalence = _compare_resume_metric_events(
            root / "runs" / run_id / "phase_d_metrics.jsonl",
            run_dir / "phase_d_metrics.jsonl",
            run_id,
        )
        sync_report = None
        if framework == "openrlhf":
            sync_path = Path(env["VDRA_TREERL_SYNC_REPORT"])
            if not sync_path.is_file():
                raise RuntimeError(f"{run_id}: TreeRL actor/vLLM sync report is missing")
            sync_report = json.loads(sync_path.read_text(encoding="utf-8"))
            if sync_report.get("matched") is not True:
                raise RuntimeError(f"{run_id}: TreeRL actor/vLLM sync did not match: {sync_report}")
        completed = True
        return {
            "run_id": run_id,
            "interrupted_after_update": 2,
            "resume_from_update": resume_from,
            "final_update": final_state["resumable_update"],
            "gpu_hours": gpu_hours,
            "budget_violations": audit["budget_violations"],
            "vdra_log_files": len(vdra_files),
            "equivalence": equivalence,
            "metric_equivalence": metric_equivalence,
            "interruption_pause_seconds": interruption_pause_seconds,
            "gpu_hours_at_interruption": hours_before_pause.get("2"),
            "sync_verification": sync_report,
            "wall_seconds": time.time() - started,
        }
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        stop_watcher.set()
        if locals().get("watcher_started", False):
            watcher.join(timeout=30)
        _commit_volume_serialized()
        elapsed = time.time() - started
        session.update(
            {
                "ended_at": time.time(),
                "wall_seconds": elapsed,
                "approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND,
                "status": "complete" if completed else "failed",
            }
        )
        session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
        append_jsonl(
            Path(env["RUNS_DIR"]) / run_id / "run_meta.jsonl",
            {
                "event": "modal_container_end",
                "time": time.time(),
                "container_call_id": call_id,
                "wall_seconds": elapsed,
                "approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND,
                "allocated_approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND,
                "status": session["status"],
                "error": None if session["status"] == "complete" else "interrupted-resume probe failed",
            },
        )
        if previous_call_id is None:
            os.environ.pop("MODAL_FUNCTION_CALL_ID", None)
        else:
            os.environ["MODAL_FUNCTION_CALL_ID"] = previous_call_id


def _run_pair_impl(
    framework: str,
    *,
    root: Path = PHASE_D_ROOT,
    igpo_max_num_seqs: int = 64,
) -> dict[str, Any]:
    if framework not in MOCK_PAIRS:
        raise ValueError(f"unknown framework: {framework}")
    igpo_max_num_seqs = _validate_igpo_max_num_seqs(igpo_max_num_seqs)
    if framework != "igpo" and igpo_max_num_seqs != 64:
        raise ValueError("IGPO sequence tuning only applies to the IGPO pair")
    reserve_seconds = (
        IGPO_PAIR_TIMEOUT_SECONDS if framework == "igpo" else MOCK_TIMEOUT_SECONDS
    )
    _check_phase_d_budget(_phase_d_cost_so_far(), reserve_seconds=reserve_seconds)
    manifest = _runtime_manifest()
    if not manifest.get("ready"):
        raise RuntimeError("run setup_data() successfully before the Phase D smoke runs")

    if framework == "igpo":
        # Compile the CUDA optimizer before step-0 evaluation can affect the
        # runtime header state; training then loads this cached B200 extension.
        subprocess.run(
            [sys.executable, "/opt/vdra/modal_app/prebuild_deepspeed_fused_adam.py"],
            check=True,
        )

    env = mock_environment(
        root,
        igpo_max_num_seqs=igpo_max_num_seqs,
    )
    env["MODEL_PATH"] = str(manifest["model_path"])
    platform_call_id = os.environ.get("MODAL_FUNCTION_CALL_ID") or uuid.uuid4().hex
    previous_call_id = os.environ.get("MODAL_FUNCTION_CALL_ID")
    results = []
    try:
        for exp in MOCK_PAIRS[framework]:
            # _run_training records each arm separately even though the pair shares a
            # single B200 container, making session files and accounting unambiguous.
            os.environ["MODAL_FUNCTION_CALL_ID"] = f"{platform_call_id}-{exp}"
            run_id = f"{exp}_s42"
            run_env = env.copy()
            run_env["VDRA_METRICS_JSONL"] = f"{run_env['RUNS_DIR']}/{run_id}/phase_d_metrics.jsonl"
            if framework == "treerl":
                run_env["VDRA_TREERL_VERIFY_SYNC"] = "1"
                run_env["VDRA_TREERL_SYNC_REPORT"] = f"{run_env['RESULTS_DIR']}/equivalence/treerl_sync_{run_id}.json"
            result = _run_training(exp, ["42"], run_env)
            results.append(result)
            if result.get("status") != "complete":
                break
    finally:
        if previous_call_id is None:
            os.environ.pop("MODAL_FUNCTION_CALL_ID", None)
        else:
            os.environ["MODAL_FUNCTION_CALL_ID"] = previous_call_id

    validation = None
    resume_results = []
    if len(results) == 2 and all(r.get("status") == "complete" for r in results):
        try:
            for exp in MOCK_PAIRS[framework]:
                run_id = f"{exp}_s42"
                _build_reference_artifact(
                    exp,
                    FRAMEWORK_NAMES[framework],
                    Path(env["RUNS_DIR"]) / run_id / "phase_d_metrics.jsonl",
                    Path(env["RESULTS_DIR"]),
                )
            validation = _validate_pair(
                framework,
                Path(env["RUNS_DIR"]),
                Path(env["RESULTS_DIR"]),
                model_path=env["MODEL_PATH"],
                dataset_specs=_phase_d_dataset_specs(env),
            )
            for exp in MOCK_PAIRS[framework]:
                resumed = _interrupt_and_resume(
                    exp,
                    FRAMEWORK_NAMES[framework],
                    str(manifest["model_path"]),
                    platform_call_id,
                    root,
                )
                resume_results.append(resumed)
        except Exception as exc:
            return {
                "framework": framework,
                "pair": MOCK_PAIRS[framework],
                "runs": results,
                "resumed_runs": resume_results,
                "status": "failed",
                "validation_error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "framework": framework,
        "pair": MOCK_PAIRS[framework],
        "runs": results,
        "resumed_runs": resume_results,
        "validation": validation,
        "status": "complete"
        if len(results) == 2
        and all(r.get("status") == "complete" for r in results)
        and len(resume_results) == 2
        else "failed",
    }


def _run_pair(
    framework: str,
    *,
    root: Path = PHASE_D_ROOT,
    igpo_max_num_seqs: int = 64,
) -> dict[str, Any]:
    if framework != "treerl":
        return _run_pair_impl(
            framework,
            root=root,
            igpo_max_num_seqs=igpo_max_num_seqs,
        )
    sampler = _GpuMemorySampler()
    sampler.start()
    try:
        result = _run_pair_impl(
            framework,
            root=root,
            igpo_max_num_seqs=igpo_max_num_seqs,
        )
    finally:
        memory_report = sampler.stop()
        report_path = Path(root) / "results" / "treerl_gpu_memory.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(memory_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not memory_report["fits"]:
            raise RuntimeError(f"TreeRL peak GPU memory did not fit: {memory_report}")
    result["gpu_memory"] = memory_report
    return result


@app.function(
    image=IMAGES["igpo"],
    cpu=1,
    timeout=5 * 60,
    retries=0,
    scaledown_window=2,
)
def preflight_igpo_eval_config() -> dict[str, Any]:
    """CPU-only config preflight so a bad eval model path cannot burn B200 time."""
    return _igpo_eval_config_preflight()


def _igpo_logprob_profile_environment(
    root: Path,
    model_path: str,
    igpo_max_num_seqs: int,
) -> dict[str, str]:
    """Build an isolated one-update IGPO profile environment for a sequence-limit arm."""
    env = mock_environment(root, igpo_max_num_seqs=igpo_max_num_seqs)
    env.update(
        {
            "MODEL_PATH": model_path,
            "RUNS_DIR": str(root / "runs"),
            "RESULTS_DIR": str(root / "results"),
            "BANKS_DIR": str(root / "banks"),
            "STOP_STEP": "1",
            "EVAL_EVERY_N": "0",
            "EVAL_AT_END": "0",
            "EVAL_STEP0": "0",
            "RESUME_EVERY": "1",
            "KEEP_EVERY": "1",
            "VDRA_METRICS_JSONL": str(root / "runs" / "R1_s42" / "phase_d_metrics.jsonl"),
        }
    )
    return env


@app.function(
    image=IMAGES["igpo"],
    gpu="B200",
    timeout=IGPO_PROFILE_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def profile_igpo_r1_logprobs(
    igpo_max_num_seqs: int = 64,
    run_tag: str = "",
) -> dict[str, Any]:
    """Profile first-batch latency in a fresh, isolated one-update IGPO R1 run."""
    igpo_max_num_seqs = _validate_igpo_max_num_seqs(igpo_max_num_seqs)
    run_tag = run_tag or f"profile{uuid.uuid4().hex[:8]}"
    if not run_tag.isalnum():
        raise ValueError("IGPO logprob profile run_tag must be alphanumeric")
    root = phase_d_run_root(run_tag, igpo_max_num_seqs=igpo_max_num_seqs)
    if root.exists():
        raise FileExistsError(f"IGPO logprob profile root already exists: {root}")
    manifest = _runtime_manifest()
    if not manifest.get("ready"):
        raise RuntimeError("run setup_data() successfully before the IGPO profile")
    env = _igpo_logprob_profile_environment(
        root,
        str(manifest["model_path"]),
        igpo_max_num_seqs,
    )
    result = _run_training("R1", ["42"], env)
    return {
        **result,
        "profile_root": str(root),
        "run_tag": run_tag,
        "igpo_max_num_seqs": igpo_max_num_seqs,
    }


@app.function(
    image=IMAGES["igpo"],
    gpu="B200",
    timeout=IGPO_R1_REFERENCE_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def run_igpo_r1_reference_to_step2() -> dict[str, Any]:
    """Train a clean IGPO baseline chunk through update 2 and record reference metrics."""
    _check_live_phase_d_budget(
        reserve_seconds=IGPO_R1_REFERENCE_TIMEOUT_SECONDS
    )
    _check_phase_d_budget(
        _phase_d_cost_so_far(), reserve_seconds=IGPO_R1_REFERENCE_TIMEOUT_SECONDS
    )
    manifest = _runtime_manifest()
    if not manifest.get("ready"):
        raise RuntimeError("run setup_data() successfully before the IGPO reference run")
    reference_root = PHASE_D_ROOT / "r1_reference_runs" / uuid.uuid4().hex
    env = mock_environment(reference_root)
    env.update(
        {
            "MODEL_PATH": str(manifest["model_path"]),
            "STOP_STEP": "2",
            "EVAL_EVERY_N": "0",
            "EVAL_AT_END": "0",
            "EVAL_STEP0": "0",
            "RESUME_EVERY": "2",
            "KEEP_EVERY": "2",
            "VDRA_METRICS_JSONL": str(
                reference_root / "runs" / "R1_s42" / "phase_d_metrics.jsonl"
            ),
        }
    )
    result = _run_training("R1", ["42"], env)
    if result.get("status") != "complete":
        return {**result, "reference_root": str(reference_root)}
    reference = _build_reference_artifact(
        "R1",
        FRAMEWORK_NAMES["igpo"],
        Path(env["RUNS_DIR"]) / "R1_s42" / "phase_d_metrics.jsonl",
        Path(env["RESULTS_DIR"]),
    )
    return {**result, "reference": reference, "reference_root": str(reference_root)}


@app.function(
    image=IMAGES["igpo"],
    gpu="B200",
    timeout=IGPO_PAIR_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def mock_igpo_pair(
    igpo_max_num_seqs: int = 64,
    run_tag: str = "",
) -> dict[str, Any]:
    root = phase_d_run_root(
        run_tag,
        igpo_max_num_seqs=igpo_max_num_seqs,
    )
    if run_tag and root.exists():
        raise FileExistsError(f"Phase D run root already exists: {root}")
    return _run_pair(
        "igpo",
        root=root,
        igpo_max_num_seqs=igpo_max_num_seqs,
    )


@app.function(
    image=IMAGES["treepo"],
    gpu="B200",
    timeout=MOCK_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def mock_treepo_pair() -> dict[str, Any]:
    return _run_pair("treepo")


@app.function(
    image=IMAGES["treerl"],
    gpu="B200",
    timeout=MOCK_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def mock_treerl_pair() -> dict[str, Any]:
    return _run_pair("treerl")


@app.local_entrypoint()
def main(
    framework: str = "all",
    igpo_image_preflight: bool = False,
    igpo_logprob_profile: bool = False,
    igpo_r1_reference: bool = False,
    igpo_max_num_seqs: int = 64,
    run_tag: str = "",
) -> None:
    functions = {
        "igpo": mock_igpo_pair,
        "treepo": mock_treepo_pair,
        "treerl": mock_treerl_pair,
    }
    selected = list(functions) if framework == "all" else [framework]
    if any(name not in functions for name in selected):
        raise ValueError("framework must be all, igpo, treepo, or treerl")
    _validate_igpo_max_num_seqs(igpo_max_num_seqs)
    tuning_requested = run_tag or igpo_max_num_seqs != 64
    if tuning_requested and framework != "igpo":
        raise ValueError("IGPO sequence tuning requires framework=igpo")
    phase_d_run_root(
        run_tag,
        igpo_max_num_seqs=igpo_max_num_seqs,
    )
    selected_special_modes = sum(
        (igpo_image_preflight, igpo_logprob_profile, igpo_r1_reference)
    )
    if selected_special_modes > 1:
        raise ValueError("choose only one special IGPO run mode")
    if selected_special_modes and tuning_requested and not (
        igpo_logprob_profile and bool(run_tag)
    ):
        raise ValueError(
            "IGPO sequence tuning with a special mode requires an isolated logprob profile run_tag"
        )
    if igpo_image_preflight:
        if framework not in {"all", "igpo"}:
            raise ValueError("the IGPO image preflight cannot be combined with another framework")
        metered_cost = _check_live_phase_d_budget(reserve_seconds=5 * 60)
        print(
            json.dumps(
                {
                    "image_preflight": "igpo-training-image",
                    "metered_cost_usd": metered_cost,
                    "timeout_seconds": 5 * 60,
                }
            )
        )
        result = preflight_igpo_eval_config.remote()
        print(json.dumps({"igpo_image_preflight": result}, indent=2, sort_keys=True))
        return
    if igpo_r1_reference:
        if framework not in {"all", "igpo"}:
            raise ValueError("the clean IGPO R1 reference cannot be combined with another framework")
        metered_cost = _check_live_phase_d_budget(
            reserve_seconds=IGPO_R1_REFERENCE_TIMEOUT_SECONDS
        )
        reserved = (
            IGPO_R1_REFERENCE_TIMEOUT_SECONDS * GPU_RATE_USD_PER_SECOND
            + PHASE_D_NON_GPU_RESERVE_USD
        )
        print(
            json.dumps(
                {
                    "reference_run": "igpo-r1-clean-through-update2",
                    "metered_cost_usd": metered_cost,
                    "reserved_usd": reserved,
                    "timeout_seconds": IGPO_R1_REFERENCE_TIMEOUT_SECONDS,
                }
            )
        )
        result = run_igpo_r1_reference_to_step2.spawn().get()
        print(json.dumps({"igpo_r1_reference": result}, indent=2, sort_keys=True))
        if result.get("status") != "complete":
            raise SystemExit("clean IGPO R1 reference run did not complete")
        return
    if igpo_logprob_profile:
        if framework not in {"all", "igpo"}:
            raise ValueError("the IGPO logprob profile cannot be combined with another framework")
        metered_cost = _check_live_phase_d_budget(
            reserve_seconds=IGPO_PROFILE_TIMEOUT_SECONDS
        )
        reserved = (
            IGPO_PROFILE_TIMEOUT_SECONDS * GPU_RATE_USD_PER_SECOND
            + PHASE_D_NON_GPU_RESERVE_USD
        )
        print(
            json.dumps(
                {
                    "profile_run": "igpo-r1-first-logprob-batches",
                    "igpo_max_num_seqs": igpo_max_num_seqs,
                    "run_tag": run_tag or "<generated-isolated-tag>",
                    "metered_cost_usd": metered_cost,
                    "reserved_usd": reserved,
                    "timeout_seconds": IGPO_PROFILE_TIMEOUT_SECONDS,
                }
            )
        )
        result = profile_igpo_r1_logprobs.spawn(
            igpo_max_num_seqs,
            run_tag,
        ).get()
        print(json.dumps({"igpo_logprob_profile": result}, indent=2, sort_keys=True))
        if result.get("status") != "complete":
            raise SystemExit("IGPO logprob profile run did not complete")
        return
    results = {}
    for name in selected:
        reserve_seconds = (
            IGPO_PAIR_TIMEOUT_SECONDS if name == "igpo" else MOCK_TIMEOUT_SECONDS
        )
        metered_cost = _check_live_phase_d_budget(reserve_seconds=reserve_seconds)
        print(json.dumps({"live_metered_cost_usd": metered_cost, "gpu_run": name}))
        if name == "igpo":
            config_check = preflight_igpo_eval_config.remote()
            print(json.dumps({"igpo_eval_config_preflight": config_check}, indent=2, sort_keys=True))
        if name == "igpo":
            result = functions[name].spawn(
                igpo_max_num_seqs,
                run_tag,
            ).get()
        else:
            result = functions[name].spawn().get()
        results[name] = result
        if result.get("status") != "complete":
            break
    print(json.dumps(results, indent=2, sort_keys=True))
    if not _all_pairs_complete(results):
        raise SystemExit("one or more Phase D mock pairs failed")
