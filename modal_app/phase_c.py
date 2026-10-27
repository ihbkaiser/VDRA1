"""Persistent data preparation and resumable B200 orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import modal
from modal.exception import FunctionTimeoutError

from modal_app.images import framework_images
from modal_app.script_utils import eval_dataset_download_args, normalize_shell_script
from modal_app.telemetry import append_jsonl, run_start_benchmark


APP_NAME = "vdra-experiments"
VOLUME_NAME = "vdra-work"
WORK = Path("/work")
WORK_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
CONTROL = modal.Dict.from_name("vdra-orchestration-state", create_if_missing=True)
GPU_RATE_USD_PER_SECOND = 0.001736
TRAIN_TIMEOUT_SECONDS = 23 * 60 * 60
APP_TIMEOUT_SECONDS = 24 * 60 * 60
PHASE_F_SMOKE_RUN_TIMEOUT_SECONDS = 10 * 60
PHASE_F_TREEPO_R4_TIMEOUT_SECONDS = 15 * 60
PHASE_F_RECOVERY_RUN_TIMEOUT_SECONDS = 10 * 60
PHASE_F_NULL_TIMEOUT_SECONDS = 65 * 60
PHASE_F_TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT = "600"
PHASE_F_TREERL_VLLM_MEM = "0.25"
PHASE_F_TREERL_VLLM_BATCHED_TOKENS = "16384"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
EXPECTED_TRAIN_ROWS = 11244
_CONTAINER_PROFILE_CACHE: dict[str, tuple[dict[str, Any], str | None]] = {}
_CONTAINER_PROFILE_CACHE_LOCK = threading.Lock()
EVAL_DATASETS = ("aime24", "aime25", "amc23", "olympiadbench_hf")
LEAKAGE_AUDIT_SHA256 = "c91ffe55075dc0e596e83cb94e6afc7bafc81f9853bf6174f088959436028a66"
EXPERIMENTS = {
    "R1": ("R1_spo.sh", "igpo"),
    "R2": ("R2_spo_vdra.sh", "igpo"),
    "R3": ("R3_treepo.sh", "treepo"),
    "R4": ("R4_treepo_vdra.sh", "treepo"),
    "R5": ("R5_treerl.sh", "treerl"),
    "R6": ("R6_treerl_vdra.sh", "treerl"),
}
PAIRS = {"spo": ("R1", "R2", "igpo"), "treepo": ("R3", "R4", "treepo"), "treerl": ("R5", "R6", "treerl")}
IMAGES = framework_images()
app = modal.App(APP_NAME)


def _load_completed_phase_f_r2_tv(root: Path) -> dict[str, Any] | None:
    """Reuse R2-TV only when the trainer, checkpoint, timing, and queue audit passed."""
    report_path = root / "results" / "phase_f_null_remainder.json"
    run_dir = root / "null_R2_tv" / "runs" / "R2_s42"
    run_meta_path = run_dir / "run_meta.jsonl"
    timing_path = run_dir / "training_timing.jsonl"
    checkpoint_path = run_dir / "checkpoints" / "ckpt--iter_0002--epoch_1.00--step_0004"
    if not all(path.is_file() for path in (report_path, run_meta_path, timing_path)):
        return None
    if not run_dir.is_dir() or not _phase_f_checkpoint_has_weights(checkpoint_path):
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    audits = report.get("audits")
    if not isinstance(audits, dict):
        return None
    run = report.get("null_R2_tv")
    audit = audits.get("null_R2_tv")
    if not isinstance(run, dict) or not isinstance(audit, dict):
        return None
    if (
        run.get("experiment") != "R2"
        or run.get("status") != "complete"
        or audit.get("status") == "failed"
        or audit.get("audit_exit_code", 0) != 0
        or audit.get("budget_evidence_complete") is not True
        or audit.get("gate_G1_budget_ok") is not True
        or type(audit.get("budget_violations")) is not int
        or audit.get("budget_violations") != 0
    ):
        return None
    call_id = run.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return None
    try:
        run_meta = [json.loads(line) for line in run_meta_path.read_text(encoding="utf-8").splitlines()]
        timings = [json.loads(line) for line in timing_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return None
    run_meta = [item for item in run_meta if isinstance(item, dict)]
    timings = [item for item in timings if isinstance(item, dict)]
    training_attempt: dict[str, int | None] | None = None
    for index, item in enumerate(run_meta):
        if (
            item.get("event") == "start"
            and item.get("run") == "R2"
            and type(item.get("seed")) is int
            and item["seed"] == 42
            and type(item.get("stop")) is int
            and item["stop"] == 2
            and type(item.get("chunk_stop")) is int
            and item["chunk_stop"] == 2
        ):
            training_attempt = {"start": index, "end": None}
        elif (
            item.get("event") == "end"
            and training_attempt is not None
            and training_attempt["end"] is None
        ):
            training_attempt["end"] = (
                index if type(item.get("rc")) is int and item["rc"] == 0 else -1
            )
    if training_attempt is None or training_attempt["end"] is None or training_attempt["end"] < 0:
        return None
    successful_container_end = next(
        (
            index
            for index, item in enumerate(run_meta)
            if item.get("event") == "modal_container_end"
            and item.get("container_call_id") == call_id
            and item.get("status") == "complete"
            and item.get("error") is None
            and index > training_attempt["end"]
        ),
        None,
    )
    if successful_container_end is None:
        return None
    step_rows = [item for item in timings if type(item.get("global_step")) is int]
    if not step_rows or max(item["global_step"] for item in step_rows) != 2:
        return None
    if not any(
        item.get("global_step") == 2
        and type(item.get("iteration")) is int
        and item["iteration"] == 0
        and all(
            type(item.get(key)) in (int, float) and item[key] > 0
            for key in (
                "timing/iter/episode_generation_seconds",
                "timing/iter/training_step_seconds",
                "timing/iter/train_total_seconds",
            )
        )
        for item in step_rows
    ):
        return None
    return run


def _phase_f_checkpoint_has_weights(checkpoint_path: Path) -> bool:
    """Require a readable HF config and nonempty weight payload for a reusable checkpoint."""
    pretrained = checkpoint_path / "hf_pretrained"
    config_path = pretrained / "config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        return False
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".model"}
    try:
        return any(
            path.is_file() and path.suffix.lower() in weight_suffixes and path.stat().st_size > 0
            for path in pretrained.rglob("*")
        )
    except OSError:
        return False


def _phase_f_smokes_complete(results: dict[str, Any], required_labels) -> bool:
    """Return true only when every explicitly required Phase F arm completed."""
    return all(
        isinstance(results.get(label), dict)
        and results[label].get("status") == "complete"
        for label in required_labels
    )


def _phase_f_missing_baseline_dirs(baseline_run_dirs, *, r3_r6_only: bool) -> list[Path]:
    """The R3-R6-only smoke does not run the IGPO null-baseline audit."""
    if r3_r6_only:
        return []
    return [Path(path) for path in baseline_run_dirs if not Path(path).is_dir()]


def _phase_f_r3_r6_smokes_complete(results: dict[str, Any], audits) -> bool:
    """Check the R3-R6 method smokes without implying full Phase F certification."""
    required_labels = ("null_R3_spo", "null_R4_tv", "null_R5_spo", "null_R6_tv")
    if not _phase_f_smokes_complete(results, required_labels):
        return False
    if not _phase_f_audits_succeeded(audits):
        return False
    audit_map = audits if isinstance(audits, dict) else {}
    return all(
        audit_map.get(name, {}).get("gate_G1_budget_ok") is True
        for name in ("null_R4_tv", "null_R6_tv")
    )


def _phase_f_audits_succeeded(audits) -> bool:
    """Reject missing, failed, or nonzero-exit audit results."""
    values = audits.values() if isinstance(audits, dict) else audits
    values = list(values)
    return bool(values) and all(
        isinstance(audit, dict)
        and audit.get("status") != "failed"
        and audit.get("audit_exit_code", 0) == 0
        for audit in values
    )


def _run_phase_f_audit(command: list[str], cwd: str = "/opt/vdra") -> dict[str, Any]:
    """Keep a failing audit's JSON report in the aggregate even on exit status 1."""
    report_path = Path(command[-1])
    process_error = None
    try:
        subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.DEVNULL)
    except Exception as exc:
        process_error = f"{type(exc).__name__}: {exc}"
    try:
        saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "execution_error": process_error or f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(saved_report, dict):
        saved_report = {"saved_report": saved_report}
        process_error = process_error or "audit report JSON must be an object"
    if process_error is not None:
        return {**saved_report, "status": "failed", "execution_error": process_error}
    return saved_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_source_math() -> None:
    destination = WORK / "source_data/math"
    if destination.is_dir():
        return
    bootstrap = Path("/tmp/igpo-data-bootstrap")
    if bootstrap.exists():
        shutil.rmtree(bootstrap)
    shutil.copytree("/opt/work/igpo", bootstrap)
    bootstrap_script = bootstrap / "scripts/download_and_prepare_dataset.sh"
    normalize_shell_script(bootstrap_script)
    subprocess.run(["bash", str(bootstrap_script)], cwd=bootstrap, check=True)
    source = bootstrap / "data/math"
    if not source.is_dir():
        candidates = list((bootstrap / "data").glob("**/math"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"training archive did not produce exactly one data/math directory: {candidates}")
        source = candidates[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    WORK_VOLUME.commit()


def _runtime_manifest() -> dict[str, Any]:
    manifest_path = WORK / "setup/manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}


def _cached_model_path(work_root: Path = WORK) -> Path:
    return (
        Path(work_root)
        / "hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots"
        / MODEL_REVISION
    )


def _dataset_split_present(path: Path) -> bool:
    return (path / "state.json").is_file() and any(path.glob("data-*.arrow"))


def _existing_data_inputs_present(work_root: Path = WORK) -> bool:
    """Detect a complete cached setup before the data preparation script can rewrite it."""
    work_root = Path(work_root)
    files = [
        work_root / "data_vdra/train.parquet",
        work_root / "data_vdra/train_treerl.jsonl",
        work_root / "data_vdra/split_manifest.json",
        work_root / "data_vdra/diag256.jsonl",
        work_root / "data_vdra/math-nodiag/dataset_dict.json",
        work_root / "source_data/math/dataset_dict.json",
        _cached_model_path(work_root) / "config.json",
        _cached_model_path(work_root) / "model.safetensors",
        _cached_model_path(work_root) / "tokenizer_config.json",
        _cached_model_path(work_root) / "tokenizer.json",
    ]
    files.extend(work_root / "source_data" / name / marker for name in EVAL_DATASETS for marker in ("dataset_dict.json", "_source.json"))
    if not all(path.is_file() and path.stat().st_size > 0 for path in files):
        return False
    return all(
        _dataset_split_present(work_root / "data_vdra/math-nodiag" / split)
        for split in ("train", "validation", "test")
    )


def _register_existing_data(work_root: Path = WORK) -> dict[str, Any]:
    """Record existing exports and caches without running data preparation or writing them."""
    work_root = Path(work_root)
    if not _existing_data_inputs_present(work_root):
        raise RuntimeError("existing training/evaluation exports or pinned model cache are incomplete")

    data_dir = work_root / "data_vdra"
    split_path = data_dir / "split_manifest.json"
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    kept_rows = int(split_manifest.get("n_train_kept", -1))
    if kept_rows != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"expected {EXPECTED_TRAIN_ROWS} unchanged training rows, found {kept_rows}")
    treerl_rows = sum(1 for line in (data_dir / "train_treerl.jsonl").open(encoding="utf-8") if line.strip())
    if treerl_rows != kept_rows:
        raise RuntimeError(f"TreeRL training export has {treerl_rows} rows, split manifest has {kept_rows}")

    import pyarrow.parquet as parquet

    parquet_rows = parquet.ParquetFile(data_dir / "train.parquet").metadata.num_rows
    if parquet_rows != kept_rows:
        raise RuntimeError(f"Parquet training export has {parquet_rows} rows, split manifest has {kept_rows}")

    model_path = _cached_model_path(work_root)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, trust_remote_code=True)
    file_hashes = {
        name: _sha256(model_path / name)
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json")
        if (model_path / name).is_file()
    }
    export_hashes = {
        name: _sha256(data_dir / name)
        for name in ("train.parquet", "train_treerl.jsonl", "split_manifest.json", "diag256.jsonl")
    }
    code_commits = {}
    for name, path in (("igpo", "/opt/work/igpo"), ("treepo", "/opt/work/TreePO"), ("treerl", "/opt/work/TreeRL")):
        try:
            code_commits[name] = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        except subprocess.CalledProcessError:
            code_commits[name] = "unavailable"

    return {
        "ready": True,
        "created_at": time.time(),
        "setup_mode": "registered_existing_exports_without_rebuild",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "tokenizer_file_sha256": file_hashes,
        "training_data": "/work/source_data/math",
        "evaluation_data_dir": "/work/source_data",
        "split_manifest": "/work/data_vdra/split_manifest.json",
        "training_rows": kept_rows,
        "training_export_sha256": export_hashes,
        "evaluation_datasets_present": ["math500", *EVAL_DATASETS],
        "leakage_decision": {
            "corpus": "unchanged",
            "action": "accepted_known_overlaps_without_filtering",
            "normalized_exact_overlaps": 7,
            "notation_equivalent_overlaps": 1,
            "audit_sha256": LEAKAGE_AUDIT_SHA256,
        },
        "code_commits": code_commits,
    }


def _training_paths(env: dict[str, str] | None = None) -> tuple[Path, Path, Path, Path]:
    values = os.environ.copy()
    values.update({key: str(value) for key, value in (env or {}).items()})
    work_root = Path(values.get("WORK", str(WORK)))
    return (
        work_root,
        Path(values.get("RUNS_DIR", str(work_root / "runs"))),
        Path(values.get("RESULTS_DIR", str(work_root / "results"))),
        Path(values.get("BANKS_DIR", str(work_root / "banks"))),
    )


def _training_control_overrides(env_values: dict[str, str]) -> dict[str, str]:
    """Use per-call checkpoint/evaluation controls while retaining production defaults."""
    return {
        "MAX_RESTARTS": env_values.get("MAX_RESTARTS", "0"),
        "EVAL_EVERY_N": env_values.get("EVAL_EVERY_N", "25"),
        "EVAL_AT_END": env_values.get("EVAL_AT_END", "1"),
        "EVAL_STEP0": env_values.get("EVAL_STEP0", "1"),
    }


def _record_container_start(
    exp: str, seeds: list[str], image_id: str | None, runs_root: Path = WORK / "runs"
) -> dict[str, Any]:
    container_id = os.environ.get("MODAL_CONTAINER_ID") or os.environ.get("MODAL_TASK_ID")
    call_id = os.environ.get("MODAL_FUNCTION_CALL_ID")
    source_call_id = call_id
    reused_profile = False
    if container_id:
        with _CONTAINER_PROFILE_CACHE_LOCK:
            cached = _CONTAINER_PROFILE_CACHE.get(container_id)
            if cached is None:
                row = run_start_benchmark()
                _CONTAINER_PROFILE_CACHE[container_id] = (dict(row), call_id)
            else:
                row, source_call_id = cached
                row = dict(row)
                reused_profile = True
    else:
        row = run_start_benchmark()

    if reused_profile:
        row["event"] = "container_profile_reused"
        row["benchmark_source_time"] = row["time"]
        row["benchmark_source_call_id"] = source_call_id
        row["benchmark_measurement_seconds"] = row["benchmark_seconds"]
        row["time"] = time.time()
        row["benchmark_seconds"] = 0.0
        row["benchmark_reused"] = True
    else:
        row["benchmark_reused"] = False
    row["image_id"] = image_id or row.get("image_id")
    row["container_call_id"] = call_id
    row["container_id"] = container_id
    row["experiment"] = exp
    row["allocated_benchmark_cost_usd"] = (
        0.0
        if reused_profile
        else row["benchmark_seconds"] * GPU_RATE_USD_PER_SECOND / max(1, len(seeds))
    )
    for seed in seeds:
        run_id = f"{exp}_s{seed}"
        append_jsonl(runs_root / run_id / "run_meta.jsonl", dict(row, run=exp, seed=int(seed)))
    _commit_volume_serialized()
    return row


def _checkpoint_signature(
    exp: str, seeds: list[str], runs_root: Path = WORK / "runs"
) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    framework = EXPERIMENTS[exp][1]
    for seed in seeds:
        run_dir = runs_root / f"{exp}_s{seed}"
        if framework == "igpo":
            paths = sorted((run_dir / "checkpoints").glob("ckpt--iter_*"))
        elif framework == "treepo":
            paths = sorted((run_dir / "ckpts").glob("global_step_*"))
        else:
            paths = sorted((run_dir / "ckpt").glob("_actor_global_step*"))
        for checkpoint in paths:
            candidates = [checkpoint]
            if checkpoint.is_dir():
                candidates.extend(path for path in checkpoint.rglob("*") if path.is_file())
            for path in candidates:
                try:
                    st = path.stat()
                    signature.append((str(path), st.st_mtime_ns, st.st_size))
                except FileNotFoundError:
                    continue
    return tuple(signature)


def _commit_volume_serialized() -> None:
    owner = uuid.uuid4().hex
    key = "global-volume-commit-lock"
    while not CONTROL.put(key, owner, skip_if_exists=True):
        time.sleep(1.0)
    try:
        WORK_VOLUME.commit()
    finally:
        if CONTROL.get(key) == owner:
            CONTROL.pop(key, None)


def _wait_for_checkpoint_stability(
    exp: str,
    seeds: list[str],
    stop: threading.Event,
    stable_seconds: float = 30.0,
    max_wait_seconds: float = 300.0,
    runs_root: Path = WORK / "runs",
) -> tuple[tuple[str, int, int], ...]:
    signature = _checkpoint_signature(exp, seeds, runs_root)
    stable_since = time.monotonic()
    deadline = stable_since + max_wait_seconds
    while time.monotonic() < deadline:
        if stop.wait(10):
            return signature
        current = _checkpoint_signature(exp, seeds, runs_root)
        if current != signature:
            signature = current
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= stable_seconds:
            break
    return signature


def _checkpoint_commit_watcher(
    exp: str, seeds: list[str], stop: threading.Event, runs_root: Path = WORK / "runs"
) -> None:
    previous = _checkpoint_signature(exp, seeds, runs_root)
    last_commit = time.monotonic()
    while not stop.wait(20):
        current = _checkpoint_signature(exp, seeds, runs_root)
        if current != previous or time.monotonic() - last_commit >= 180:
            if current != previous:
                current = _wait_for_checkpoint_stability(exp, seeds, stop, runs_root=runs_root)
            _commit_volume_serialized()
            previous = current
            last_commit = time.monotonic()


def _run_training(exp: str, seeds: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    if exp not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {exp}")
    if not seeds or any(not seed.isdigit() for seed in seeds):
        raise ValueError("seeds must be a non-empty list of decimal integers")

    env_values = {key: str(value) for key, value in (env or {}).items()}
    work_root, runs_root, results_root, banks_root = _training_paths(env_values)
    call_id = os.environ.get("MODAL_FUNCTION_CALL_ID") or uuid.uuid4().hex
    session_dir = results_root / "modal_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / f"{call_id}.json"
    started = time.time()
    process_env = os.environ.copy()
    process_env.update(env_values)
    process_env.update(
        {
            "VDRA_HOME": "/opt/vdra",
            "WORK": str(work_root),
            "RUNS_DIR": str(runs_root),
            "RESULTS_DIR": str(results_root),
            "BANKS_DIR": str(banks_root),
            "DATA_DIR": "/work/data_vdra",
            "MATH_DICT": "/work/source_data/math",
            "IGPO_DIR": "/opt/work/igpo",
            "TREEPO_DIR": "/opt/work/TreePO",
            "TREERL_DIR": "/opt/work/TreeRL",
            "MODEL_PATH": str(_runtime_manifest().get("model_path", MODEL_ID)),
            "VDRA_B200": "1",
            "GPU": "0",
            "GPUS": "0",
            "GPUS_PER_RUN": "1",
            "GPU_GROUPS": "0",
            "SEEDS": " ".join(seeds),
            "COMPUTE_MATCHED": "0",
            "VDRA_SINGLE_RANK_COLLECTIVES": "1" if EXPERIMENTS[exp][1] in {"igpo", "treerl"} else "0",
            "WANDB_MODE": "offline",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **_training_control_overrides(env_values),
        }
    )
    image_id = os.environ.get("MODAL_IMAGE_ID")
    session: dict[str, Any] = {
        "call_id": call_id,
        "experiment": exp,
        "seeds": seeds,
        "gpu": "B200",
        "image_id": image_id,
        "started_at": started,
        "approx_cost_usd_per_second": GPU_RATE_USD_PER_SECOND,
        "status": "running",
    }
    session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

    _record_container_start(exp, seeds, image_id, runs_root)
    stop_watcher = threading.Event()
    watcher = threading.Thread(
        target=_checkpoint_commit_watcher, args=(exp, seeds, stop_watcher, runs_root), daemon=True
    )
    watcher.start()
    script = EXPERIMENTS[exp][0]
    error: str | None = None
    try:
        result = subprocess.run(["bash", f"/opt/vdra/scripts/exp/{script}"], env=process_env, check=False)
        status = "complete" if result.returncode == 0 else "failed"
        if result.returncode:
            error = f"launcher exited with code {result.returncode}"
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_watcher.set()
        watcher.join(timeout=30)
        _commit_volume_serialized()
        elapsed = time.time() - started
        session.update(
            {
                "ended_at": time.time(),
                "wall_seconds": elapsed,
                "approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND,
                "status": status if "status" in locals() else "failed",
                "error": error,
            }
        )
        session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
        for seed in seeds:
            append_jsonl(
                runs_root / f"{exp}_s{seed}" / "run_meta.jsonl",
                {
                    "event": "modal_container_end",
                    "time": time.time(),
                    "container_call_id": call_id,
                    "wall_seconds": elapsed,
                    "approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND,
                    "allocated_approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND / len(seeds),
                    "status": session["status"],
                    "error": error,
                },
            )
    return {"experiment": exp, "seeds": seeds, "status": status, "error": error, "call_id": call_id}


@app.function(
    image=IMAGES["treepo"],
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def setup_data() -> dict[str, Any]:
    """Download the public source data and model once, then create shared exports."""
    setup_dir = WORK / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = setup_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if existing.get("ready") and (WORK / "data_vdra/split_manifest.json").is_file():
        return existing

    if _existing_data_inputs_present(WORK):
        manifest = _register_existing_data(WORK)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (WORK / "hf_cache/.setup_ready").write_text(f"{MODEL_ID}@{MODEL_REVISION}\n", encoding="utf-8")
        WORK_VOLUME.commit()
        return manifest

    _prepare_source_math()
    subprocess.run(
        eval_dataset_download_args(
            sys.executable, "/opt/work/igpo/scripts/download_eval_datasets.py", "/work/source_data"
        ),
        check=True,
    )
    WORK_VOLUME.commit()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_sha = MODEL_REVISION
    model_path = snapshot_download(MODEL_ID, revision=model_sha, cache_dir="/work/hf_cache")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer_file_hashes = {}
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json"):
        path = Path(model_path) / filename
        if path.is_file():
            tokenizer_file_hashes[filename] = _sha256(path)

    setup_env = os.environ.copy()
    setup_env.update(
        {
            "VDRA_HOME": "/opt/vdra",
            "WORK": "/work",
            "RUNS_DIR": "/work/runs",
            "RESULTS_DIR": "/work/results",
            "BANKS_DIR": "/work/banks",
            "DATA_DIR": "/work/data_vdra",
            "MATH_DICT": "/work/source_data/math",
            "MODEL_PATH": model_path,
            "IGPO_DIR": "/opt/work/igpo",
            "TREEPO_DIR": "/opt/work/TreePO",
            "TREERL_DIR": "/opt/work/TreeRL",
            "VDRA_B200": "1",
        }
    )
    subprocess.run(["bash", "/opt/vdra/scripts/01_prepare_data.sh"], env=setup_env, check=True)

    manifest = {
        "ready": True,
        "created_at": time.time(),
        "model_id": MODEL_ID,
        "model_revision": model_sha,
        "model_path": model_path,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "tokenizer_file_sha256": tokenizer_file_hashes,
        "training_data": "/work/source_data/math",
        "evaluation_data_dir": "/work/source_data",
        "split_manifest": "/work/data_vdra/split_manifest.json",
        "code_commits": {},
    }
    for name, path in (("igpo", "/opt/work/igpo"), ("treepo", "/opt/work/TreePO"), ("treerl", "/opt/work/TreeRL")):
        try:
            manifest["code_commits"][name] = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            manifest["code_commits"][name] = "unavailable"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (WORK / "hf_cache/.setup_ready").write_text(f"{MODEL_ID}@{model_sha}\n", encoding="utf-8")
    WORK_VOLUME.commit()
    return manifest


@app.function(
    image=IMAGES["igpo"],
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def register_migrated_data() -> dict[str, Any]:
    """Fetch the pinned model and register already-prepared, unchanged exports."""
    data_dir = WORK / "data_vdra"
    required_exports = [
        data_dir / "train.parquet",
        data_dir / "train_treerl.jsonl",
        data_dir / "split_manifest.json",
        data_dir / "diag256.jsonl",
        data_dir / "math-nodiag/dataset_dict.json",
        WORK / "source_data/math/dataset_dict.json",
    ]
    required_exports.extend(
        WORK / "source_data" / name / marker
        for name in EVAL_DATASETS
        for marker in ("dataset_dict.json", "_source.json")
    )
    missing = [str(path) for path in required_exports if not path.is_file()]
    if missing:
        raise RuntimeError(f"migrated exports are incomplete; missing {missing}")
    missing_splits = [
        split
        for split in ("train", "validation", "test")
        if not _dataset_split_present(data_dir / "math-nodiag" / split)
    ]
    if missing_splits:
        raise RuntimeError(f"migrated math-nodiag export lacks splits {missing_splits}")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir="/work/hf_cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if not _existing_data_inputs_present(WORK):
        raise RuntimeError("the pinned model or migrated exports are incomplete after download")

    manifest = _register_existing_data(WORK)
    manifest["tokenizer_class"] = type(tokenizer).__name__
    manifest["tokenizer_name_or_path"] = tokenizer.name_or_path
    manifest_path = WORK / "setup/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ready_marker = WORK / "hf_cache/.setup_ready"
    ready_marker.parent.mkdir(parents=True, exist_ok=True)
    ready_marker.write_text(f"{MODEL_ID}@{MODEL_REVISION}\n", encoding="utf-8")
    WORK_VOLUME.commit()
    return manifest


def train(exp: str, seeds: list[str], env: dict[str, str] | None = None) -> modal.FunctionCall[Any]:
    """Submit an experiment to the image pinned for its framework."""
    if exp not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {exp}")
    function_name = {"igpo": train_igpo, "treepo": train_treepo, "treerl": train_treerl}[EXPERIMENTS[exp][1]]
    return function_name.spawn(exp, seeds, env or {})


@app.function(
    image=IMAGES["igpo"],
    gpu="B200",
    timeout=TRAIN_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def train_igpo(exp: str, seeds: list[str], env: dict[str, str]) -> dict[str, Any]:
    if EXPERIMENTS.get(exp, (None, None))[1] != "igpo":
        raise ValueError(f"{exp} does not use IGPO")
    return _run_training(exp, [str(s) for s in seeds], env)


@app.function(
    image=IMAGES["treepo"],
    gpu="B200",
    timeout=TRAIN_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def train_treepo(exp: str, seeds: list[str], env: dict[str, str]) -> dict[str, Any]:
    if EXPERIMENTS.get(exp, (None, None))[1] != "treepo":
        raise ValueError(f"{exp} does not use TreePO")
    return _run_training(exp, [str(s) for s in seeds], env)


@app.function(
    image=IMAGES["treerl"],
    gpu="B200",
    timeout=TRAIN_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def train_treerl(exp: str, seeds: list[str], env: dict[str, str]) -> dict[str, Any]:
    if EXPERIMENTS.get(exp, (None, None))[1] != "treerl":
        raise ValueError(f"{exp} does not use TreeRL")
    return _run_training(exp, [str(s) for s in seeds], env)


@app.function(
    image=IMAGES["igpo"], gpu="B200", timeout=PHASE_F_SMOKE_RUN_TIMEOUT_SECONDS,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_igpo_smoke(exp: str, env: dict[str, str]) -> dict[str, Any]:
    if exp not in {"R1", "R2"}:
        raise ValueError(f"{exp} does not use IGPO")
    return _run_training(exp, ["42"], env)


@app.function(
    image=IMAGES["igpo"], gpu="B200", timeout=PHASE_F_RECOVERY_RUN_TIMEOUT_SECONDS,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_igpo_recover_smoke(exp: str, env: dict[str, str]) -> dict[str, Any]:
    if exp != "R2":
        raise ValueError(f"{exp} is not the IGPO VDRA recovery arm")
    return _run_training(exp, ["42"], env)


@app.function(
    image=IMAGES["treepo"], gpu="B200", timeout=PHASE_F_SMOKE_RUN_TIMEOUT_SECONDS,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_treepo_smoke(exp: str, env: dict[str, str]) -> dict[str, Any]:
    if exp != "R3":
        raise ValueError(f"{exp} does not use the 10-minute TreePO smoke")
    return _run_training(exp, ["42"], env)


@app.function(
    image=IMAGES["treepo"], gpu="B200", timeout=PHASE_F_TREEPO_R4_TIMEOUT_SECONDS,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_treepo_r4_smoke(exp: str, env: dict[str, str]) -> dict[str, Any]:
    if exp != "R4":
        raise ValueError(f"{exp} is not the TreePO VDRA smoke arm")
    return _run_training(exp, ["42"], env)


def _phase_f_treepo_smoke_function(exp: str):
    if exp == "R3":
        return phase_f_treepo_smoke
    if exp == "R4":
        return phase_f_treepo_r4_smoke
    raise ValueError(f"{exp} does not use TreePO")


@app.function(
    image=IMAGES["treerl"], gpu="B200", timeout=PHASE_F_SMOKE_RUN_TIMEOUT_SECONDS,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_treerl_smoke(exp: str, env: dict[str, str]) -> dict[str, Any]:
    if exp not in {"R5", "R6"}:
        raise ValueError(f"{exp} does not use TreeRL")
    return _run_training(exp, ["42"], env)


@app.function(
    image=IMAGES["treepo"], gpu="B200", timeout=PHASE_F_TREEPO_R4_TIMEOUT_SECONDS,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_treepo_one_update(exp: str, env: dict[str, str]) -> dict[str, Any]:
    if exp != "R4":
        raise ValueError(f"{exp} is not the TreePO VDRA smoke arm")
    return _run_training(exp, ["42"], env)


@app.function(
    image=IMAGES["treerl"], gpu="B200", timeout=10 * 60,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_treerl_one_update(exp: str, env: dict[str, str]) -> dict[str, Any]:
    if exp != "R6":
        raise ValueError(f"{exp} is not the TreeRL VDRA smoke arm")
    return _run_training(exp, ["42"], env)


@app.function(
    image=IMAGES["igpo"], cpu=2, memory=16384, timeout=PHASE_F_NULL_TIMEOUT_SECONDS,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_null_smoke(run_tag: str) -> dict[str, Any]:
    """Run the five Phase F null arms serially with a hard timeout per B200 task."""
    if not run_tag or not run_tag.isalnum():
        raise ValueError("run_tag must contain only letters and digits")
    root = WORK / "mock" / f"phase_f_null_{run_tag}"
    shared_env = {
        "STOP_STEP": "2",
        "RESUME_EVERY": "1",
        "KEEP_EVERY": "1",
        "MAX_RESTARTS": "0",
        "EVAL_EVERY_N": "0",
        "EVAL_AT_END": "0",
        "EVAL_STEP0": "0",
        "COMPUTE_MATCHED": "0",
        "SEEDS": "42",
        "WORK": "/work",
        "DATA_DIR": "/work/data_vdra",
        "MATH_DICT": "/work/source_data/math",
        "VDRA_B200": "1",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VDRA_CONFIG_JSON": "{}",
        "INGPO_DATASET_NUM_SAMPLES_PER_ITERATION": "4",
        "INGPO_NUM_EPISODES_PER_ITERATION": "64",
        "INGPO_TARGET_TRAIN_BATCH_SIZE": "32",
        "INGPO_VLLM_GPU_MEMORY_UTILIZATION": "0.25",
        "INGPO_VLLM_MAX_NUM_SEQS": "64",
        "INGPO_MAX_CONCURRENT_PROGRAMS": "16",
        "INGPO_MAX_CONCURRENT_GENERATIONS": "64",
        "INGPO_DISABLE_INTERNAL_EVALUATION": "1",
        "TREEPO_WIDTH": "2",
        "TREEPO_GEN_BATCH_SIZE": "8",
        "TREEPO_TRAIN_BATCH_SIZE": "8",
        "TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT": PHASE_F_TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT,
        "TREERL_VLLM_MEM": PHASE_F_TREERL_VLLM_MEM,
        "TREERL_VLLM_BATCHED_TOKENS": PHASE_F_TREERL_VLLM_BATCHED_TOKENS,
        "TREERL_M": "2",
        "TREERL_N": "1",
        "TREERL_L": "1",
        "TREERL_T": "1",
    }
    specs = [
        ("null_R2_constant", "R2", '{"proxy":"constant"}'),
        ("null_R1", "R1", "{}"),
        ("null_R2_tv", "R2", "{}"),
        ("null_R4_tv", "R4", "{}"),
        ("null_R6_tv", "R6", "{}"),
    ]
    smoke_functions = {
        "igpo": phase_f_igpo_smoke,
        "treerl": phase_f_treerl_smoke,
    }
    results: dict[str, Any] = {}
    run_dirs: dict[str, str] = {}
    for label, exp, variant in specs:
        run_root = root / label
        env = {
            **shared_env,
            "RUNS_DIR": str(run_root / "runs"),
            "RESULTS_DIR": str(run_root / "results"),
            "BANKS_DIR": str(run_root / "banks"),
            "VDRA_VARIANT_JSON": variant,
        }
        framework = EXPERIMENTS[exp][1]
        function = (
            _phase_f_treepo_smoke_function(exp)
            if framework == "treepo"
            else smoke_functions[framework]
        )
        try:
            result = function.spawn(exp, env).get()
        except Exception as exc:
            results[label] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            break
        results[label] = result
        run_dirs[label] = str(run_root / "runs" / f"{exp}_s42")
        if result.get("status") != "complete":
            break

    if len(results) == len(specs) and all(item.get("status") == "complete" for item in results.values()):
        audit_dir = root / "results" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        commands = [
            [sys.executable, "/opt/vdra/analysis/budget_audit.py", "--null",
             run_dirs["null_R2_constant"], run_dirs["null_R1"], "--out", str(audit_dir / "null_igpo.json")],
            *[
                [sys.executable, "/opt/vdra/analysis/budget_audit.py", "--require-budget-evidence",
                 "--logs", run_dirs[label], "--out", str(audit_dir / f"{label}.json")]
                for label in ("null_R2_tv", "null_R4_tv", "null_R6_tv")
            ],
        ]
        audit_results = []
        for command in commands:
            audit_results.append(_run_phase_f_audit(command))
        results["audits"] = audit_results
        results["status"] = (
            "complete"
            if _phase_f_audits_succeeded(audit_results)
            and audit_results[0].get("pass") is True
            and all(audit.get("gate_G1_budget_ok") is True for audit in audit_results[1:])
            else "failed"
        )
    else:
        results["status"] = "failed"

    report_path = root / "results" / "phase_f_null_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    WORK_VOLUME.commit()
    return {"root": str(root), **results}


@app.function(
    image=IMAGES["igpo"], cpu=2, memory=16384, timeout=PHASE_F_NULL_TIMEOUT_SECONDS,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_remaining_smoke(
    run_tag: str, baseline_run_tag: str = "1629e2315f", r3_r6_only: bool = False
) -> dict[str, Any]:
    """Recover R2-TV if needed, then smoke R3-R6; optional R3-R6-only partial run."""
    if not run_tag or not run_tag.isalnum():
        raise ValueError("run_tag must contain only letters and digits")
    if not baseline_run_tag or not baseline_run_tag.isalnum():
        raise ValueError("baseline_run_tag must contain only letters and digits")
    root = WORK / "mock" / f"phase_f_null_{run_tag}"
    baseline_root = WORK / "mock" / f"phase_f_null_{baseline_run_tag}"
    if r3_r6_only and root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"R3-R6-only smoke root already contains artifacts: {root}; use a fresh run tag"
        )
    baseline_run_dirs = (
        baseline_root / "null_R1" / "runs" / "R1_s42",
        baseline_root / "null_R2_constant" / "runs" / "R2_s42",
    )
    missing_baselines = _phase_f_missing_baseline_dirs(
        baseline_run_dirs, r3_r6_only=r3_r6_only
    )
    if missing_baselines:
        raise RuntimeError(
            f"Phase F null baseline artifacts are missing under {baseline_root}: "
            f"{missing_baselines}; "
            "refusing to spend on an audit that cannot be evaluated"
        )
    shared_env = {
        "STOP_STEP": "2",
        "RESUME_EVERY": "1",
        "KEEP_EVERY": "1",
        "MAX_RESTARTS": "0",
        "EVAL_EVERY_N": "0",
        "EVAL_AT_END": "0",
        "EVAL_STEP0": "0",
        "COMPUTE_MATCHED": "0",
        "SEEDS": "42",
        "WORK": "/work",
        "DATA_DIR": "/work/data_vdra",
        "MATH_DICT": "/work/source_data/math",
        "VDRA_B200": "1",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VDRA_CONFIG_JSON": "{}",
        "INGPO_DATASET_NUM_SAMPLES_PER_ITERATION": "4",
        "INGPO_NUM_EPISODES_PER_ITERATION": "64",
        "INGPO_TARGET_TRAIN_BATCH_SIZE": "32",
        "INGPO_VLLM_GPU_MEMORY_UTILIZATION": "0.25",
        "INGPO_VLLM_MAX_NUM_SEQS": "64",
        "INGPO_MAX_CONCURRENT_PROGRAMS": "16",
        "INGPO_MAX_CONCURRENT_GENERATIONS": "64",
        "INGPO_DISABLE_INTERNAL_EVALUATION": "1",
        "TREEPO_WIDTH": "2",
        "TREEPO_GEN_BATCH_SIZE": "8",
        "TREEPO_TRAIN_BATCH_SIZE": "8",
        "TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT": PHASE_F_TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT,
        "TREERL_VLLM_MEM": PHASE_F_TREERL_VLLM_MEM,
        "TREERL_VLLM_BATCHED_TOKENS": PHASE_F_TREERL_VLLM_BATCHED_TOKENS,
        "TREERL_M": "2",
        "TREERL_N": "1",
        "TREERL_L": "1",
        "TREERL_T": "1",
    }
    results: dict[str, Any] = {}
    run_dirs: dict[str, str] = {}
    prior_r2_tv = None if r3_r6_only else _load_completed_phase_f_r2_tv(root)
    if not r3_r6_only and prior_r2_tv is None:
        r2_run_dir = root / "null_R2_tv" / "runs" / "R2_s42"
        if root.exists() and any(root.rglob("*")):
            raise RuntimeError(
                f"Unverified Phase F artifacts already occupy {root}; "
                "use a fresh Phase F run tag to preserve them"
            )
        recovery_root = root / "null_R2_tv"
        recovery_env = {
            **shared_env,
            "RUNS_DIR": str(recovery_root / "runs"),
            "RESULTS_DIR": str(recovery_root / "results"),
            "BANKS_DIR": str(recovery_root / "banks"),
            "VDRA_VARIANT_JSON": "{}",
        }
        try:
            recovery = phase_f_igpo_recover_smoke.spawn("R2", recovery_env).get()
        except Exception as exc:
            recovery = {
                "experiment": "R2",
                "seeds": ["42"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        recovery["reused_saved_run_path"] = False
        results["null_R2_tv"] = recovery
        run_dirs["null_R2_tv"] = str(r2_run_dir)
        recovery_audit_dir = root / "results" / "audit"
        recovery_audit_dir.mkdir(parents=True, exist_ok=True)
        recovery_audit_path = recovery_audit_dir / "null_R2_tv.json"
        recovery_audit = _run_phase_f_audit(
            [
                sys.executable,
                "/opt/vdra/analysis/budget_audit.py",
                "--require-budget-evidence",
                "--logs",
                str(r2_run_dir),
                "--out",
                str(recovery_audit_path),
            ]
        )
        interim_report = {
            "null_R2_tv": recovery,
            "audits": {"null_R2_tv": recovery_audit},
        }
        report_path = root / "results" / "phase_f_null_remainder.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(interim_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        WORK_VOLUME.commit()
        prior_r2_tv = _load_completed_phase_f_r2_tv(root)
        if prior_r2_tv is None:
            interim_report["status"] = "failed"
            interim_report["recovery_error"] = (
                "fresh R2-TV training did not satisfy the trainer, checkpoint, "
                "timing, and queue-budget evidence gate"
            )
            report_path.write_text(
                json.dumps(interim_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            WORK_VOLUME.commit()
            return {"root": str(root), **interim_report}
        results["null_R2_tv"] = {
            **prior_r2_tv,
            "reused_completed_run_artifact": False,
        }
    elif r3_r6_only:
        results["null_R2_tv"] = {
            "status": "skipped",
            "reason": "R3-R6-only smoke; R2-TV recovery was not budgeted or run",
        }
    else:
        results["null_R2_tv"] = {
            **prior_r2_tv,
            "reused_completed_run_artifact": True,
        }
        run_dirs["null_R2_tv"] = str(
            root / "null_R2_tv" / "runs" / "R2_s42"
        )
    smoke_functions = {
        "igpo": phase_f_igpo_recover_smoke,
        "treerl": phase_f_treerl_smoke,
    }
    specs = [
        ("null_R3_spo", "R3", "{}"),
        ("null_R4_tv", "R4", "{}"),
        ("null_R5_spo", "R5", "{}"),
        ("null_R6_tv", "R6", "{}"),
    ]
    for label, exp, variant in specs:
        run_root = root / label
        env = {
            **shared_env,
            "RUNS_DIR": str(run_root / "runs"),
            "RESULTS_DIR": str(run_root / "results"),
            "BANKS_DIR": str(run_root / "banks"),
            "VDRA_VARIANT_JSON": variant,
        }
        framework = EXPERIMENTS[exp][1]
        function = (
            _phase_f_treepo_smoke_function(exp)
            if framework == "treepo"
            else smoke_functions[framework]
        )
        try:
            result = function.spawn(exp, env).get()
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        results[label] = result
        run_dirs[label] = str(run_root / "runs" / f"{exp}_s42")

    audit_dir = root / "results" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_specs = []
    if not r3_r6_only:
        audit_specs.extend([
            ("null_igpo", [
                sys.executable, "/opt/vdra/analysis/budget_audit.py", "--null",
                str(baseline_run_dirs[1]),
                str(baseline_run_dirs[0]),
                "--out", str(audit_dir / "null_igpo.json"),
            ]),
            ("null_R2_tv", [
                sys.executable, "/opt/vdra/analysis/budget_audit.py", "--require-budget-evidence",
                "--logs", run_dirs["null_R2_tv"], "--out", str(audit_dir / "null_R2_tv.json"),
            ]),
        ])
    audit_specs.extend([
        ("null_R4_tv", [
            sys.executable, "/opt/vdra/analysis/budget_audit.py", "--require-budget-evidence",
            "--logs", run_dirs["null_R4_tv"], "--out", str(audit_dir / "null_R4_tv.json"),
        ]),
        ("null_R6_tv", [
            sys.executable, "/opt/vdra/analysis/budget_audit.py", "--require-budget-evidence",
            "--logs", run_dirs["null_R6_tv"], "--out", str(audit_dir / "null_R6_tv.json"),
        ]),
    ])
    audits: dict[str, Any] = {}
    for name, command in audit_specs:
        audits[name] = _run_phase_f_audit(command)
    results["audits"] = audits
    if r3_r6_only:
        results["scope"] = "R3-R6-only method smoke; not full Phase F certification"
        results["r3_r6_status"] = (
            "complete" if _phase_f_r3_r6_smokes_complete(results, audits) else "failed"
        )
        results["phase_f_certification"] = "not_certified_r2_tv_recovery_skipped"
        results["status"] = "partial" if results["r3_r6_status"] == "complete" else "failed"
    else:
        results["status"] = (
            "complete"
            if _phase_f_smokes_complete(
                results,
                ("null_R2_tv", "null_R3_spo", "null_R4_tv", "null_R5_spo", "null_R6_tv"),
            )
            and _phase_f_audits_succeeded(audits)
            and audits.get("null_igpo", {}).get("pass") is True
            and all(audits.get(name, {}).get("gate_G1_budget_ok") is True
                    for name in ("null_R2_tv", "null_R4_tv", "null_R6_tv"))
            else "failed"
        )
    report_path = root / "results" / "phase_f_null_remainder.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    WORK_VOLUME.commit()
    return {"root": str(root), **results}


@app.function(
    image=IMAGES["igpo"], cpu=2, memory=16384, timeout=35 * 60,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_final_methods_smoke(run_tag: str) -> dict[str, Any]:
    """Run one bounded R4/R6 update after compatibility fixes, then audit available paths."""
    if not run_tag or not run_tag.isalnum():
        raise ValueError("run_tag must contain only letters and digits")
    root = WORK / "mock" / f"phase_f_null_{run_tag}"
    shared_env = {
        "STOP_STEP": "1",
        "RESUME_EVERY": "1",
        "KEEP_EVERY": "1",
        "MAX_RESTARTS": "0",
        "EVAL_EVERY_N": "0",
        "EVAL_AT_END": "0",
        "EVAL_STEP0": "0",
        "COMPUTE_MATCHED": "0",
        "SEEDS": "42",
        "WORK": "/work",
        "DATA_DIR": "/work/data_vdra",
        "MATH_DICT": "/work/source_data/math",
        "VDRA_B200": "1",
        "VDRA_CONFIG_JSON": "{}",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TREEPO_WIDTH": "2",
        "TREEPO_GEN_BATCH_SIZE": "8",
        "TREEPO_TRAIN_BATCH_SIZE": "8",
        "TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT": PHASE_F_TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT,
        "TREERL_ROLLOUT_BATCH_SIZE": "4",
        "TREERL_MAX_TIME_USE": "60",
        "TREERL_VLLM_MEM": PHASE_F_TREERL_VLLM_MEM,
        "TREERL_VLLM_BATCHED_TOKENS": PHASE_F_TREERL_VLLM_BATCHED_TOKENS,
        "TREERL_M": "2",
        "TREERL_N": "1",
        "TREERL_L": "1",
        "TREERL_T": "1",
    }
    specs = [
        ("null_R4_tv", "R4", "{}", phase_f_treepo_one_update),
        ("null_R6_tv", "R6", "{}", phase_f_treerl_one_update),
    ]
    results: dict[str, Any] = {}
    run_dirs: dict[str, str] = {}
    for label, exp, variant, function in specs:
        run_root = root / label
        env = {
            **shared_env,
            "RUNS_DIR": str(run_root / "runs"),
            "RESULTS_DIR": str(run_root / "results"),
            "BANKS_DIR": str(run_root / "banks"),
            "VDRA_VARIANT_JSON": variant,
        }
        try:
            result = function.spawn(exp, env).get()
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        results[label] = result
        run_dirs[label] = str(run_root / "runs" / f"{exp}_s42")

    audit_dir = root / "results" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_specs = [
        ("null_igpo", [
            sys.executable, "/opt/vdra/analysis/budget_audit.py", "--null",
            str(root / "null_R2_constant" / "runs" / "R2_s42"),
            str(root / "null_R1" / "runs" / "R1_s42"),
            "--out", str(audit_dir / "null_igpo.json"),
        ]),
        ("null_R2_tv", [
            sys.executable, "/opt/vdra/analysis/budget_audit.py", "--require-budget-evidence",
            "--logs", str(root / "null_R2_tv" / "runs" / "R2_s42"),
            "--out", str(audit_dir / "null_R2_tv.json"),
        ]),
        ("null_R4_tv", [
            sys.executable, "/opt/vdra/analysis/budget_audit.py", "--require-budget-evidence",
            "--logs", run_dirs["null_R4_tv"], "--out", str(audit_dir / "null_R4_tv.json"),
        ]),
        ("null_R6_tv", [
            sys.executable, "/opt/vdra/analysis/budget_audit.py", "--require-budget-evidence",
            "--logs", run_dirs["null_R6_tv"], "--out", str(audit_dir / "null_R6_tv.json"),
        ]),
    ]
    audits: dict[str, Any] = {}
    for name, command in audit_specs:
        try:
            audit_process = subprocess.run(command, cwd="/opt/vdra", check=False, stdout=subprocess.DEVNULL)
            audit_path = Path(command[-1])
            if audit_path.is_file():
                audits[name] = json.loads(audit_path.read_text(encoding="utf-8"))
                audits[name]["audit_exit_code"] = audit_process.returncode
            else:
                audits[name] = {"status": "failed", "error": f"audit exited {audit_process.returncode} without a report"}
        except Exception as exc:
            audits[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    results["audits"] = audits
    results["status"] = (
        "complete"
        if all(results[label].get("status") == "complete" for label, *_ in specs)
        and _phase_f_audits_succeeded(audits)
        and audits.get("null_igpo", {}).get("pass") is True
        and all(audits.get(name, {}).get("gate_G1_budget_ok") is True
                for name in ("null_R2_tv", "null_R4_tv", "null_R6_tv"))
        else "failed"
    )
    report_path = root / "results" / "phase_f_final_methods.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    WORK_VOLUME.commit()
    return {"root": str(root), **results}


@app.function(
    image=IMAGES["igpo"], cpu=2, memory=16384, timeout=15 * 60,
    retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME},
)
def phase_f_treerl_retry_smoke(run_tag: str) -> dict[str, Any]:
    """Retry only the corrected R6 one-update smoke and audit its queue evidence."""
    if not run_tag or not run_tag.isalnum():
        raise ValueError("run_tag must contain only letters and numbers")
    root = WORK / "mock" / f"phase_f_treerl_retry_{run_tag}"
    run_root = root / "runs"
    env = {
        "STOP_STEP": "1",
        "RESUME_EVERY": "1",
        "KEEP_EVERY": "1",
        "MAX_RESTARTS": "0",
        "EVAL_EVERY_N": "0",
        "EVAL_AT_END": "0",
        "EVAL_STEP0": "0",
        "COMPUTE_MATCHED": "0",
        "SEEDS": "42",
        "WORK": "/work",
        "DATA_DIR": "/work/data_vdra",
        "MATH_DICT": "/work/source_data/math",
        "VDRA_B200": "1",
        "VDRA_CONFIG_JSON": "{}",
        "VDRA_VARIANT_JSON": "{}",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "RUNS_DIR": str(run_root),
        "RESULTS_DIR": str(root / "results"),
        "BANKS_DIR": str(root / "banks"),
        "TREERL_ROLLOUT_BATCH_SIZE": "4",
        "TREERL_MAX_TIME_USE": "60",
        "TREERL_VLLM_MEM": PHASE_F_TREERL_VLLM_MEM,
        "TREERL_VLLM_BATCHED_TOKENS": PHASE_F_TREERL_VLLM_BATCHED_TOKENS,
        "TREERL_M": "2",
        "TREERL_N": "1",
        "TREERL_L": "1",
        "TREERL_T": "1",
    }
    result = phase_f_treerl_one_update.spawn("R6", env).get()
    run_dir = run_root / "R6_s42"
    audit_dir = root / "results" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "r6_budget.json"
    command = [
        sys.executable, "/opt/vdra/analysis/budget_audit.py", "--require-budget-evidence",
        "--logs", str(run_dir), "--out", str(audit_path),
    ]
    process = subprocess.run(command, cwd="/opt/vdra", check=False, stdout=subprocess.DEVNULL)
    audit = (
        {**json.loads(audit_path.read_text(encoding="utf-8")), "audit_exit_code": process.returncode}
        if audit_path.is_file()
        else {"status": "failed", "error": f"budget audit exited {process.returncode} without a report"}
    )
    report = {
        "root": str(root),
        "R6": result,
        "R6_budget_audit": audit,
        "status": "complete" if result.get("status") == "complete" and _phase_f_audits_succeeded([audit]) and audit.get("gate_G1_budget_ok") is True else "failed",
    }
    report_path = root / "results" / "phase_f_treerl_retry.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    WORK_VOLUME.commit()
    return report


def _continue_pair(pair: str, seeds: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    if pair not in PAIRS:
        raise ValueError(f"unknown pair: {pair}")
    baseline, hybrid, _ = PAIRS[pair]
    call_id = os.environ.get("MODAL_FUNCTION_CALL_ID") or uuid.uuid4().hex
    started = time.time()
    session_dir = WORK / "results/modal_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / f"{call_id}.json"
    benchmark = run_start_benchmark()
    session_path.write_text(
        json.dumps(
            {
                "call_id": call_id,
                "pair": pair,
                "baseline": baseline,
                "hybrid": hybrid,
                "seeds": seeds,
                "gpu": "B200",
                "image_id": os.environ.get("MODAL_IMAGE_ID"),
                "started_at": started,
                "status": "running",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for seed in seeds:
        append_jsonl(
            WORK / "runs" / f"{baseline}_s{seed}" / "run_meta.jsonl",
            dict(
                benchmark,
                run=baseline,
                seed=int(seed),
                container_call_id=call_id,
                container_id=os.environ.get("MODAL_CONTAINER_ID") or os.environ.get("MODAL_TASK_ID"),
                allocated_benchmark_cost_usd=benchmark["benchmark_seconds"] * GPU_RATE_USD_PER_SECOND / len(seeds),
            ),
        )
    env_values = {key: str(value) for key, value in (env or {}).items()}
    process_env = os.environ.copy()
    process_env.update(env_values)
    process_env.update(
        {
            "VDRA_HOME": "/opt/vdra",
            "WORK": "/work",
            "RUNS_DIR": "/work/runs",
            "RESULTS_DIR": "/work/results",
            "BANKS_DIR": "/work/banks",
            "DATA_DIR": "/work/data_vdra",
            "MATH_DICT": "/work/source_data/math",
            "MODEL_PATH": str(_runtime_manifest().get("model_path", MODEL_ID)),
            "IGPO_DIR": "/opt/work/igpo",
            "TREEPO_DIR": "/opt/work/TreePO",
            "TREERL_DIR": "/opt/work/TreeRL",
            "VDRA_B200": "1",
            "GPUS": "0",
            "MAX_RESTARTS": "0",
        }
    )
    for seed in seeds:
        pair_env = dict(process_env, PAIR=pair, SEED=seed, GPUS="0")
        result = subprocess.run(["bash", "/opt/vdra/scripts/30_compute_matched.sh"], env=pair_env, check=False)
        if result.returncode:
            status, error = "failed", f"{pair} seed {seed} continuation exited {result.returncode}"
            break
    else:
        status, error = "complete", None
    elapsed = time.time() - started
    session_path.write_text(
        json.dumps(
            {
                "call_id": call_id,
                "pair": pair,
                "baseline": baseline,
                "hybrid": hybrid,
                "seeds": seeds,
                "gpu": "B200",
                "image_id": os.environ.get("MODAL_IMAGE_ID"),
                "started_at": started,
                "ended_at": time.time(),
                "wall_seconds": elapsed,
                "approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND,
                "allocated_approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND / len(seeds),
                "status": status,
                "error": error,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for seed in seeds:
        append_jsonl(
            WORK / "runs" / f"{baseline}_s{seed}" / "run_meta.jsonl",
            {
                "event": "modal_continuation_end",
                "time": time.time(),
                "container_call_id": call_id,
                "wall_seconds": elapsed,
                "approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND,
                "allocated_approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND / len(seeds),
                "status": status,
                "error": error,
            },
        )
    WORK_VOLUME.commit()
    return {"pair": pair, "seeds": seeds, "status": status, "error": error, "call_id": call_id}


@app.function(
    image=IMAGES["igpo"],
    gpu="B200",
    timeout=TRAIN_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def continue_spo(seeds: list[str], env: dict[str, str]) -> dict[str, Any]:
    return _continue_pair("spo", [str(s) for s in seeds], env)


@app.function(
    image=IMAGES["treepo"],
    gpu="B200",
    timeout=TRAIN_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def continue_treepo(seeds: list[str], env: dict[str, str]) -> dict[str, Any]:
    return _continue_pair("treepo", [str(s) for s in seeds], env)


@app.function(
    image=IMAGES["treerl"],
    gpu="B200",
    timeout=TRAIN_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": WORK_VOLUME},
)
def continue_treerl(seeds: list[str], env: dict[str, str]) -> dict[str, Any]:
    return _continue_pair("treerl", [str(s) for s in seeds], env)


def _job_function(name: str):
    return {"spo": continue_spo, "treepo": continue_treepo, "treerl": continue_treerl}[name]


def _campaign_id(exps: list[str], seeds: list[str], env: dict[str, str]) -> str:
    raw = json.dumps({"exps": exps, "seeds": seeds, "env": env}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _get_or_spawn(key: str, function, args: tuple[Any, ...]) -> tuple[modal.FunctionCall[Any], bool]:
    state = CONTROL.get(key)
    if state and state.get("call_id"):
        call = modal.FunctionCall.from_id(state["call_id"])
        try:
            result = call.get(timeout=0)
        except FunctionTimeoutError:
            call = function.spawn(*args)
            CONTROL.put(key, {"call_id": call.object_id, "status": "running", "retries": state.get("retries", 0) + 1})
            return call, True
        except modal.exception.TimeoutError:
            return call, False
        except Exception as exc:
            if state.get("status") == "running":
                # Non-timeout failures are surfaced for diagnosis; no blind deterministic retry.
                CONTROL.put(key, {**state, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return call, False
        else:
            if isinstance(result, dict) and result.get("status") == "complete":
                CONTROL.put(key, {**state, "status": "complete", "result": result})
            else:
                CONTROL.put(key, {**state, "status": "failed", "result": result})
            return call, False
    call = function.spawn(*args)
    CONTROL.put(key, {"call_id": call.object_id, "status": "running", "retries": 0})
    return call, True


def _await_jobs(
    prefix: str,
    campaign: str,
    jobs: dict[str, tuple[Any, tuple[Any, ...]]],
) -> dict[str, dict[str, Any]]:
    active: dict[str, modal.FunctionCall[Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for name, (function, args) in jobs.items():
        call, _ = _get_or_spawn(f"{prefix}:{campaign}:{name}", function, args)
        active[name] = call
    while active:
        for name, call in list(active.items()):
            try:
                result = call.get(timeout=10)
            except FunctionTimeoutError:
                function, args = jobs[name]
                next_call = function.spawn(*args)
                key = f"{prefix}:{campaign}:{name}"
                old = CONTROL.get(key) or {}
                CONTROL.put(key, {"call_id": next_call.object_id, "status": "running", "retries": old.get("retries", 0) + 1})
                active[name] = next_call
                continue
            except modal.exception.TimeoutError:
                continue
            except Exception as exc:
                results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                active.pop(name, None)
                continue
            results[name] = result
            key = f"{prefix}:{campaign}:{name}"
            old = CONTROL.get(key) or {}
            CONTROL.put(key, {**old, "status": result.get("status", "unknown"), "result": result})
            active.pop(name, None)
        if active:
            print(f"[{prefix}] still running: {', '.join(sorted(active))}", flush=True)
    return results


@app.function(timeout=APP_TIMEOUT_SECONDS, retries=0, scaledown_window=2)
def orchestrate(exps: list[str], seeds: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run six primary queues in parallel, then compute-matched continuations."""
    env = dict(env or {})
    unknown = sorted(set(exps) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"unknown experiments: {unknown}")
    if len(exps) != len(set(exps)):
        raise ValueError("exps must not contain duplicates")
    if not seeds:
        raise ValueError("seeds cannot be empty")
    seeds = [str(s) for s in seeds]
    env.update(
        {
            "STOP_STEP": "220",
            "SEEDS": " ".join(seeds),
            "EVAL_EVERY_N": "25",
            "EVAL_AT_END": "1",
            "EVAL_STEP0": "1",
            "MAX_RESTARTS": "0",
            "COMPUTE_MATCHED": "0",
        }
    )
    if not _runtime_manifest().get("ready"):
        setup_data.remote()
    campaign = _campaign_id(exps, seeds, env)

    train_jobs = {
        exp: (
            {"igpo": train_igpo, "treepo": train_treepo, "treerl": train_treerl}[EXPERIMENTS[exp][1]],
            (exp, seeds, env),
        )
        for exp in exps
    }
    training_results = _await_jobs("train", campaign, train_jobs)
    failed = {name: result for name, result in training_results.items() if result.get("status") != "complete"}
    continuation_results: dict[str, dict[str, Any]] = {}
    if not failed:
        selected_pairs = [pair for pair, (baseline, hybrid, _) in PAIRS.items() if baseline in exps and hybrid in exps]
        continuation_jobs = {pair: (_job_function(pair), (seeds, env)) for pair in selected_pairs}
        continuation_results = _await_jobs("continuation", campaign, continuation_jobs)
    return {
        "campaign_id": campaign,
        "training": training_results,
        "continuations": continuation_results,
        "status": "complete" if not failed and all(r.get("status") == "complete" for r in continuation_results.values()) else "failed",
    }


def evaluate(run_id: str, limit: int | None = None, steps: str = "") -> modal.FunctionCall[Any]:
    """Submit a missing-checkpoint evaluation in the run's framework image."""
    if not run_id.startswith(tuple(f"R{i}_" for i in range(1, 7))):
        raise ValueError(f"invalid run id: {run_id}")
    exp = run_id.split("_", 1)[0]
    function = {"igpo": evaluate_igpo, "treepo": evaluate_treepo, "treerl": evaluate_treerl}[EXPERIMENTS[exp][1]]
    return function.spawn(run_id, limit, steps)


def _run_evaluation(run_id: str, limit: int | None, steps: str) -> dict[str, Any]:
    exp = run_id.split("_", 1)[0]
    framework = EXPERIMENTS[exp][1]
    run_dir = WORK / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest = _runtime_manifest()
    model_path = str(manifest.get("model_path", MODEL_ID))
    call_id = os.environ.get("MODAL_FUNCTION_CALL_ID") or uuid.uuid4().hex
    function_started = time.time()
    session_dir = WORK / "results/modal_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / f"{call_id}.json"
    session_path.write_text(
        json.dumps(
            {
                "call_id": call_id,
                "run_id": run_id,
                "seeds": [run_id.split("_s", 1)[1]],
                "gpu": "B200",
                "image_id": os.environ.get("MODAL_IMAGE_ID"),
                "started_at": function_started,
                "status": "running",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = run_start_benchmark()
    append_jsonl(
        run_dir / "run_meta.jsonl",
        dict(
            benchmark,
            event="eval_container_start",
            run_id=run_id,
            container_call_id=call_id,
            container_id=os.environ.get("MODAL_CONTAINER_ID") or os.environ.get("MODAL_TASK_ID"),
            allocated_benchmark_cost_usd=benchmark["benchmark_seconds"] * GPU_RATE_USD_PER_SECOND,
        ),
    )
    env = os.environ.copy()
    env.update(
        {
            "VDRA_HOME": "/opt/vdra",
            "WORK": "/work",
            "RUNS_DIR": "/work/runs",
            "RESULTS_DIR": "/work/results",
            "BANKS_DIR": "/work/banks",
            "DATA_DIR": "/work/data_vdra",
            "MATH_DICT": "/work/source_data/math",
            "MODEL_PATH": model_path,
            "IGPO_DIR": "/opt/work/igpo",
            "TREEPO_DIR": "/opt/work/TreePO",
            "TREERL_DIR": "/opt/work/TreeRL",
            "VDRA_B200": "1",
            "EVAL_STEP0": "0",
            "VDRA_SINGLE_RANK_COLLECTIVES": "0",
            "WANDB_MODE": "offline",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    fw_name = {"igpo": "treetune", "treepo": "verl", "treerl": "openrlhf"}[framework]
    gpus = 1
    gpu_hours_path = WORK / "results/gpu_hours" / f"{run_id}.json"
    subprocess.run(
        [sys.executable, "/opt/vdra/analysis/gpu_hours.py", "--run_dir", str(run_dir), "--framework", fw_name,
         "--gpus", str(gpus), "--out", str(gpu_hours_path)],
        env=env,
        check=False,
    )
    eval_args = [
        "bash",
        "/opt/vdra/scripts/50_eval.sh",
    ]
    env["RUN_IDS"] = run_id
    env["EVAL_GPU_GROUPS"] = "0"
    env["EVAL_STEP0"] = "0"
    if limit is not None:
        # Preserve the one shared evaluator path by passing the cap through its environment.
        env["VDRA_EVAL_LIMIT"] = str(limit)
    if steps:
        env["VDRA_EVAL_STEPS"] = steps
    eval_started = time.time()
    result = subprocess.run(eval_args, env=env, check=False)
    elapsed = time.time() - eval_started
    append_jsonl(
        run_dir / "run_meta.jsonl",
        {
            "event": "eval_container_end",
            "time": time.time(),
            "container_call_id": call_id,
            "wall_seconds": elapsed,
            "approx_cost_usd": elapsed * GPU_RATE_USD_PER_SECOND,
            "returncode": result.returncode,
        },
    )
    session_path.write_text(
        json.dumps(
            {
                "call_id": call_id,
                "run_id": run_id,
                "seeds": [run_id.split("_s", 1)[1]],
                "gpu": "B200",
                "image_id": os.environ.get("MODAL_IMAGE_ID"),
                "started_at": function_started,
                "ended_at": time.time(),
                "wall_seconds": time.time() - function_started,
                "approx_cost_usd": (time.time() - function_started) * GPU_RATE_USD_PER_SECOND,
                "status": "complete" if result.returncode == 0 else "failed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    WORK_VOLUME.commit()
    return {"run_id": run_id, "status": "complete" if result.returncode == 0 else "failed", "returncode": result.returncode}


@app.function(image=IMAGES["igpo"], gpu="B200", timeout=2 * 60 * 60, retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME})
def evaluate_igpo(run_id: str, limit: int | None = None, steps: str = "") -> dict[str, Any]:
    return _run_evaluation(run_id, limit, steps)


@app.function(image=IMAGES["treepo"], gpu="B200", timeout=2 * 60 * 60, retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME})
def evaluate_treepo(run_id: str, limit: int | None = None, steps: str = "") -> dict[str, Any]:
    return _run_evaluation(run_id, limit, steps)


@app.function(image=IMAGES["treerl"], gpu="B200", timeout=2 * 60 * 60, retries=0, scaledown_window=2, volumes={"/work": WORK_VOLUME})
def evaluate_treerl(run_id: str, limit: int | None = None, steps: str = "") -> dict[str, Any]:
    return _run_evaluation(run_id, limit, steps)


@app.local_entrypoint()
def main(
    action: str = "setup",
    exps: str = "R1 R2 R3 R4 R5 R6",
    seeds: str = "42 123 2026",
    env_json: str = "{}",
    run_id: str = "",
    baseline_run_id: str = "1629e2315f",
    r3_r6_only: bool = False,
    limit: int = 0,
    steps: str = "",
) -> None:
    if action == "setup":
        print(json.dumps(setup_data.remote(), indent=2, sort_keys=True))
    elif action == "register_migrated":
        print(json.dumps(register_migrated_data.remote(), indent=2, sort_keys=True))
    elif action == "launch":
        setup_data.remote()
        env = json.loads(env_json)
        call = orchestrate.spawn(exps.split(), seeds.split(), env)
        print(json.dumps({"orchestrator_call_id": call.object_id, "dashboard_url": call.get_dashboard_url()}))
    elif action == "phase_f_null_smoke":
        result = phase_f_null_smoke.spawn(uuid.uuid4().hex[:10]).get()
        print(json.dumps({"phase_f_null_smoke": result}, indent=2, sort_keys=True))
        if result.get("status") != "complete":
            raise SystemExit("Phase F null smoke did not pass")
    elif action == "phase_f_remaining_smoke":
        if not run_id:
            raise SystemExit("action=phase_f_remaining_smoke requires run_id")
        result = phase_f_remaining_smoke.spawn(run_id, baseline_run_id, r3_r6_only).get()
        print(json.dumps({"phase_f_remaining_smoke": result}, indent=2, sort_keys=True))
        completed = (
            result.get("r3_r6_status") == "complete"
            if r3_r6_only
            else result.get("status") == "complete"
        )
        if not completed:
            raise SystemExit("Phase F remaining smoke did not pass")
    elif action == "phase_f_final_methods_smoke":
        if not run_id:
            raise SystemExit("action=phase_f_final_methods_smoke requires run_id")
        result = phase_f_final_methods_smoke.spawn(run_id).get()
        print(json.dumps({"phase_f_final_methods_smoke": result}, indent=2, sort_keys=True))
        if result.get("status") != "complete":
            raise SystemExit("Phase F final methods smoke did not pass")
    elif action == "phase_f_treerl_retry_smoke":
        if not run_id:
            raise SystemExit("action=phase_f_treerl_retry_smoke requires run_id")
        result = phase_f_treerl_retry_smoke.spawn(run_id).get()
        print(json.dumps({"phase_f_treerl_retry_smoke": result}, indent=2, sort_keys=True))
        if result.get("status") != "complete":
            raise SystemExit("Phase F TreeRL retry smoke did not pass")
    elif action == "evaluate":
        if not run_id:
            raise SystemExit("action=evaluate requires run_id")
        call = evaluate(run_id, limit or None, steps)
        print(json.dumps({"evaluation_call_id": call.object_id, "dashboard_url": call.get_dashboard_url()}))
    else:
        raise ValueError(
            "action must be setup, register_migrated, launch, phase_f_null_smoke, "
            "phase_f_remaining_smoke, phase_f_final_methods_smoke, "
            "phase_f_treerl_retry_smoke, or evaluate"
        )
