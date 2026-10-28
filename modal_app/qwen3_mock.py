"""One-update Modal smoke for Qwen3-1.7B thinking-mode SPO and VDRA."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import modal

from modal_app.images import framework_images


APP_NAME = "vdra-qwen3-thinking-mock"
VOLUME_NAME = "vdra-qwen3-thinking-mock"
WORK = Path("/work")
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
RUNS = (
    ("spo", "R1", "qwen3_spo.sh", "Qwen3_SPO_s42"),
    ("vdra", "R2", "qwen3_vdra.sh", "Qwen3_VDRA_s42"),
)
GPU_RATE_USD_PER_SECOND = 0.001736
GPU_TYPES = ["A100-80GB", "B200"]
PAIR_TIMEOUT_SECONDS = 27 * 60
PER_RUN_TIMEOUT_SECONDS = 12 * 60 + 30

IMAGES = framework_images()
APP = modal.App(APP_NAME)
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ready_manifest() -> dict[str, Any] | None:
    path = WORK / "setup/qwen3_thinking_manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_path = Path(manifest.get("model_path", ""))
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or not (model_path / "config.json").is_file()
        or not (WORK / "source_data/math/dataset_dict.json").is_file()
        or not (WORK / "data_vdra/math-nodiag/dataset_dict.json").is_file()
        or not (WORK / "data_vdra/split_manifest.json").is_file()
    ):
        return None
    return manifest


def _prepare_math() -> None:
    destination = WORK / "source_data/math"
    if (destination / "dataset_dict.json").is_file():
        return
    bootstrap = Path("/tmp/qwen3-math-bootstrap")
    if bootstrap.exists():
        shutil.rmtree(bootstrap)
    shutil.copytree("/opt/work/igpo", bootstrap)
    script = bootstrap / "scripts/download_and_prepare_dataset.sh"
    source = script.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    script.write_text(source, encoding="utf-8", newline="\n")
    subprocess.run(["bash", str(script)], cwd=bootstrap, check=True)
    dataset = bootstrap / "data/math"
    if not (dataset / "dataset_dict.json").is_file():
        candidates = [p for p in (bootstrap / "data").glob("**/math") if (p / "dataset_dict.json").is_file()]
        if len(candidates) != 1:
            raise FileNotFoundError(f"MATH dataset archive did not create one DatasetDict: {candidates}")
        dataset = candidates[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset, destination)


def _preflight_configs(model_path: str) -> dict[str, Any]:
    """Render the same Qwen configs and fast runtime overlay used by both runs."""
    import _jsonnet
    from transformers import AutoTokenizer

    config_root = Path("/opt/work/igpo/configs")
    env = os.environ.copy()
    env.update(
        {
            "APP_DIRECTORY": "/tmp/qwen3-config-preflight",
            "INGPO_ROOT": "/opt/work/igpo",
            "INGPO_FAST_OVERLAY": "1",
            "INGPO_GPUS": "0",
            "INGPO_SAVE_STEPS": "1",
            "INGPO_CHECKPOINT_KEEP_STEPS": "1",
            "INGPO_NUM_EPISODES_PER_ITERATION": "16",
            "INGPO_DATASET_NUM_SAMPLES_PER_ITERATION": "4",
            "INGPO_TARGET_TRAIN_BATCH_SIZE": "8",
            "INGPO_PER_DEVICE_TRAIN_BATCH_SIZE": "1",
            "INGPO_VLLM_GPU_MEMORY_UTILIZATION": "0.25",
            "INGPO_VLLM_MAX_NUM_SEQS": "16",
            "INGPO_MAX_CONCURRENT_PROGRAMS": "8",
            "INGPO_MAX_CONCURRENT_GENERATIONS": "16",
            "INGPO_VALUE_MAX_CONCURRENT_PROGRAMS": "16",
            "INGPO_VALUE_MAX_CONCURRENT_GENERATIONS": "32",
        }
    )
    bootstrap = subprocess.run(
        ["bash", "-lc", 'source /opt/work/igpo/scripts/_common.sh; ensure_runtime_config; ensure_gpu_config 0'],
        cwd="/opt/work/igpo",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    generated = [Path(line.strip()) for line in bootstrap.stdout.splitlines() if line.strip().endswith(".jsonnet")]
    runtime_path = next((p for p in generated if "runtime_overrides" in str(p)), None)
    gpu_path = next((p for p in generated if "/gpus/" in str(p)), None)
    if runtime_path is None or gpu_path is None:
        raise RuntimeError(f"could not locate generated Jsonnet overlays: {bootstrap.stdout}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve 1 + 1."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    expected_assistant_prefix = "<|im_start|>assistant\n<think>\n"
    assistant_start = chat_prompt.rfind("<|im_start|>assistant")
    if not chat_prompt.startswith("<|im_start|>user\n") or assistant_start < 0:
        raise RuntimeError(f"Qwen tokenizer did not render ChatML roles: suffix={chat_prompt[-120:]!r}")

    common = [
        "qwen3/branch_factor_222.jsonnet",
        "episode_generators/no_prefix_caching.jsonnet",
        "vdra/data_nodiag.jsonnet",
        "vdra/model_path.jsonnet",
        "vdra/stop.jsonnet",
        "vdra/no_internal_eval.jsonnet",
        "qwen3/smoke_short.jsonnet",
        "episode_generators/noSamplRplc.jsonnet",
    ]
    ext_vars = {
        "APP_SEED": "42",
        "APP_DISABLE_FLASH_ATTENTION": "0",
        "APP_OPENAI_VLLM_API_BASE": "none",
        "APP_VDRA_TRAIN_DATASET_DICT": "/work/data_vdra/math-nodiag",
        "APP_VDRA_MODEL_PATH": model_path,
        "APP_VDRA_STOP_STEP": "1",
        "APP_VDRA_VARIANT_JSON": "{}",
    }
    rendered: dict[str, Any] = {}
    for label, base_name, is_vdra in (
        ("spo", "polIter_qwen3_1_7b_spo_logged_MATH.jsonnet", False),
        ("vdra", "polIter_qwen3_1_7b_vdra_MATH.jsonnet", True),
    ):
        names = [base_name, *common]
        if is_vdra:
            names.extend(["vdra/variant.jsonnet", "qwen3/smoke_vdra_fast.jsonnet"])
        names.extend([str(gpu_path), str(runtime_path)])
        imports = "+".join(
            f"(import {json.dumps(str(p if Path(p).is_absolute() else config_root / p))})" for p in names
        )
        config = json.loads(_jsonnet.evaluate_snippet(f"qwen3-{label}-mock", imports, ext_vars=ext_vars))
        eg = config["episode_generator"]
        strategy = eg["inference_strategy"]
        prompt = eg["question_template"]
        formatted_prompt = prompt.format(query="Solve 1 + 1.")
        if not formatted_prompt.endswith(expected_assistant_prefix):
            raise RuntimeError(f"{label}: effective prompt is not Qwen3 thinking mode")
        if eg["initial_model_name_or_path"] != model_path:
            raise RuntimeError(f"{label}: model path was not overridden to the pinned local Qwen snapshot")
        if strategy.get("type") != ("vdra" if is_vdra else "spo_logged"):
            raise RuntimeError(f"{label}: wrong inference strategy {strategy.get('type')!r}")
        if strategy["node_expander"]["program_kwargs"].get("max_tokens") != 256:
            raise RuntimeError(f"{label}: smoke generation cap was not applied")
        rendered[label] = {
            "strategy": strategy.get("type"),
            "model_path": eg["initial_model_name_or_path"],
            "thinking_prompt_suffix": expected_assistant_prefix,
            "max_tokens": strategy["node_expander"]["program_kwargs"]["max_tokens"],
            "num_episodes_per_iteration": config["num_episodes_per_iteration"],
            "dataset_num_samples_per_iteration": eg["dataset_num_samples_per_iteration"],
            "branch_factors": strategy["branch_factor_strategy"]["branch_factors"],
            "vdra_config": strategy.get("vdra") if is_vdra else None,
        }
    return {
        "tokenizer_chat_template_suffix": chat_prompt[-120:],
        "tokenizer_template_has_think_prefix": "<think>" in chat_prompt[assistant_start:],
        "effective_prompt_thinking_suffix": expected_assistant_prefix,
        "runs": rendered,
    }


@APP.function(
    image=IMAGES["igpo"],
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    retries=0,
    scaledown_window=2,
    volumes={"/work": VOLUME},
)
def setup_data() -> dict[str, Any]:
    """Download the pinned Qwen snapshot and prepare the unchanged MATH split."""
    existing = _ready_manifest()
    if existing is not None:
        existing["config_preflight"] = _preflight_configs(existing["model_path"])
        return {"status": "ready_cached", **existing}

    _prepare_math()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir="/work/hf_cache")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    from datasets import load_from_disk

    source = load_from_disk("/work/source_data/math")
    if "train" not in source or len(source["train"]) == 0:
        raise RuntimeError("downloaded MATH dataset does not contain a nonempty train split")
    subprocess.run(
        [
            sys.executable,
            "/opt/vdra/analysis/prepare_data.py",
            "--math_dict",
            "/work/source_data/math",
            "--out",
            "/work/data_vdra",
            "--n_diag",
            "256",
            "--seed",
            "2026",
        ],
        cwd="/opt/vdra",
        check=True,
        env={**os.environ, "PYTHONPATH": "/opt/vdra:/opt/work/igpo"},
    )
    setup_config = _preflight_configs(model_path)
    weights = sorted((Path(model_path)).glob("*.safetensors"))
    if not weights:
        raise RuntimeError(f"no safetensors weights found in pinned model snapshot: {model_path}")
    manifest = {
        "status": "ready",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": model_path,
        "model_weight_shards": len(weights),
        "model_weight_bytes": sum(p.stat().st_size for p in weights),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_chat_template_sha256": hashlib.sha256(
            (tokenizer.chat_template or "").encode("utf-8")
        ).hexdigest(),
        "source_training_rows": len(source["train"]),
        "train_dataset_path": "/work/data_vdra/math-nodiag",
        "data_split_manifest_sha256": _sha256(Path("/work/data_vdra/split_manifest.json")),
        "config_preflight": setup_config,
    }
    target = WORK / "setup/qwen3_thinking_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    VOLUME.commit()
    return manifest


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)
    except ProcessLookupError:
        pass


def _run_streamed(command: list[str], env: dict[str, str], log_path: Path, timeout_s: int) -> dict[str, Any]:
    """Run a training arm with live Modal logs and a durable text copy."""
    proc = subprocess.Popen(
        command,
        cwd="/opt/vdra",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    eof = False
    with log_path.open("w", encoding="utf-8") as log_file:
        while not eof:
            if not timed_out and time.monotonic() - started > timeout_s:
                timed_out = True
                print(f"[qwen3-mock] timeout; stopping process group for {command[-1]}", flush=True)
                _terminate_process_group(proc)
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None and lines.empty():
                    eof = True
                continue
            if line is None:
                eof = True
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        returncode = proc.wait()
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": round(time.monotonic() - started, 2),
        "log_path": str(log_path),
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _collect_run_artifacts(name: str, run_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    status_run = subprocess.run(
        [sys.executable, "/opt/vdra/analysis/run_status.py", str(run_dir), "--framework", "treetune", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = json.loads(status_run.stdout)
    timing_paths = sorted(run_dir.rglob("training_timing.jsonl"))
    timing_rows = [row for path in timing_paths for row in _jsonl(path)]
    iteration_rows = [row for row in timing_rows if isinstance(row.get("iteration"), int)]
    max_iteration = max((row["iteration"] for row in iteration_rows), default=None)
    checkpoints = sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt--iter_*"))
    tree_logs = sorted(run_dir.rglob("vdra_trees.jsonl"))
    tree_rows = [row for path in tree_logs for row in _jsonl(path)]
    meta_rows = _jsonl(run_dir / "run_meta.jsonl")
    arm = {
        "run_name": name,
        "status": status.get("status"),
        "resumable_update": status.get("resumable_update"),
        "stop": status.get("stop"),
        "checkpoint_dirs": checkpoints,
        "has_resumable_checkpoint": bool(checkpoints),
        "training_timing_rows": len(timing_rows),
        "max_iteration_index": max_iteration,
        "tree_log_files": [str(p.relative_to(run_dir)) for p in tree_logs],
        "tree_log_rows": len(tree_rows),
        "logged_strategy_names": sorted({str(row.get("method")) for row in tree_rows}),
        "run_meta_events": [row.get("event") for row in meta_rows],
        "timing_rows": timing_rows,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run_meta.jsonl").is_file():
        shutil.copy2(run_dir / "run_meta.jsonl", artifact_dir / "run_meta.jsonl")
    if timing_paths:
        shutil.copy2(timing_paths[-1], artifact_dir / "training_timing.jsonl")
    tree_target = artifact_dir / "tree_logs"
    for index, path in enumerate(tree_logs):
        tree_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, tree_target / f"{index:02d}_{path.name}")
    return arm


@APP.function(
    image=IMAGES["igpo"],
    gpu=GPU_TYPES,
    timeout=PAIR_TIMEOUT_SECONDS,
    retries=0,
    scaledown_window=2,
    volumes={"/work": VOLUME},
)
def run_pair(tag: str) -> dict[str, Any]:
    """Run one SPO and one VDRA update on an A100-80GB or B200."""
    manifest = _ready_manifest()
    if manifest is None:
        raise RuntimeError("Qwen3 mock data/model are not prepared; run setup_data first")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", tag):
        raise ValueError("tag must contain only letters, digits, hyphen, or underscore")

    artifact_root = WORK / "mock/qwen3_thinking" / tag
    artifact_root.mkdir(parents=True, exist_ok=True)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Modal allocated a GPU, but torch cannot see it")
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    arch = f"{compute_capability[0]}.{compute_capability[1]}"
    model_path = str(manifest["model_path"])
    if not Path(model_path, "config.json").is_file():
        raise FileNotFoundError(f"pinned Qwen model snapshot missing: {model_path}")

    results: dict[str, Any] = {}
    pair_started_at = time.time()
    extension_cache = Path("/tmp/qwen3_mock_shared/torch_extensions")
    for label, experiment, script, run_name in RUNS:
        scratch_root = Path("/tmp") / f"qwen3_mock_{tag}_{label}"
        run_dir = scratch_root / "runs" / run_name
        env = os.environ.copy()
        env.update(
            {
                "VDRA_HOME": "/opt/vdra",
                "WORK": "/work",
                "RUNS_DIR": str(scratch_root / "runs"),
                "RESULTS_DIR": str(scratch_root / "results"),
                "BANKS_DIR": str(scratch_root / "banks"),
                "DATA_DIR": "/work/data_vdra",
                "MATH_DICT": "/work/source_data/math",
                "MODEL_PATH": model_path,
                "VDRA_B200": "1",
                "TORCH_CUDA_ARCH_LIST": arch,
                "TORCH_EXTENSIONS_DIR": str(extension_cache),
                "SEED": "42",
                "STOP_STEP": "1",
                "RESUME_EVERY": "1",
                "KEEP_EVERY": "1",
                "SAVE_EVERY": "1",
                "MAX_RESTARTS": "0",
                "RESTART_DELAY": "1",
                "EVAL_EVERY_N": "0",
                "EVAL_AT_END": "0",
                "EVAL_STEP0": "0",
                "COMPUTE_MATCHED": "0",
                "SEEDS": "42",
                "GPUS": "0",
                "WANDB_MODE": "offline",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VDRA_CONFIG_JSON": "{}",
                "IGPO_R1_CONFIG": "/opt/work/igpo/configs/polIter_qwen3_1_7b_spo_logged_MATH.jsonnet",
                "IGPO_R2_CONFIG": "/opt/work/igpo/configs/polIter_qwen3_1_7b_vdra_MATH.jsonnet",
                "IGPO_BRANCH_FACTOR_CONFIG": "/opt/work/igpo/configs/qwen3/branch_factor_222.jsonnet",
                "IGPO_EXTRA_CONFIGS": ",".join(
                    [
                        "/opt/work/igpo/configs/qwen3/smoke_short.jsonnet",
                        *(
                            ["/opt/work/igpo/configs/qwen3/smoke_vdra_fast.jsonnet"]
                            if label == "vdra"
                            else []
                        ),
                    ]
                ),
                "INGPO_GPUS": "0",
                "INGPO_SAVE_STEPS": "1",
                "INGPO_CHECKPOINT_KEEP_STEPS": "1",
                "INGPO_NUM_EPISODES_PER_ITERATION": "16",
                "INGPO_DATASET_NUM_SAMPLES_PER_ITERATION": "4",
                "INGPO_TARGET_TRAIN_BATCH_SIZE": "8",
                "INGPO_PER_DEVICE_TRAIN_BATCH_SIZE": "1",
                "INGPO_VLLM_GPU_MEMORY_UTILIZATION": "0.25",
                "INGPO_VLLM_MIN_AVAILABLE_GPU_MEMORY_MB": str(12 * 1024),
                "INGPO_VLLM_MAX_NUM_SEQS": "16",
                "INGPO_VLLM_SWAP_SPACE": "8",
                "INGPO_MAX_CONCURRENT_PROGRAMS": "8",
                "INGPO_MAX_CONCURRENT_GENERATIONS": "16",
                "INGPO_VALUE_MAX_CONCURRENT_PROGRAMS": "16",
                "INGPO_VALUE_MAX_CONCURRENT_GENERATIONS": "32",
                "INGPO_DISABLE_INTERNAL_EVALUATION": "1",
                "INGPO_FAST_OVERLAY": "1",
                "APP_OPENAI_VLLM_API_BASE": "none",
                "PYTHONUNBUFFERED": "1",
                "VDRA_VALIDATE_TOKEN_IDS": "1",
            }
        )
        log_path = artifact_root / f"{label}.log"
        print(
            f"[qwen3-mock] begin {label}: {experiment}, seed 42, one update, "
            f"{gpu_name} (sm_{compute_capability[0]}{compute_capability[1]}), "
            f"max {PER_RUN_TIMEOUT_SECONDS}s; artifacts={artifact_root}",
            flush=True,
        )
        launched = time.time()
        outcome = _run_streamed(
            ["bash", f"/opt/vdra/scripts/{script}"], env, log_path, PER_RUN_TIMEOUT_SECONDS
        )
        arm = {
            "experiment": experiment,
            "started_at_unix": launched,
            **outcome,
        }
        if run_dir.is_dir():
            arm.update(_collect_run_artifacts(run_name, run_dir, artifact_root / label))
        else:
            arm.update(
                {
                    "status": "failed",
                    "resumable_update": 0,
                    "stop": 1,
                    "checkpoint_dirs": [],
                    "training_timing_rows": 0,
                    "tree_log_files": [],
                    "tree_log_rows": 0,
                    "run_meta_events": [],
                }
            )
        arm["status"] = (
            "complete"
            if outcome["returncode"] == 0
            and not outcome["timed_out"]
            and arm.get("resumable_update", 0) >= 1
            and arm.get("has_resumable_checkpoint") is True
            and arm.get("training_timing_rows", 0) > 0
            and arm.get("tree_log_rows", 0) > 0
            and arm.get("logged_strategy_names") == (["spo"] if label == "spo" else ["vdra"])
            else "failed"
        )
        results[label] = arm
        print(
            f"[qwen3-mock] end {label}: status={arm['status']} "
            f"update={arm.get('resumable_update')} wall={outcome['wall_seconds']}s",
            flush=True,
        )

    summary = {
        "status": "complete" if all(item.get("status") == "complete" for item in results.values()) else "failed",
        "tag": tag,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "gpu": gpu_name,
        "compute_capability": list(compute_capability),
        "gpu_types_requested_in_order": GPU_TYPES,
        "seed": 42,
        "updates_per_arm": 1,
        "thinking_mode": True,
        "generation_max_tokens": 256,
        "train_rows": manifest.get("source_training_rows"),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pair_started_at)),
        "arms": results,
    }
    summary["gpu_cost_upper_bound_usd"] = round(
        PAIR_TIMEOUT_SECONDS * GPU_RATE_USD_PER_SECOND, 2
    )
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    VOLUME.commit()
    return summary


@APP.local_entrypoint()
def main(action: str = "smoke", tag: str = "20261029") -> None:
    """Prepare inputs and run the two bounded Qwen3 mock updates."""
    if action not in {"prepare", "smoke"}:
        raise SystemExit("action must be prepare or smoke")
    print(json.dumps(setup_data.remote(), indent=2, sort_keys=True), flush=True)
    if action == "prepare":
        return
    result = run_pair.remote(tag)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result.get("status") != "complete":
        raise SystemExit("Qwen3 SPO/VDRA mock did not complete both one-update runs")
