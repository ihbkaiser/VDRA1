"""CPU-only validation for the IGPO R1/R2 training and evaluation configs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import _jsonnet


def _render_runtime_overlay(config_root: Path) -> str:
    """Render the actual fast-runtime overlay without allocating a GPU."""
    script_path = config_root.parent / "scripts" / "_common.sh"
    if not script_path.is_file():
        raise RuntimeError(f"missing IGPO runtime config generator: {script_path}")
    runtime_env = {
        **os.environ,
        "APP_DIRECTORY": "",
        "INGPO_ROOT": str(config_root.parent),
        "MASTER_PORT": "29500",
        "INGPO_FAST_OVERLAY": "1",
        "INGPO_VLLM_GPU_MEMORY_UTILIZATION": "0.25",
        "INGPO_VLLM_MIN_AVAILABLE_GPU_MEMORY_MB": str(20 * 1024),
        "INGPO_WAIT_UNTIL_MEMORY_RELEASE": "false",
        "INGPO_DATASET_NUM_SAMPLES_PER_ITERATION": "4",
        "INGPO_NUM_EPISODES_PER_ITERATION": "64",
        "INGPO_TARGET_TRAIN_BATCH_SIZE": "32",
        "INGPO_VLLM_MAX_NUM_SEQS": "64",
        "INGPO_MAX_CONCURRENT_PROGRAMS": "16",
        "INGPO_MAX_CONCURRENT_GENERATIONS": "64",
        "INGPO_SAVE_STEPS": "2",
        "INGPO_CHECKPOINT_KEEP_STEPS": "2",
    }
    with tempfile.TemporaryDirectory(prefix="igpo-runtime-overlay-") as temp_dir:
        runtime_env["APP_DIRECTORY"] = temp_dir
        normalized_script = Path(temp_dir) / "_common.sh"
        source_text = script_path.read_text(encoding="utf-8")
        normalized_script.write_text(
            source_text.replace("\r\n", "\n").replace("\r", "\n"),
            encoding="utf-8",
            newline="\n",
        )
        runtime_env["INGPO_COMMON_PATH"] = str(normalized_script)
        completed = subprocess.run(
            ["bash", "-c", 'source "$INGPO_COMMON_PATH"; ensure_runtime_config'],
            cwd=config_root.parent,
            env=runtime_env,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "IGPO runtime overlay generation failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        overlay_path = (
            Path(temp_dir)
            / "_generated_configs"
            / "runtime_overrides"
            / "short_run_fast_runtime.jsonnet"
        )
        if not overlay_path.is_file():
            raise RuntimeError("IGPO runtime config generator did not emit its overlay")
        overlay_source = overlay_path.read_text(encoding="utf-8")
    if "enable_prefix_caching:" in overlay_source:
        raise RuntimeError(
            "generated IGPO runtime overlay overrides prefix caching; it must inherit "
            "the selected Jsonnet config"
        )
    return overlay_source


def preflight_igpo_configs(
    config_root: Path = Path("/opt/work/igpo/configs"),
    model_path: str = "/work/hf_cache/model",
) -> dict[str, Any]:
    runtime_overlay_source = _render_runtime_overlay(config_root)
    ext_vars = {
        "APP_SEED": "42",
        "APP_DISABLE_FLASH_ATTENTION": "0",
        "APP_OPENAI_VLLM_API_BASE": "none",
        "APP_VDRA_TRAIN_DATASET_DICT": "/work/data_vdra/math-nodiag",
        "APP_VDRA_MODEL_PATH": model_path,
        "APP_VDRA_STOP_STEP": "4",
        "APP_VDRA_VARIANT_JSON": "{}",
    }
    shared_configs = [
        "episode_generators/branch_factor_666.jsonnet",
        "episode_generators/no_prefix_caching.jsonnet",
        "vdra/data_nodiag.jsonnet",
        "vdra/model_path.jsonnet",
        "vdra/stop.jsonnet",
    ]
    base_configs = {
        "R1": "polIter_deepseekR1Qwen_spo_logged_MATH.jsonnet",
        "R2": "polIter_deepseekR1Qwen_vdra_MATH.jsonnet",
    }
    checked = {}
    for run, base_config in base_configs.items():
        names = [base_config, *shared_configs]
        if run == "R2":
            names.append("vdra/variant.jsonnet")
        names.append("episode_generators/noSamplRplc.jsonnet")
        paths = [config_root / name for name in names]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"{run}: missing IGPO Jsonnet config(s): {missing}")

        no_internal_eval = config_root / "vdra/no_internal_eval.jsonnet"
        if not no_internal_eval.is_file():
            raise RuntimeError(f"{run}: missing Phase D evaluation overlay: {no_internal_eval}")

        imports = "+".join(f"(import {json.dumps(path.as_posix())})" for path in paths)
        effective_imports = imports + "+" + runtime_overlay_source
        config = json.loads(
            _jsonnet.evaluate_snippet(
                f"phase-d-{run}-preflight", effective_imports, ext_vars=ext_vars
            )
        )
        phase_d_config = json.loads(
            _jsonnet.evaluate_snippet(
                f"phase-d-{run}-no-internal-eval-preflight",
                effective_imports
                + f"+(import {json.dumps(no_internal_eval.as_posix())})",
                ext_vars=ext_vars,
            )
        )
        if phase_d_config.get("inference_pipelines") != []:
            raise RuntimeError(f"{run}: Phase D internal evaluation was not disabled")
        episode_generator = config.get("episode_generator") or {}
        vllm_server = episode_generator.get("vllm_server") or {}
        if vllm_server.get("enable_prefix_caching") is not False:
            raise RuntimeError(
                f"{run}: IGPO vLLM prefix caching must be disabled for prompt-logprob stability"
            )
        inference_strategy = episode_generator.get("inference_strategy") or {}
        if run == "R2":
            if "vdra" in config:
                raise RuntimeError(
                    "R2 VDRA settings must be nested in episode_generator.inference_strategy; "
                    "a root-level 'vdra' key is passed to Runtime and is unsupported"
                )
            if inference_strategy.get("type") != "vdra":
                raise RuntimeError("R2 inference strategy is not configured as VDRA")
            vdra_config = inference_strategy.get("vdra") or {}
            missing_vdra = [
                key for key in ("k0", "pilot_tokens", "r", "m", "proxy")
                if key not in vdra_config
            ]
            if missing_vdra:
                raise RuntimeError(f"R2 nested VDRA config is missing {missing_vdra}")
        trainer = config.get("trainer") or {}
        model_paths = {
            "episode_generator.initial_model_name_or_path": episode_generator.get(
                "initial_model_name_or_path"
            ),
            "episode_generator.inference_strategy.guidance_llm.model": (
                (episode_generator.get("inference_strategy") or {}).get("guidance_llm") or {}
            ).get("model"),
            "episode_generator.inference_strategy.guidance_llm.tokenizer_name": (
                (episode_generator.get("inference_strategy") or {}).get("guidance_llm") or {}
            ).get("tokenizer_name"),
            "episode_generator.value_estimation_inference_strategy.guidance_llm.model": (
                (episode_generator.get("value_estimation_inference_strategy") or {}).get(
                    "guidance_llm"
                )
                or {}
            ).get("model"),
            "episode_generator.value_estimation_inference_strategy.guidance_llm.tokenizer_name": (
                (episode_generator.get("value_estimation_inference_strategy") or {}).get(
                    "guidance_llm"
                )
                or {}
            ).get("tokenizer_name"),
            "tokenizer.hf_model_name": (config.get("tokenizer") or {}).get("hf_model_name"),
            "trainer.actor_model.hf_model_name": (
                (trainer.get("actor_model") or {}).get("hf_model_name")
            ),
            "trainer.reference_model.hf_model_name": (
                (trainer.get("reference_model") or {}).get("hf_model_name")
            ),
        }
        wrong_paths = {
            name: value for name, value in model_paths.items() if value != model_path
        }
        if wrong_paths:
            raise RuntimeError(
                f"{run}: model path overrides differ from {model_path!r}: {wrong_paths}"
            )

        expected_stop_step = int(ext_vars["APP_VDRA_STOP_STEP"])
        actual_stop_step = config.get("early_stop_iteration")
        if actual_stop_step != expected_stop_step:
            raise RuntimeError(
                f"{run}: early_stop_iteration is {actual_stop_step!r}; "
                f"expected {expected_stop_step}"
            )

        pipelines = config.get("inference_pipelines") or []
        if len(pipelines) != 6:
            raise RuntimeError(
                f"{run}: expected 6 inference_pipelines, found {len(pipelines)}"
            )
        paths_found = {}
        for pipeline in pipelines:
            name = pipeline.get("inference_name", "<unnamed>")
            strategy = pipeline.get("inference_strategy") or {}
            node_expander = strategy.get("node_expander") or {}
            tokenizer_path = (node_expander.get("tokenizer") or {}).get("hf_model_name")
            guidance = strategy.get("guidance_llm") or {}
            if tokenizer_path != model_path:
                raise RuntimeError(
                    f"{run}/{name}: evaluation tokenizer path is {tokenizer_path!r}; "
                    f"expected shared model path {model_path!r}"
                )
            if guidance.get("model") != model_path:
                raise RuntimeError(
                    f"{run}/{name}: guidance model path is {guidance.get('model')!r}; "
                    f"expected shared model path {model_path!r}"
                )
            if guidance.get("tokenizer_name") != model_path:
                raise RuntimeError(
                    f"{run}/{name}: guidance tokenizer path is {guidance.get('tokenizer_name')!r}; "
                    f"expected shared model path {model_path!r}"
                )
            paths_found[name] = tokenizer_path
        checked[run] = {
            "pipeline_count": len(pipelines),
            "phase_d_internal_pipeline_count": len(phase_d_config["inference_pipelines"]),
            "early_stop_iteration": actual_stop_step,
            "model_paths": model_paths,
            "tokenizer_paths": paths_found,
            "vdra_config": (
                inference_strategy.get("vdra") if run == "R2" else None
            ),
            "runtime_overlay_generated": True,
            "runtime_overlay_prefix_cache_field": False,
        }
    return {"status": "pass", "expected_model_path": model_path, "runs": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-root", type=Path, default=Path("/opt/work/igpo/configs"))
    parser.add_argument("--model-path", default="/work/hf_cache/model")
    args = parser.parse_args()
    print(json.dumps(preflight_igpo_configs(args.configs_root, args.model_path), sort_keys=True))


if __name__ == "__main__":
    main()
