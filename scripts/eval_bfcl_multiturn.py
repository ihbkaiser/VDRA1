#!/usr/bin/env python3
"""Generate and score BFCL v4 multi-turn results through an OpenAI-compatible vLLM API.

Run from the ``vdra_experiments`` repository root:

    python scripts/eval_bfcl_multiturn.py

The script uses BFCL's stateful multi-turn evaluator and writes one combined
JSON report, raw generation JSONL files, and the official BFCL score CSVs.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

# Match the endpoint and model from the supplied EvalHub evaluation example.
BASE_URL = os.getenv(
    "BFCL_OPENAI_BASE_URL",
    "https://vai-agent-super-nvfp4-runai-data-process.runai-inference.cyberspace.vn",
)
API_KEY = os.getenv("OPENAI_API_KEY") or "dummy"
MODEL_NAME = os.getenv("BFCL_VLLM_MODEL", "VAI-1.4-30B")
MODEL_ALIAS = "evalhub-vllm-local"
TEST_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]


def optional_bool_env(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, or yes/no")


# BFCL generic-handler defaults. Raise the thread count only to match the
# endpoint's documented concurrency limits. The other sampling overrides are
# optional and should be set only for a model-specific serving profile.
NUM_THREADS = int(os.getenv("BFCL_NUM_THREADS", "1"))
TEMPERATURE = float(os.getenv("BFCL_TEMPERATURE", "0.001"))
TOP_P = (
    float(os.environ["BFCL_TOP_P"])
    if os.getenv("BFCL_TOP_P")
    else None
)
MAX_COMPLETION_TOKENS = (
    int(os.environ["BFCL_MAX_COMPLETION_TOKENS"])
    if os.getenv("BFCL_MAX_COMPLETION_TOKENS")
    else None
)
THINK = optional_bool_env("BFCL_ENABLE_THINKING")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("BFCL_TIMEOUT_SECONDS", "3000"))
VERIFY_TLS = os.getenv("BFCL_VERIFY_TLS", "0").strip().lower() in {"1", "true", "yes"}
ALLOW_OVERWRITE = os.getenv("BFCL_ALLOW_OVERWRITE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

OUTPUT_FILE = (
    REPO_ROOT
    / "outputs"
    / "rlvr_nemo_gym-30B"
    / "rlvr-Step-54-bfcl-multiturn.json"
)
BFCL_ROOT = OUTPUT_FILE.with_suffix("")
RESULT_ROOT = BFCL_ROOT / "result"
SCORE_ROOT = BFCL_ROOT / "score"


def normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("BFCL_OPENAI_BASE_URL must be an http(s) URL")
    if parsed.path.rstrip("/").endswith("/v1"):
        return value
    return f"{value}/v1"


BASE_URL = normalize_base_url(BASE_URL)

def _load_bfcl_runtime():
    """Set BFCL paths before imports and return the official entry points."""
    # Prefer the adjacent editable package in this workspace; fall back to the
    # package installed in the active Python environment on the evaluation host.
    evalhub_src = WORKSPACE_ROOT / "evalhub-pkg" / "src"
    if evalhub_src.is_dir():
        sys.path.insert(0, str(evalhub_src))

    os.environ["BFCL_PROJECT_ROOT"] = str(BFCL_ROOT)
    os.environ["BFCL_VLLM_MODEL"] = MODEL_NAME
    os.environ["OPENAI_BASE_URL"] = BASE_URL
    os.environ["OPENAI_API_KEY"] = API_KEY

    import httpx

    from bfcl_eval._llm_response_generation import main as generate_bfcl_results
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.eval_checker.eval_runner import main as evaluate_bfcl_results
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    class ConfiguredVLLMHandler(OpenAICompletionsHandler):
        """Use BFCL defaults and optional model-specific API overrides."""

        def _build_client_kwargs(self):
            kwargs = super()._build_client_kwargs()
            kwargs["http_client"] = httpx.Client(
                verify=VERIFY_TLS,
                timeout=httpx.Timeout(
                    REQUEST_TIMEOUT_SECONDS,
                    connect=min(60.0, REQUEST_TIMEOUT_SECONDS),
                ),
                limits=httpx.Limits(
                    max_connections=NUM_THREADS,
                    max_keepalive_connections=NUM_THREADS,
                ),
            )
            kwargs["timeout"] = REQUEST_TIMEOUT_SECONDS
            return kwargs

        def generate_with_backoff(self, **kwargs):
            if TOP_P is not None:
                kwargs["top_p"] = TOP_P
            if MAX_COMPLETION_TOKENS is not None:
                kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
            if THINK is not None:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": THINK}
                }
            return super().generate_with_backoff(**kwargs)

    return (
        generate_bfcl_results,
        evaluate_bfcl_results,
        MODEL_CONFIG_MAPPING,
        ConfiguredVLLMHandler,
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _find_category_file(root: Path, category: str, suffix: str) -> Path:
    name = f"BFCL_v4_{category}_{suffix}.json"
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {name} below {root}, found {len(matches)}"
        )
    return matches[0]


def _load_category_results() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    category_scores: dict[str, dict] = {}
    category_results: dict[str, list[dict]] = {}

    for category in TEST_CATEGORIES:
        raw_path = _find_category_file(
            RESULT_ROOT / MODEL_ALIAS, category, "result"
        )
        score_path = _find_category_file(
            SCORE_ROOT / MODEL_ALIAS, category, "score"
        )
        raw_rows = _read_jsonl(raw_path)
        score_rows = _read_jsonl(score_path)
        if not score_rows:
            raise ValueError(f"BFCL produced an empty score file: {score_path}")

        category_scores[category] = score_rows[0]
        failures = {
            str(row["id"]): row for row in score_rows[1:] if "id" in row
        }
        cases = []
        for raw in raw_rows:
            case = {
                "id": raw["id"],
                "result": raw.get("result"),
                "valid": str(raw["id"]) not in failures,
            }
            if "inference_log" in raw:
                case["inference_log"] = raw["inference_log"]
            if str(raw["id"]) in failures:
                case["score_details"] = failures[str(raw["id"])]
            cases.append(case)
        category_results[category] = cases

    return category_scores, category_results


def _load_official_multi_turn_row() -> dict[str, str]:
    path = SCORE_ROOT / "data_multi_turn.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        for row in rows:
            if row.get("Model") == MODEL_NAME:
                return row
    raise LookupError(f"Could not find {MODEL_NAME!r} in {path}")


def _write_report() -> None:
    category_scores, category_results = _load_category_results()
    report = {
        "benchmark": "BFCL_v4",
        "model": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "endpoint": BASE_URL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "enable_thinking": THINK,
            "num_threads": NUM_THREADS,
        },
        "official_multi_turn_score": _load_official_multi_turn_row(),
        "category_scores": category_scores,
        "results": category_results,
        "artifacts": {
            "bfcl_root": str(BFCL_ROOT),
            "raw_results": str(RESULT_ROOT / MODEL_ALIAS),
            "score_files": str(SCORE_ROOT / MODEL_ALIAS),
            "official_multi_turn_csv": str(SCORE_ROOT / "data_multi_turn.csv"),
        },
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if NUM_THREADS < 1:
        raise ValueError("BFCL_NUM_THREADS must be at least 1")
    if REQUEST_TIMEOUT_SECONDS <= 0:
        raise ValueError("BFCL_TIMEOUT_SECONDS must be positive")
    if MAX_COMPLETION_TOKENS is not None and MAX_COMPLETION_TOKENS < 1:
        raise ValueError("BFCL_MAX_COMPLETION_TOKENS must be positive")
    if OUTPUT_FILE.exists() and not ALLOW_OVERWRITE:
        raise FileExistsError(f"Output already exists: {OUTPUT_FILE}")

    generate_bfcl_results, evaluate_bfcl_results, model_configs, handler_class = (
        _load_bfcl_runtime()
    )
    config = model_configs[MODEL_ALIAS]
    config.model_name = MODEL_NAME
    config.display_name = MODEL_NAME
    config.model_handler = handler_class

    print("BFCL categories:", TEST_CATEGORIES)
    print("vLLM endpoint:", BASE_URL)
    print("Served model:", MODEL_NAME)
    print("Concurrent BFCL threads:", NUM_THREADS)
    print("Report file:", OUTPUT_FILE)

    generation_args = SimpleNamespace(
        model=[MODEL_ALIAS],
        test_category=TEST_CATEGORIES,
        temperature=TEMPERATURE,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=NUM_THREADS,
        gpu_memory_utilization=0.9,
        backend="vllm",
        skip_server_setup=True,
        local_model_path=None,
        result_dir=None,
        allow_overwrite=ALLOW_OVERWRITE,
        run_ids=False,
        enable_lora=False,
        max_lora_rank=None,
        lora_modules=None,
    )

    generate_bfcl_results(generation_args)
    evaluate_bfcl_results(
        [MODEL_ALIAS],
        TEST_CATEGORIES,
        result_dir=None,
        score_dir=None,
        partial_eval=False,
    )
    _write_report()

    score = _load_official_multi_turn_row()
    print("Official BFCL multi-turn score:", score.get("Multi Turn Acc"))
    print("Saved combined report to:", OUTPUT_FILE)
    print("Official score CSV:", SCORE_ROOT / "data_multi_turn.csv")


if __name__ == "__main__":
    main()
