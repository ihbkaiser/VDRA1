import os
import json
import subprocess
from pathlib import Path

from rebuttal_igpo.rebuttal.eval_avgk import build_command, main


def test_shared_tokenizer_overlay_selects_evaluation_runtime():
    import _jsonnet

    overlay = Path(__file__).parents[1] / "configs" / "eval_shared_tokenizer_server.jsonnet"
    config = json.loads(_jsonnet.evaluate_file(
        str(overlay),
        ext_vars={"APP_VLLM_SHARED_TOKENIZER_SERVER": "/tmp/shared-tokenizer-server.sh"},
    ))
    assert config["runtime_type"] == "policy_iteration"
    assert config["evaluation_vllm_server"]["script_path"] == "/tmp/shared-tokenizer-server.sh"


def test_aime24_overlay_maps_archived_column_names():
    import _jsonnet
    import json

    overlay = Path(__file__).parents[1] / "configs" / "eval_aime24_field_names.jsonnet"
    snippet = f"""
      local base = {{ inference_pipelines: [
        {{ inference_name: 'aime24_test', task: {{ problem_field: 'problem', solution_field: 'solution' }} }},
        {{ inference_name: 'aime25_test', task: {{ problem_field: 'problem', answer_field: 'answer' }} }},
      ] }};
      base + (import {json.dumps(str(overlay))})
    """
    config = json.loads(_jsonnet.evaluate_snippet("aime24_fields_test", snippet))
    pipelines = {row["inference_name"]: row for row in config["inference_pipelines"]}
    assert pipelines["aime24_test"]["task"]["problem_field"] == "Problem"
    assert pipelines["aime24_test"]["task"]["solution_field"] == "Solution"
    assert pipelines["aime25_test"]["task"]["problem_field"] == "problem"


def test_eval_wrapper_uses_supported_source_cli_and_fixed_protocol():
    command = build_command(
        "/src/scripts/evaluate.sh",
        config="gear_config",
        model="/checkpoint/hf_pretrained",
        datasets=["aime24", "aime25", "amc23"],
        output_dir="/tmp/eval",
        samples=16,
        seed=42,
        tokenizer="/checkpoint/tokenizer",
        context_length=4096,
        max_new_tokens=1024,
        generation_overlay="/configs/eval_generation_0.6_0.95.jsonnet",
        server_overlay="/configs/eval_shared_tokenizer_server.jsonnet",
    )
    assert command[:3] == ["bash", "/src/scripts/evaluate.sh", "gear_config"]
    assert "--datasets" in command and "aime24,aime25,amc23" in command
    assert "--metrics" not in command
    assert "--num-samples" in command and "16" in command
    assert "--tokenizer" in command and "/checkpoint/tokenizer" in command
    assert "--context-length" in command and "4096" in command
    assert "--max-new-tokens" in command and "1024" in command
    assert command[-4:-1] == [
        "APP_MINIMIZE_STORED_FILES=False",
        "WANDB_MODE=disabled",
        "WANDB_DISABLE_UPLOAD_FILES_TO_CLOUD=True",
    ]
    assert command[-7:-4] == [
        "APP_DIRECTORY=/tmp/eval", "APP_EXPERIMENT_NAME=avgk16", "APP_SEED=42"
    ]
    assert "APP_MINIMIZE_STORED_FILES=False" in command
    assert command[-1].startswith("APP_VLLM_SHARED_TOKENIZER_SERVER=")
    assert "--config" in command and "/configs/eval_shared_tokenizer_server.jsonnet" in command
    assert any(arg.startswith("APP_VLLM_SHARED_TOKENIZER_SERVER=") for arg in command)


def test_eval_wrapper_forwards_source_runtime_options(capsys):
    result = main([
        "--script", "/src/scripts/evaluate.sh",
        "--model", "/checkpoint",
        "--config", "gear_config",
        "--datasets", "aime24",
        "--output-dir", "/tmp/eval",
        "--server-overlay", "/configs/eval_shared_tokenizer_server.jsonnet",
        "--debug_mode=true",
        "--dry-run",
    ])
    assert result == 0
    output = capsys.readouterr().out
    assert "--debug_mode=true" in output
    assert "--config /configs/eval_shared_tokenizer_server.jsonnet" in output


def test_shared_tokenizer_vllm_launcher_passes_tokenizer_and_loopback(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "captured_args.txt"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n', encoding="utf-8"
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "APP_EVAL_TOKENIZER": "/shared/tokenizer",
        "CAPTURE_ARGS": str(capture),
    })
    script = Path(__file__).parents[1] / "rebuttal" / "start_vllm_server_shared_tokenizer.sh"
    subprocess.run(
        [str(script), "--model", "/spo/checkpoint", "--port", "12345",
         "--seed", "42", "--gpu-memory-utilization", "0.7",
         "--max-num-seqs", "16", "--max-model-len", "4096"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--model") + 1] == "/spo/checkpoint"
    assert args[args.index("--tokenizer") + 1] == "/shared/tokenizer"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--max-model-len") + 1] == "4096"
