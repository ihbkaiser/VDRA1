from rebuttal_igpo.rebuttal.eval_avgk import build_command, main


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
    )
    assert command[:3] == ["bash", "/src/scripts/evaluate.sh", "gear_config"]
    assert "--datasets" in command and "aime24,aime25,amc23" in command
    assert "--metrics" not in command
    assert "--num-samples" in command and "16" in command
    assert "--tokenizer" in command and "/checkpoint/tokenizer" in command
    assert "--context-length" in command and "4096" in command
    assert "--max-new-tokens" in command and "1024" in command
    assert command[-3:] == [
        "APP_MINIMIZE_STORED_FILES=False",
        "WANDB_MODE=disabled",
        "WANDB_DISABLE_UPLOAD_FILES_TO_CLOUD=True",
    ]
    assert command[-6:-3] == [
        "APP_DIRECTORY=/tmp/eval", "APP_EXPERIMENT_NAME=avgk16", "APP_SEED=42"
    ]
    assert "APP_MINIMIZE_STORED_FILES=False" in command


def test_eval_wrapper_forwards_source_runtime_options(capsys):
    result = main([
        "--script", "/src/scripts/evaluate.sh",
        "--model", "/checkpoint",
        "--config", "gear_config",
        "--datasets", "aime24",
        "--output-dir", "/tmp/eval",
        "--debug_mode=true",
        "--dry-run",
    ])
    assert result == 0
    assert "--debug_mode=true" in capsys.readouterr().out
