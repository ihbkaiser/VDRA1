import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bash_executable():
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe",
            Path("C:/Program Files/Git/bin/bash.exe"),
            Path("C:/Program Files/Git/usr/bin/bash.exe"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return shutil.which("bash") or "bash"


def _resolved_command(exp, tmp_path):
    bash = Path(_bash_executable())
    env = os.environ.copy()
    temp = tmp_path / "resolved"
    env.update(
        {
            "VDRA_HOME": str(ROOT),
            "VDRA_B200": "1",
            "DRY_RUN": "1",
            "SEEDS": "42",
            "STOP_STEP": "4",
            "EVAL_EVERY_N": "2",
            "EVAL_STEP0": "0",
            "EVAL_AT_END": "0",
            "RESUME_EVERY": "2",
            "SAVE_EVERY": "2",
            "SAVE_EVERY_CONT": "2",
            "COMPUTE_MATCHED": "0",
            "MAX_RESTARTS": "0",
            "GPUS": "0",
            "WORK": str(temp / "work"),
            "RUNS_DIR": str(temp / "runs"),
            "RESULTS_DIR": str(temp / "results"),
            "BANKS_DIR": str(temp / "banks"),
            "DATA_DIR": str(temp / "data"),
            "MATH_DICT": str(temp / "math"),
            "IGPO_DIR": str(ROOT / "work/igpo"),
            "TREEPO_DIR": str(ROOT / "work/TreePO"),
            "TREERL_DIR": str(ROOT / "work/TreeRL"),
            "MODEL_PATH": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "VDRA_CONFIG_JSON": "{}",
            "VDRA_VARIANT_JSON": "{}",
            "HF_HUB_OFFLINE": "1",
        }
    )
    bash_env = temp / "bash_env.sh"
    bash_env.parent.mkdir(parents=True, exist_ok=True)
    python_command = 'py -3.12 "$@"' if sys.platform == "win32" else 'command python3 "$@"'
    bash_env.write_text(
        "python3() { case \" $* \" in *AutoTokenizer*) printf '151643\\n';; "
        f"*) {python_command};; esac; }}\n",
        encoding="utf-8",
    )
    env["BASH_ENV"] = str(bash_env)
    if sys.platform == "win32":
        env["PATH"] = str(bash.parent) + ";" + str(bash.parent.parent / "usr" / "bin") + ";" + env.get("PATH", "")

    run_id = f"parity_{exp}"
    result = subprocess.run(
        [str(bash), str(ROOT / "scripts/20_train.sh"), exp, "42", "4", run_id],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, f"{exp} dry launch failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"

    prefix = {
        "R1": "+ ingpo_run ",
        "R2": "+ ingpo_run ",
        "R3": "+ python3 -m recipe.dapo.main_dapo ",
        "R4": "+ python3 -m recipe.dapo.main_dapo ",
        "R5": "+ python3 train_reinforce_ray.py ",
        "R6": "+ python3 train_reinforce_ray.py ",
    }[exp]
    combined_output = result.stdout + "\n" + result.stderr
    command_lines = [line.split("+ ", 1)[1].strip() for line in combined_output.splitlines() if prefix in line]
    assert len(command_lines) == 1, (
        f"{exp} should resolve exactly one trainer command, got {command_lines}; "
        f"output:\n{combined_output[-3500:]}"
    )
    return command_lines[0], combined_output


def _normalize_run_id(command, exp):
    return command.replace(f"parity_{exp}", "<RUN_ID>")


def test_resolved_pair_launch_commands_differ_only_by_the_method_switch(tmp_path):
    commands = {}
    outputs = {}
    for exp in ("R1", "R2", "R3", "R4", "R5", "R6"):
        commands[exp], outputs[exp] = _resolved_command(exp, tmp_path)

    # TreeTune's method switch is its policy-iteration config plus the VDRA variant config.
    r1, r2 = _normalize_run_id(commands["R1"], "R1"), _normalize_run_id(commands["R2"], "R2")
    assert "polIter_deepseekR1Qwen_spo_logged_MATH.jsonnet" in r1 and "vdra/variant.jsonnet" not in r1
    assert "polIter_deepseekR1Qwen_vdra_MATH.jsonnet" in r2 and r2.count("vdra/variant.jsonnet") == 1
    r1 = re.sub(r"polIter_deepseekR1Qwen_spo_logged_MATH\.jsonnet", "<POLICY_CONFIG>", r1)
    r2 = re.sub(r"polIter_deepseekR1Qwen_vdra_MATH\.jsonnet", "<POLICY_CONFIG>", r2)
    r2 = re.sub(r",[^,\s]*/configs/vdra/variant\.jsonnet", "", r2)
    assert r1 == r2

    # TreePO changes only the divergence policy and supplies the configured VDRA keys.
    r3, r4 = _normalize_run_id(commands["R3"], "R3"), _normalize_run_id(commands["R4"], "R4")
    assert "tree_div_policy=fixed_avg" in r3 and "tree_div_policy=vdra" not in r3
    assert "tree_div_policy=vdra" in r4 and "tree_div_policy=fixed_avg" not in r4
    r3 = re.sub(r"\+actor_rollout_ref\.rollout\.tree_div_policy=fixed_avg", "+actor_rollout_ref.rollout.tree_div_policy=<METHOD>", r3)
    r4 = re.sub(r"\+actor_rollout_ref\.rollout\.tree_div_policy=vdra", "+actor_rollout_ref.rollout.tree_div_policy=<METHOD>", r4)
    r4 = " ".join(token for token in r4.split() if not token.startswith("+actor_rollout_ref.rollout.vdra."))
    assert r3 == r4

    # TreeRL uses the same trainer command; its one method switch is the allocation mode.
    r5, r6 = _normalize_run_id(commands["R5"], "R5"), _normalize_run_id(commands["R6"], "R6")
    assert r5 == r6
    method_lines = {}
    for exp in ("R5", "R6"):
        method_lines[exp] = [
            line.split("+ method ", 1)[1].strip()
            for line in outputs[exp].splitlines()
            if "+ method VDRA_TREERL_ALLOC=" in line
        ]
        assert len(method_lines[exp]) == 1
    assert method_lines["R5"] == ["VDRA_TREERL_ALLOC=off"]
    assert method_lines["R6"] == ["VDRA_TREERL_ALLOC=vdra"]
