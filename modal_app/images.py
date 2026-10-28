"""Pinned common CUDA/PyTorch/vLLM stack and framework-specific probe layers."""

from __future__ import annotations

from pathlib import Path

import modal


CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
PYTHON_VERSION = "3.10"
TORCH_VERSION = "2.7.0"
TORCHVISION_VERSION = "0.22.0"
TORCHAUDIO_VERSION = "2.7.0"
VLLM_VERSION = "0.9.2"
TRANSFORMERS_VERSION = "4.51.3"
TOKENIZERS_VERSION = "0.21.1"
DEEPSPEED_VERSION = "0.16.5"
ROOT = Path(__file__).resolve().parent.parent

_PROJECT_IGNORE = [
    "work",
    "remote_artifacts",
    "**/FETCH_HEAD",
    ".pytest_cache",
    "**/__pycache__",
    "*.pyc",
    "codex/dry_runs",
    "codex/phase_b_results",
    "codex/STATUS.md",
]
_CHECKOUT_IGNORE = [
    "**/__pycache__",
    "*.pyc",
    # Git worktree FETCH_HEAD files can change asynchronously during Modal's
    # source snapshot. They are not needed for `git rev-parse HEAD` metadata.
    "**/FETCH_HEAD",
    "**/data",
    "**/outputs",
    "**/runs",
]


def common_image() -> modal.Image:
    """Build the same CUDA, Torch and vLLM core for every framework."""
    image = (
        modal.Image.from_registry(CUDA_BASE, add_python=PYTHON_VERSION)
        .apt_install(
            "build-essential",
            "clang",
            "git",
            "libaio-dev",
            "libnuma1",
            "ninja-build",
            "pkg-config",
            "unzip",
            "wget",
        )
        .pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            f"torchaudio=={TORCHAUDIO_VERSION}",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            f"vllm=={VLLM_VERSION}",
            f"transformers=={TRANSFORMERS_VERSION}",
            f"tokenizers=={TOKENIZERS_VERSION}",
            "openai==1.90.0",
            "numpy==1.26.4",
            "sentencepiece==0.2.0",
            "jsonnet==0.20.0",
            "datasets==2.17.1",
            "accelerate==0.33.0",
            "ray[default]==2.43.0",
            "hydra-core==1.3.2",
            "codetiming==1.4.0",
            # The CUDA base image has no Modal SDK dependencies. The runtime
            # injects /pkg/modal, so install its pinned SDK dependencies too.
            "modal==1.6.1",
            "dill==0.3.8",
            "pandas==2.2.3",
            "pyarrow==19.0.1",
            "tensordict==0.6.2",
            "torchdata==0.11.0",
            "scipy==1.15.2",
            "einops==0.8.1",
            "jsonlines==4.0.0",
            "torchmetrics==1.7.0",
            "bitsandbytes==0.45.5",
            "word2number==1.1",
            "math-verify==0.8.0",
            "sortedcontainers==2.4.0",
            "matplotlib==3.8.3",
            "py-cpuinfo==9.0.0",
            "peft==0.15.2",
            "packaging>=24.2",
            "pyyaml>=6.0",
            "psutil>=5.9.8",
            "tqdm>=4.66.2",
            # vLLM 0.9.2's current OpenTelemetry exporter requires protobuf 5+;
            # this W&B release supports that range and preserves local logging.
            "protobuf==5.29.5",
            "wandb==0.19.11",
        )
        .run_commands(
            "python -c 'import grpclib, modal; print(modal.__version__)' "
            "&& python -c 'import accelerate, peft; "
            "from accelerate.utils.memory import clear_device_cache; "
            "print(f\"PEFT {peft.__version__} with Accelerate {accelerate.__version__} OK\")'"
        )
        .env(
            {
                # TreeRL's integration hooks into vLLM's V0 Worker API.
                "VLLM_USE_V1": "0",
                "WORK": "/work",
                "VDRA_HOME": "/opt/vdra",
                "VDRA_B200": "1",
                "RUNS_DIR": "/work/runs",
                "RESULTS_DIR": "/work/results",
                "BANKS_DIR": "/work/banks",
                "DATA_DIR": "/work/data_vdra",
                "MATH_DICT": "/work/source_data/math",
                "IGPO_DIR": "/opt/work/igpo",
                "TREEPO_DIR": "/opt/work/TreePO",
                "TREERL_DIR": "/opt/work/TreeRL",
                "CONDA_IGPO": "",
                "CONDA_TREEPO": "",
                "CONDA_TREERL": "",
                "GPUS_PER_RUN": "1",
                "GPU_GROUPS": "0",
                "TREERL_ACTOR_GPUS": "1",
                "TREERL_REF_GPUS": "1",
                "TREERL_VLLM_ENGINES": "1",
                "HF_HOME": "/work/hf_cache",
                "HF_HUB_CACHE": "/work/hf_cache/hub",
                "HF_DATASETS_CACHE": "/work/hf_cache/datasets",
                "TMPDIR": "/tmp",
                # Build DeepSpeed extensions for the B200 once in the image and
                # reuse them at runtime instead of JIT-compiling on the first update.
                "TORCH_EXTENSIONS_DIR": "/opt/torch_extensions",
                "TORCH_CUDA_ARCH_LIST": "10.0",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": "/opt/vdra:/opt/work/igpo:/opt/work/TreePO:/opt/work/TreeRL:/opt/vdra/modal_app/runtime",
            }
        )
    )
    # The 2.8.3 release publishes CUDA 12 / Torch 2.7 wheels for both PyTorch
    # C++ ABIs. Select the wheel from the ABI of the already-installed Torch.
    image = image.run_commands(
        "python - <<'PY'\n"
        "import subprocess, sys, torch\n"
        "abi = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'\n"
        "url = ('https://github.com/Dao-AILab/flash-attention/releases/download/'\n"
        "       'v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abi' + abi +\n"
        "       '-cp310-cp310-linux_x86_64.whl')\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', url])\n"
        "PY"
    )
    image = image.add_local_dir(ROOT, "/opt/vdra", copy=True, ignore=_PROJECT_IGNORE)
    for source, destination in (
        (ROOT / "work" / "igpo", "/opt/work/igpo"),
        (ROOT / "work" / "TreePO", "/opt/work/TreePO"),
        (ROOT / "work" / "TreeRL", "/opt/work/TreeRL"),
    ):
        image = image.add_local_dir(source, destination, copy=True, ignore=_CHECKOUT_IGNORE)
    return image


def framework_images() -> dict[str, modal.Image]:
    """Return one derived image per framework, sharing the exact core stack."""
    common = common_image()
    deepspeed = common.pip_install(f"deepspeed=={DEEPSPEED_VERSION}")
    igpo = deepspeed.pip_install(
        "fire==0.5.0",
        "overrides==7.7.0",
        "base58==2.1.1",
        "evaluate==0.4.1",
        "graphviz==0.20.1",
        "nltk==3.9.1",
        "plotnine==0.13.0",
        "peft==0.8.2",
        "pylatexenc==2.10",
        "pygtrie==2.5.0",
    ).run_commands(
        "cp /opt/vdra/integrations/igpo/configs/vdra/*.jsonnet /opt/work/igpo/configs/vdra/ "
        "&& mkdir -p /opt/work/igpo/configs/model_overrides /opt/work/igpo/configs/qwen3 "
        "&& cp /opt/vdra/integrations/igpo/configs/model_overrides/qwen3_1_7b.jsonnet /opt/work/igpo/configs/model_overrides/ "
        "&& cp /opt/vdra/integrations/igpo/configs/qwen3/*.jsonnet /opt/work/igpo/configs/qwen3/ "
        "&& cp /opt/vdra/integrations/igpo/configs/polIter_qwen3_1_7b_*.jsonnet /opt/work/igpo/configs/ "
        "&& python -c \"from pathlib import Path; from modal_app.script_utils import normalize_shell_script; "
        "normalize_shell_script(Path('/opt/work/igpo/scripts/_common.sh')); "
        "normalize_shell_script(Path('/opt/work/igpo/scripts/start_vllm_server_named_params.sh'))\" "
        "&& bash -n /opt/work/igpo/scripts/_common.sh "
        "&& bash -n /opt/work/igpo/scripts/start_vllm_server_named_params.sh "
        "&& bash -n /opt/vdra/scripts/20_train.sh "
        "&& bash -n /opt/vdra/scripts/qwen3_spo.sh "
        "&& bash -n /opt/vdra/scripts/qwen3_vdra.sh "
        "&& python -c \"from pathlib import Path; "
        "g=Path('/opt/work/igpo/treetune/episode_generators/on_policy_episode_generator.py').read_text(); "
        "h=Path('/opt/work/igpo/treetune/episode_generators/hybrid_episode_generator.py').read_text(); "
        "v=Path('/opt/work/igpo/treetune/common/vllm_server.py').read_text(); "
        "assert 'Timing [' in g and 'Timing [iteration=' in h and '_stop_owned_processes' in v; "
        "print('IGPO timing and vLLM cleanup markers OK')\" "
        "&& python -c \"import fire, overrides, base58, evaluate, graphviz, nltk, plotnine, peft, accelerate, "
        "pylatexenc, pygtrie; import treetune.main, treetune.common, "
        "treetune.tasks.math_grader; print('IGPO runtime imports OK')\" "
        "&& python -c \"import _jsonnet; p='/opt/work/igpo/configs/vdra/'; "
        "e={'APP_VDRA_TRAIN_DATASET_DICT':'/work/data_vdra/math-nodiag', "
        "'APP_VDRA_MODEL_PATH':'/work/hf_cache/model','APP_VDRA_STOP_STEP':'4', "
        "'APP_VDRA_VARIANT_JSON':'{}'}; "
        "[_jsonnet.evaluate_file(p+n, ext_vars=e) for n in "
        "['data_nodiag.jsonnet','stop.jsonnet','variant.jsonnet']]; "
        "print('IGPO VDRA Jsonnet configs OK')\" "
        "&& python /opt/vdra/modal_app/igpo_config_preflight.py "
        "--configs-root /opt/work/igpo/configs --model-path /work/hf_cache/model"
    )
    treepo = common.run_commands("python -m pip install --no-deps -e /opt/work/TreePO")
    treerl = deepspeed.pip_install("Levenshtein==0.27.4").run_commands(
        "python -m pip install --no-deps -e /opt/work/TreeRL "
        "&& python -c 'from transformers.integrations import HfDeepSpeedConfig; "
        "import openrlhf.models.actor, openrlhf.models.model; "
        "import openrlhf.trainer.ppo_utils.parallel_mcts; "
        "print(\"TreeRL Transformers DeepSpeed imports OK\")'"
    )
    return {
        "igpo": igpo,
        "treepo": treepo,
        "treerl": treerl,
    }
