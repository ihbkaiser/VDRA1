"""Lightweight CPU-only Modal app for IGPO R1/R2 Jsonnet preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT_REMOTE = "/opt/work/igpo/configs"
CONFIG_ROOT = Path(CONFIG_ROOT_REMOTE)
MODEL_PATH = "/work/hf_cache/model"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("bash")
    .pip_install("jsonnet==0.20.0")
    .add_local_dir(ROOT / "work" / "igpo" / "configs", CONFIG_ROOT_REMOTE, copy=True)
    .add_local_file(
        ROOT / "work" / "igpo" / "scripts" / "_common.sh",
        "/opt/work/igpo/scripts/_common.sh",
        copy=True,
    )
    .add_local_dir(
        ROOT / "integrations" / "igpo" / "configs" / "vdra",
        f"{CONFIG_ROOT_REMOTE}/vdra",
        copy=True,
    )
    .add_local_file(
        ROOT / "modal_app" / "igpo_config_preflight.py",
        "/root/igpo_config_preflight.py",
        copy=True,
    )
)

app = modal.App("vdra-igpo-config-preflight")


@app.function(image=image, cpu=1, timeout=5 * 60, retries=0, scaledown_window=2)
def preflight_igpo_eval_config() -> dict[str, Any]:
    """Validate R1/R2 paths and stop step without building training images."""
    from igpo_config_preflight import preflight_igpo_configs

    return preflight_igpo_configs(CONFIG_ROOT, MODEL_PATH)
