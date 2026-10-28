#!/usr/bin/env python3
"""Sync Qwen3 VDRA code/configs into the IGPO source in an offline bundle."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[qwen3] {message}")


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        fail(f"required source file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.read_bytes() != source.read_bytes():
        shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        fail(f"required source directory is missing: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            copy_file(path, destination / path.relative_to(source))


def replace_once(path: Path, old: str, new: str, label: str, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    if old not in text:
        fail(f"cannot apply {label}; source does not match: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_qwen3_top_k(igpo: Path) -> None:
    path = igpo / "guidance/llms/_openai_vllm.py"
    old = (
        '        _ = kwargs.pop("request_timeout", None)\n'
        "\n"
        "        if self.chat_mode:\n"
    )
    new = (
        '        _ = kwargs.pop("request_timeout", None)\n'
        "\n"
        "        # vLLM accepts top_k in the request body, but the OpenAI SDK\n"
        "        # rejects it as a top-level keyword.\n"
        '        top_k = kwargs.pop("top_k", None)\n'
        "        if top_k is not None:\n"
        '            extra_body = dict(kwargs.get("extra_body") or {})\n'
        '            extra_body["top_k"] = top_k\n'
        '            kwargs["extra_body"] = extra_body\n'
        "\n"
        "        if self.chat_mode:\n"
    )
    replace_once(path, old, new, "Qwen3 top_k SDK compatibility", 'top_k = kwargs.pop("top_k", None)')

    old = '            "echo": kwargs.get("echo", False),\n        }\n        if self.chat_mode:\n'
    new = (
        '            "echo": kwargs.get("echo", False),\n'
        "        }\n"
        '        if kwargs.get("top_k") is not None:\n'
        '            data["top_k"] = kwargs["top_k"]\n'
        "        if self.chat_mode:\n"
    )
    replace_once(path, old, new, "Qwen3 top_k REST compatibility", 'data["top_k"] = kwargs["top_k"]')


def patch_anomaly_mask(igpo: Path) -> None:
    path = igpo / "treetune/trainers/utils.py"
    text = path.read_text(encoding="utf-8")
    marker = "    mask_bool = mask.to(dtype=torch.bool)\n"
    if marker in text:
        return
    start = text.find("def monitor_tensor_anomalies(")
    end = text.find("\ndef ", start + 1)
    if start < 0:
        fail(f"cannot apply anomaly mask compatibility; function missing: {path}")
    if end < 0:
        end = len(text)
    block = text[start:end]
    anchor = "    nan_mask = torch.isnan(tensor)\n"
    if anchor not in block:
        fail(f"cannot apply anomaly mask compatibility; source differs: {path}")
    block = block.replace(anchor, marker + anchor, 1)
    replacements = {
        "& mask]": "& mask_bool]",
        "masked_mean(tensor, mask), masked_var(tensor, mask,":
            "masked_mean(tensor, mask_bool), masked_var(tensor, mask_bool,",
        "(nan_mask * mask)": "(nan_mask & mask_bool)",
        "(inf_mask * mask)": "(inf_mask & mask_bool)",
        "(high_z_mask * mask)": "(high_z_mask & mask_bool)",
    }
    for old, new in replacements.items():
        if old not in block:
            fail(f"cannot apply anomaly mask compatibility ({old.strip()}); source differs: {path}")
        block = block.replace(old, new, 1)
    path.write_text(text[:start] + block + text[end:], encoding="utf-8")


def patch_empty_probability_mask(igpo: Path) -> None:
    path = igpo / "treetune/trainers/ppo_trainer.py"
    old = "            action_mask = action_mask & prob_mask"
    new = (
        "            prob_filtered_mask = action_mask & prob_mask\n"
        "            # Keep the loss finite if no response token passes the threshold.\n"
        "            action_mask = torch.where(\n"
        "                prob_filtered_mask.any(), prob_filtered_mask, action_mask\n"
        "            )"
    )
    replace_once(path, old, new, "empty PPO probability-mask fallback", "prob_filtered_mask = action_mask & prob_mask")


def append_line(path: Path, line: str) -> None:
    text = path.read_text(encoding="utf-8")
    if line in text.splitlines():
        return
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + line + "\n", encoding="utf-8")


def source_digest(repo: Path) -> str:
    roots = [
        repo / "vdra_core",
        repo / "integrations/igpo/treetune/inference_strategies/vdra_inference_strategy.py",
        repo / "integrations/igpo/treetune/episode_generators/resumable_hybrid_episode_generator.py",
        repo / "integrations/igpo/configs/vdra",
        repo / "integrations/igpo/configs/model_overrides/qwen3_1_7b.jsonnet",
        repo / "integrations/igpo/configs/qwen3",
        repo / "integrations/igpo/configs/polIter_qwen3_1_7b_spo_logged_MATH.jsonnet",
        repo / "integrations/igpo/configs/polIter_qwen3_1_7b_vdra_MATH.jsonnet",
    ]
    files: list[Path] = []
    for item in roots:
        if item.is_dir():
            files.extend(p for p in item.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        elif item.is_file():
            files.append(item)
        else:
            fail(f"required code/config is missing: {item}")
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in sorted(set(files)):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sync(igpo: Path, repo: Path) -> None:
    if not (igpo / "treetune").is_dir() or not (igpo / "vdra_core").is_dir():
        fail(f"expected the unpacked IGPO source tree at {igpo}")

    stamp = igpo / ".vdra_qwen3_setup.sha256"
    current_digest = source_digest(repo)
    required_config = igpo / "configs/polIter_qwen3_1_7b_vdra_MATH.jsonnet"
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == current_digest:
        try:
            top_k_source = (igpo / "guidance/llms/_openai_vllm.py").read_text(encoding="utf-8")
            anomaly_source = (igpo / "treetune/trainers/utils.py").read_text(encoding="utf-8")
            ppo_source = (igpo / "treetune/trainers/ppo_trainer.py").read_text(encoding="utf-8")
        except OSError:
            pass
        else:
            if (
                required_config.is_file()
                and 'top_k = kwargs.pop("top_k", None)' in top_k_source
                and "mask_bool = mask.to(dtype=torch.bool)" in anomaly_source
                and "prob_filtered_mask = action_mask & prob_mask" in ppo_source
            ):
                print("[qwen3] bundled IGPO source is already prepared")
                return

    copy_tree(repo / "vdra_core", igpo / "vdra_core")
    copy_file(
        repo / "integrations/igpo/treetune/inference_strategies/vdra_inference_strategy.py",
        igpo / "treetune/inference_strategies/vdra_inference_strategy.py",
    )
    append_line(
        igpo / "treetune/inference_strategies/__init__.py",
        "from .vdra_inference_strategy import VDRAInferenceStrategy, SPOLoggedInferenceStrategy",
    )
    copy_file(
        repo / "integrations/igpo/treetune/episode_generators/resumable_hybrid_episode_generator.py",
        igpo / "treetune/episode_generators/resumable_hybrid_episode_generator.py",
    )
    append_line(
        igpo / "treetune/episode_generators/__init__.py",
        "from .resumable_hybrid_episode_generator import ResumableHybridEpisodeGenerator",
    )

    for source in (repo / "integrations/igpo/configs/vdra").glob("*.jsonnet"):
        copy_file(source, igpo / "configs/vdra" / source.name)
    for relative in (
        "configs/polIter_deepseekR1Qwen_vdra_MATH.jsonnet",
        "configs/polIter_deepseekR1Qwen_spo_logged_MATH.jsonnet",
        "configs/model_overrides/qwen3_1_7b.jsonnet",
        "configs/polIter_qwen3_1_7b_spo_logged_MATH.jsonnet",
        "configs/polIter_qwen3_1_7b_vdra_MATH.jsonnet",
    ):
        copy_file(repo / "integrations/igpo" / relative, igpo / relative)
    copy_tree(repo / "integrations/igpo/configs/qwen3", igpo / "configs/qwen3")

    patch_qwen3_top_k(igpo)
    patch_anomaly_mask(igpo)
    patch_empty_probability_mask(igpo)
    stamp.write_text(current_digest + "\n", encoding="utf-8")
    print("[qwen3] offline IGPO source, Qwen3 configs, and compatibility fixes are ready")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        fail("usage: qwen3_prepare_offline_igpo.py IGPO_DIR VDRA_HOME")
    sync(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
