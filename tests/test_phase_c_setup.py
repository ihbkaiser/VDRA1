from pathlib import Path

from modal_app.images import _CHECKOUT_IGNORE
from modal_app.phase_c import MODEL_REVISION, _existing_data_inputs_present
from modal_app.script_utils import eval_dataset_download_args, normalize_shell_script


def test_normalize_shell_script_converts_crlf_to_lf(tmp_path: Path):
    script = tmp_path / "download.sh"
    script.write_bytes(b"#!/bin/bash\r\nurl=https://example.test/data.zip\r\n")

    normalize_shell_script(script)

    assert script.read_bytes() == b"#!/bin/bash\nurl=https://example.test/data.zip\n"


def test_eval_dataset_download_command_selects_all_supported_sets():
    assert eval_dataset_download_args("python", "download_eval_datasets.py", "/work/source_data") == [
        "python",
        "download_eval_datasets.py",
        "aime24",
        "aime25",
        "amc23",
        "olympiadbench_hf",
        "--data-dir",
        "/work/source_data",
    ]


def test_modal_checkout_upload_ignores_mutable_git_fetch_head_files():
    assert "**/.git/FETCH_HEAD" in _CHECKOUT_IGNORE


def test_existing_exports_are_registered_only_when_the_complete_cache_is_present(tmp_path: Path):
    data_dir = tmp_path / "data_vdra"
    for name in ("train.parquet", "train_treerl.jsonl", "split_manifest.json", "diag256.jsonl"):
        path = data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached\n", encoding="utf-8")
    (data_dir / "math-nodiag" / "dataset_dict.json").parent.mkdir(parents=True)
    (data_dir / "math-nodiag" / "dataset_dict.json").write_text("{}", encoding="utf-8")
    for split in ("train", "validation", "test"):
        split_dir = data_dir / "math-nodiag" / split
        split_dir.mkdir(parents=True)
        (split_dir / "state.json").write_text("{}", encoding="utf-8")
        (split_dir / "data-00000-of-00001.arrow").write_bytes(b"arrow")

    source_root = tmp_path / "source_data"
    math = source_root / "math"
    math.mkdir(parents=True)
    (math / "dataset_dict.json").write_text("{}", encoding="utf-8")
    for name in ("aime24", "aime25", "amc23", "olympiadbench_hf"):
        dataset = source_root / name
        dataset.mkdir()
        (dataset / "dataset_dict.json").write_text("{}", encoding="utf-8")
        (dataset / "_source.json").write_text("{}", encoding="utf-8")

    model = (
        tmp_path
        / "hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots"
        / MODEL_REVISION
    )
    model.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json"):
        (model / name).write_bytes(b"cached")

    assert _existing_data_inputs_present(tmp_path)
    (data_dir / "train.parquet").unlink()
    assert not _existing_data_inputs_present(tmp_path)
