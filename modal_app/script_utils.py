"""Small helpers for running checked-out shell scripts inside Linux containers."""

from pathlib import Path

EVALUATION_DATASETS = ("aime24", "aime25", "amc23", "olympiadbench_hf")


def normalize_shell_script(path: Path) -> None:
    """Convert Windows CRLF newlines to LF before invoking a script with Bash."""
    contents = path.read_bytes()
    normalized = contents.replace(b"\r\n", b"\n")
    if normalized != contents:
        path.write_bytes(normalized)


def eval_dataset_download_args(python: str, script: str, data_dir: str) -> list[str]:
    """Build an explicit downloader invocation, avoiding argparse's empty-choice edge case."""
    return [python, script, *EVALUATION_DATASETS, "--data-dir", data_dir]
