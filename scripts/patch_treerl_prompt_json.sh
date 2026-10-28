#!/usr/bin/env bash
set -euo pipefail

# Run from the extracted vdra-new root, or pass that root as the first argument.
ROOT="${1:-$PWD}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
PROMPTS="${ROOT}/work/TreeRL/openrlhf/datasets/prompts_dataset.py"
MANIFEST="${ROOT}/SHA256SUMS"
SCRIPT_FILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"

if [[ ! -f "${PROMPTS}" ]]; then
  echo "TreeRL prompt dataset file not found: ${PROMPTS}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "SHA256SUMS not found under ${ROOT}" >&2
  exit 2
fi

python3 - "${ROOT}" "${PROMPTS}" "${MANIFEST}" "${SCRIPT_FILE}" <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

root, prompt_path, manifest_path, script_path = map(Path, sys.argv[1:])
source = prompt_path.read_bytes()
newline = b"\r\n" if b"\r\n" in source else b"\n"
marker = b"# Always serialize TreeRL prompt records for MCTS rollouts."
if marker not in source:
    pattern = re.compile(
        rb"(?ms)^    def __getitem__\(self, idx\):\r?\n"
        rb".*?^        else:\r?\n            return self\.prompts\[idx\](?:\r?\n|$)"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one model-dependent PromptDataset.__getitem__ block; found {len(matches)}. "
            "Left files unchanged."
        )
    method = (
        b'    def __getitem__(self, idx):\n'
        b'        print("current model", self.current_model)\n'
        b'        # Always serialize TreeRL prompt records for MCTS rollouts.\n'
        b'        # The model can be loaded from a generic local path such as /model/model.\n'
        b'        history = self.history[idx]\n'
        b'        if history is None:\n'
        b'            history = []\n'
        b'\n'
        b'        output = history + [{"prompt": self.prompts[idx]}]\n'
        b'        label = self.labels[idx]\n'
        b'        if label is not None:\n'
        b'            output[-1][self.strategy.args.label_key] = label\n'
        b'\n'
        b'        source = self.sources[idx]\n'
        b'        if source is not None:\n'
        b'            output[-1][self.strategy.args.source_key] = source\n'
        b'\n'
        b'        return json.dumps(output)\n'
    ).replace(b"\n", newline)
    source = pattern.sub(method, source, count=1)
    print("Patched PromptDataset to return the JSON record expected by TreeRL rollout sampling.")
else:
    print("TreeRL PromptDataset JSON serialization patch is already present.")

def atomic_write(path: Path, data: bytes) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

updates = {
    prompt_path.relative_to(root).as_posix(): hashlib.sha256(source).hexdigest(),
}
try:
    script_relative = script_path.relative_to(root).as_posix()
except ValueError:
    script_relative = None
if script_relative and script_path.is_file():
    updates[script_relative] = hashlib.sha256(script_path.read_bytes()).hexdigest()

manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
for relative, digest in updates.items():
    indices = [i for i, line in enumerate(manifest_lines) if line.split("  ", 1)[-1] == relative]
    if len(indices) > 1:
        raise SystemExit(f"Duplicate SHA256SUMS entries for {relative}; left manifest unchanged.")
    entry = f"{digest}  {relative}"
    if indices:
        manifest_lines[indices[0]] = entry
    else:
        manifest_lines.append(entry)
manifest_lines.sort(key=lambda line: line.split("  ", 1)[-1])
manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")

atomic_write(prompt_path, source)
atomic_write(manifest_path, manifest_bytes)
print("Updated SHA256SUMS:")
for relative in updates:
    print(f"  {relative}")
PY
