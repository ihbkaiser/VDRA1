#!/usr/bin/env bash
set -euo pipefail

# Run from the extracted vdra-new root, or pass that root as the first arg.
ROOT="${1:-$PWD}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
SOURCE="${ROOT}/work/TreeRL/openrlhf/trainer/ppo_utils/experience_maker.py"
MANIFEST="${ROOT}/SHA256SUMS"
SCRIPT_FILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"

if [[ ! -f "${SOURCE}" || ! -f "${MANIFEST}" ]]; then
  echo "Expected TreeRL experience_maker.py and SHA256SUMS under ${ROOT}" >&2
  exit 2
fi

python3 - "${ROOT}" "${SOURCE}" "${MANIFEST}" "${SCRIPT_FILE}" <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile

root, source_path, manifest_path, script_path = map(Path, sys.argv[1:])
source = source_path.read_bytes()
old = b"batch_size_multiplier = len(sequences) // forward_batch_size"
new = b"batch_size_multiplier = (len(sequences) + forward_batch_size - 1) // forward_batch_size"
old_count = source.count(old)
new_count = source.count(new)
if old_count == 2 and new_count == 0:
    source = source.replace(old, new)
    print("Patched both TreeRL MCTS chunk loops to include a final partial batch.")
elif old_count == 0 and new_count == 2:
    print("TreeRL MCTS remainder patch is already present.")
else:
    raise SystemExit(
        f"Expected both MCTS chunk loops to be uniformly patched; "
        f"found old={old_count}, patched={new_count}. Left files unchanged."
    )

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
    source_path.relative_to(root).as_posix(): hashlib.sha256(source).hexdigest(),
}
try:
    script_relative = script_path.relative_to(root).as_posix()
except ValueError:
    script_relative = None
if script_relative and script_path.is_file():
    updates[script_relative] = hashlib.sha256(script_path.read_bytes()).hexdigest()

lines = manifest_path.read_text(encoding="utf-8").splitlines()
for relative, digest in updates.items():
    indices = [i for i, line in enumerate(lines) if line.split("  ", 1)[-1] == relative]
    if len(indices) > 1:
        raise SystemExit(f"Duplicate SHA256SUMS entries for {relative}; left manifest unchanged.")
    entry = f"{digest}  {relative}"
    if indices:
        lines[indices[0]] = entry
    else:
        lines.append(entry)
lines.sort(key=lambda line: line.split("  ", 1)[-1])
manifest_bytes = ("\n".join(lines) + "\n").encode("utf-8")

atomic_write(source_path, source)
atomic_write(manifest_path, manifest_bytes)
print("Updated SHA256SUMS:")
for relative in updates:
    print(f"  {relative}")
PY
