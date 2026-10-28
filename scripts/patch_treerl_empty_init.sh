#!/usr/bin/env bash
set -euo pipefail

# Run from the extracted VDRA payload root, or pass that root as the first arg.
ROOT="${1:-$PWD}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
SCRIPT_FILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
ACTOR="${ROOT}/work/TreeRL/openrlhf/models/actor.py"
MANIFEST="${ROOT}/SHA256SUMS"

if [[ ! -f "${ACTOR}" ]]; then
  echo "TreeRL actor file not found: ${ACTOR}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Payload checksum manifest not found: ${MANIFEST}" >&2
  exit 2
fi

python3 - "${ROOT}" "${ACTOR}" "${MANIFEST}" "${SCRIPT_FILE}" <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

root = Path(sys.argv[1])
actor_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
script_path = Path(sys.argv[4])
source = actor_path.read_bytes()

# Remove the unsupported Transformers 4.51 keyword only from the model loader
# call where it follows torch_dtype="auto".
pattern = re.compile(
    rb'(?m)^([ \t]*torch_dtype="auto",\r?\n)[ \t]*empty_init=False,[ \t]*\r?\n'
)
matches = list(pattern.finditer(source))
if len(matches) > 1:
    raise SystemExit(f"Expected at most one empty_init argument, found {len(matches)}")
if matches:
    source = pattern.sub(rb"\1", source, count=1)
    print("Removed unsupported empty_init=False from TreeRL actor loader.")
elif b"empty_init=False," in source:
    raise SystemExit("Found empty_init=False in actor.py, but not in the expected loader context; left file unchanged.")
else:
    print("TreeRL actor loader is already patched.")

updates = {actor_path.relative_to(root).as_posix(): hashlib.sha256(source).hexdigest()}
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
        raise SystemExit(f"Duplicate checksum entries for {relative}; left manifest unchanged.")
    new_entry = f"{digest}  {relative}"
    if indices:
        manifest_lines[indices[0]] = new_entry
    else:
        manifest_lines.append(new_entry)
manifest_lines.sort(key=lambda line: line.split("  ", 1)[-1])
manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")

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

atomic_write(actor_path, source)
atomic_write(manifest_path, manifest_bytes)
print(f"Updated {manifest_path} entries for: {', '.join(updates)}")
PY
