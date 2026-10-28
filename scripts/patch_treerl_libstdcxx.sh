#!/usr/bin/env bash
set -euo pipefail

# Install the TreeRL user-space C++ runtime path into an extracted payload.
# Run from the payload root, or pass the payload root as the first argument.
ROOT="${1:-$PWD}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
TRAIN_SCRIPT="${ROOT}/project/scripts/20_train.sh"
MANIFEST="${ROOT}/SHA256SUMS"
SCRIPT_FILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
RUNTIME_DIR="${ROOT}/runtime/libstdcxx_gcc13/lib"

for library in libstdc++.so.6 libgcc_s.so.1; do
  if [[ ! -f "${RUNTIME_DIR}/${library}" ]]; then
    echo "Missing bundled runtime library: ${RUNTIME_DIR}/${library}" >&2
    exit 2
  fi
done
if [[ ! -f "${TRAIN_SCRIPT}" || ! -f "${MANIFEST}" ]]; then
  echo "Expected project/scripts/20_train.sh and SHA256SUMS under ${ROOT}" >&2
  exit 2
fi

python3 - "${ROOT}" "${TRAIN_SCRIPT}" "${MANIFEST}" "${SCRIPT_FILE}" <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile

root, train_path, manifest_path, script_path = map(Path, sys.argv[1:])
raw = train_path.read_bytes()
newline = b"\r\n" if b"\r\n" in raw else b"\n"
needle = b'  use_env "${CONDA_TREERL}"' + newline
block_lines = [
    b'  local gcc_compat_dir="${VDRA_HOME}/../runtime/libstdcxx_gcc13/lib"',
    b'  if [[ -f "${gcc_compat_dir}/libstdc++.so.6" && -f "${gcc_compat_dir}/libgcc_s.so.1" ]]; then',
    b'    export LD_LIBRARY_PATH="${gcc_compat_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
    b'    log "TreeRL GCC 13 runtime enabled from ${gcc_compat_dir}"',
    b'  fi',
]
block = newline.join(block_lines) + newline
if b"local gcc_compat_dir=" in raw:
    print("TreeRL launcher already has the GCC runtime path.")
else:
    if raw.count(needle) != 1:
        raise SystemExit("Could not uniquely locate the TreeRL launcher setup; left it unchanged.")
    raw = raw.replace(needle, needle + block, 1)

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

updates: dict[str, str] = {}
for path in (
    train_path,
    root / "runtime/libstdcxx_gcc13/lib/libstdc++.so.6",
    root / "runtime/libstdcxx_gcc13/lib/libgcc_s.so.1",
    root / "runtime/libstdcxx_gcc13/SOURCE.txt",
    root / "runtime/libstdcxx_gcc13/licenses/RUNTIME.LIBRARY.EXCEPTION",
    script_path,
):
    if path.is_file():
        updates[path.relative_to(root).as_posix()] = hashlib.sha256(
            raw if path == train_path else path.read_bytes()
        ).hexdigest()

manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
for relative, digest in updates.items():
    indices = [i for i, line in enumerate(manifest_lines) if line.split("  ", 1)[-1] == relative]
    if len(indices) > 1:
        raise SystemExit(f"Duplicate SHA256SUMS entries for {relative}; manifest left unchanged.")
    entry = f"{digest}  {relative}"
    if indices:
        manifest_lines[indices[0]] = entry
    else:
        manifest_lines.append(entry)
manifest_lines.sort(key=lambda line: line.split("  ", 1)[-1])
manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")

atomic_write(train_path, raw)
atomic_write(manifest_path, manifest_bytes)
print("Updated TreeRL launcher and SHA256SUMS:")
for relative in updates:
    print(f"  {relative}")
PY
