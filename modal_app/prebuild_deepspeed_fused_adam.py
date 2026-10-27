"""Validate PyTorch headers and compile DeepSpeed FusedAdam in the image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/opt/torch_extensions")
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
os.environ.setdefault("MAX_JOBS", "4")
os.environ["DS_ACCELERATOR"] = "cuda"

import torch


header = Path(torch.__file__).resolve().parent / "include/ATen/ops/randn_like_native.h"
raw = header.read_bytes()
try:
    text = raw.decode("utf-8")
except UnicodeDecodeError as exc:
    raise RuntimeError(f"PyTorch header is not valid UTF-8: {header}") from exc
if "#pragma once" not in text[:4096] or "randn_like" not in text[:4096]:
    raise RuntimeError(f"PyTorch header contents are invalid: {header}")
if any(byte == 0 or (byte < 32 and byte not in (9, 10, 13)) for byte in raw):
    raise RuntimeError(f"PyTorch header contains binary/control bytes: {header}")

print(
    "TORCH_HEADER_CHECK_JSON="
    + json.dumps(
        {
            "torch_version": torch.__version__,
            "path": str(header),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        sort_keys=True,
    ),
    flush=True,
)

from deepspeed.ops.op_builder import FusedAdamBuilder

module = FusedAdamBuilder().load()
if not hasattr(module, "multi_tensor_adam"):
    raise RuntimeError("DeepSpeed loaded a non-CUDA FusedAdam module")
print(
    f"DEEPSPEED_FUSED_ADAM_PREBUILT={module.__name__} "
    "accelerator=cuda TORCH_CUDA_ARCH_LIST=10.0",
    flush=True,
)
