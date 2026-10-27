"""Append-only JSONL logger used by every integration (one row per node / queue)."""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Dict, Optional


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


class JsonlLogger:
    def __init__(self, path: Optional[str]):
        self.path = path or ""
        self._lock = threading.Lock()
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def write(self, row: Dict[str, Any]) -> None:
        if not self.path:
            return
        line = json.dumps(_clean(row), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
