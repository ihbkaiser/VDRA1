#!/usr/bin/env python3
"""Report how far a run has progressed, from its resumable checkpoints.

  python analysis/run_status.py <run_dir> [--framework treetune|verl|openrlhf] [--json]

Prints the last policy update that can be resumed from (0 if none).  With
--json, also the stop target and launch parameters of the last launch, the
seconds since the last heartbeat, and a status word:
  done        last resumable update >= stop
  running     heartbeat within --alive_s seconds
  incomplete  not done and not running (resume it)
  new         no checkpoint and never launched
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Dict, Optional


def detect_framework(run_dir: str) -> Optional[str]:
    if glob.glob(os.path.join(run_dir, "checkpoints", "ckpt--iter_*")):
        return "treetune"
    if os.path.isdir(os.path.join(run_dir, "ckpts")):
        return "verl"
    if os.path.isdir(os.path.join(run_dir, "ckpt")):
        return "openrlhf"
    meta = last_launch(run_dir)
    return (meta or {}).get("framework")


def resumable_update(run_dir: str, framework: Optional[str]) -> int:
    fw = framework or detect_framework(run_dir)
    if fw == "treetune":
        best = 0
        for d in glob.glob(os.path.join(run_dir, "checkpoints", "ckpt--iter_*")):
            # treetune always keeps optimizer state for its newest checkpoint and
            # resumes from it, so the newest iteration is the resumable one
            m = re.search(r"ckpt--iter_([0-9.]+)", d)
            if m:
                best = max(best, int(float(m.group(1))))
        return best
    if fw == "verl":
        p = os.path.join(run_dir, "ckpts", "latest_checkpointed_iteration.txt")
        if os.path.exists(p):
            try:
                return int(open(p).read().strip())
            except ValueError:
                return 0
        return 0
    if fw == "openrlhf":
        p = os.path.join(run_dir, "ckpt", "_actor_resume", "latest")
        if os.path.exists(p):
            m = re.search(r"(\d+)", open(p).read())
            return int(m.group(1)) if m else 0
        return 0
    return 0


def last_launch(run_dir: str) -> Optional[Dict]:
    p = os.path.join(run_dir, "run_meta.jsonl")
    if not os.path.exists(p):
        return None
    last = None
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            last = r
    return last


def last_heartbeat(run_dir: str) -> Optional[float]:
    p = os.path.join(run_dir, "run_meta.jsonl")
    if not os.path.exists(p):
        return None
    t, open_launch = None, False
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") in ("start", "beat"):
            t, open_launch = float(r["time"]), True
        elif r.get("event") == "end":
            open_launch = False
    return t if open_launch else None


def status(run_dir: str, framework: Optional[str] = None, alive_s: float = 600.0) -> Dict:
    meta = last_launch(run_dir) or {}
    fw = framework or meta.get("framework") or detect_framework(run_dir)
    upd = resumable_update(run_dir, fw)
    stop = int(meta.get("stop", 0) or 0)
    beat = last_heartbeat(run_dir)
    age = (time.time() - beat) if beat is not None else None
    if stop and upd >= stop:
        word = "done"
    elif age is not None and age < alive_s:
        word = "running"
    elif upd > 0 or meta:
        word = "incomplete"
    else:
        word = "new"
    return {"run_dir": run_dir, "name": os.path.basename(os.path.normpath(run_dir)), "framework": fw,
            "resumable_update": upd, "stop": stop, "status": word, "heartbeat_age_s": age, "last_launch": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--framework", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--alive_s", type=float, default=600.0)
    a = ap.parse_args()
    if a.json:
        print(json.dumps(status(a.run_dir, a.framework, a.alive_s)))
    else:
        print(resumable_update(a.run_dir, a.framework))


if __name__ == "__main__":
    main()
