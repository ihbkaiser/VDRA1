#!/usr/bin/env python3
"""Cumulative GPU-hours at every checkpoint of a run, for every framework.

* treetune: sums `timing/iter/train_total_seconds` from training_timing.jsonl
  (episode generation + policy step, evaluation excluded), keeping the last
  record per iteration so resumed runs are not double counted.  Step k = after
  k iterations.
* verl / OpenRLHF: wall-clock inside the launch segments recorded by our
  launchers in run_meta.jsonl ({"event": "start"|"end", "time": ...}), up to
  the checkpoint directory's modification time.  Time between launches
  (queueing, crashes) is excluded.

  python analysis/gpu_hours.py --run_dir runs/R2_s42 --framework treetune --gpus 2 --out results/gpu_hours/R2_s42.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple


def treetune_hours(run_dir: str, gpus: int) -> Dict[int, float]:
    path = None
    for cand in glob.glob(os.path.join(run_dir, "**", "training_timing.jsonl"), recursive=True):
        path = cand
        break
    if path is None:
        raise FileNotFoundError(f"no training_timing.jsonl under {run_dir}")
    last: Dict[int, float] = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        last[int(r["iteration"])] = float(r.get("timing/iter/train_total_seconds") or 0.0)
    out, cum = {0: 0.0}, 0.0
    for it in sorted(last):
        cum += last[it]
        out[it + 1] = cum * gpus / 3600.0
    return out


def segments(run_dir: str) -> List[Tuple[float, float, bool]]:
    """Wall-clock launches as (start, end, clean_exit).

    A launch that died without an "end" record (node crash, OOM kill of the
    wrapper) is closed at its last heartbeat and marked unclean, so downtime
    before the next launch is not charged."""
    meta = os.path.join(run_dir, "run_meta.jsonl")
    segs, start, last = [], None, None
    for line in open(meta):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev, t = r.get("event"), float(r.get("time", 0.0))
        if ev == "start":
            if start is not None:
                segs.append((start, last if last is not None else start, False))
            start, last = t, t
        elif ev == "beat" and start is not None:
            last = t
        elif ev == "end" and start is not None:
            segs.append((start, t, int(r.get("rc", 0) or 0) == 0))
            start, last = None, None
    if start is not None:
        segs.append((start, float("inf"), True))     # still running
    return segs


def ckpt_hours(run_dir: str, gpus: int, pattern: str, step_re: str) -> Dict[int, float]:
    """GPU-hours at each checkpoint, counting only productive time.

    In a launch that ended uncleanly, the time after its last checkpoint was
    lost and is recomputed by the next launch; it is excluded so a crash does
    not inflate a run's compute."""
    segs = segments(run_dir)
    seen: Dict[int, float] = {}
    # checkpoint times recorded before old checkpoints were pruned (verl watcher)
    rec = os.path.join(run_dir, "ckpt_times.jsonl")
    if os.path.exists(rec):
        for line in open(rec):
            try:
                r = json.loads(line)
                seen.setdefault(int(r["step"]), float(r["mtime"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    for d in glob.glob(os.path.join(run_dir, pattern)):
        m = re.search(step_re, d)
        if m:
            seen.setdefault(int(m.group(1)), os.path.getmtime(d))
    ckpts = sorted(seen.items(), key=lambda x: x[1])
    productive = []
    for s, e, clean in segs:
        inside = [t for _, t in ckpts if s <= t <= e]
        if clean or not inside:
            productive.append((s, e if clean else (max(inside) if inside else s)))
        else:
            productive.append((s, max(inside)))
    out = {0: 0.0}
    for step, t in ckpts:
        secs = sum(max(0.0, min(t, e) - s) for s, e in productive if s < t)
        out[step] = secs * gpus / 3600.0
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--framework", choices=["treetune", "verl", "openrlhf"], required=True)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.framework == "treetune":
        h = treetune_hours(a.run_dir, a.gpus)
    elif a.framework == "verl":
        h = ckpt_hours(a.run_dir, a.gpus, "ckpts/global_step_*", r"global_step_(\d+)")
    else:
        h = ckpt_hours(a.run_dir, a.gpus, "ckpt/_actor_global_step*", r"global_step(\d+)$")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({str(k): v for k, v in h.items()}, open(a.out, "w"), indent=1)
    last = max(h)
    print(f"[gpu_hours] {a.run_dir}: step {last} -> {h[last]:.1f} GPU-h")


if __name__ == "__main__":
    main()
