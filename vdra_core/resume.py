"""Resume bookkeeping shared by the training integrations.

`ResumePlan` turns a saved client state into what a training loop needs:
how many prompt micro-batches to skip (the data order is deterministic given
the sampler seed and epoch, so skipping reproduces the uninterrupted order),
which policy update to continue from, and when to stop.  `capture_rng` /
`restore_rng` save and restore python, numpy and torch RNG streams.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional


def env_int(name: str, default: int = 0) -> int:
    v = os.environ.get(name, "")
    try:
        return int(v) if v.strip() else default
    except ValueError:
        return default


@dataclass
class ResumePlan:
    start_update: int = 0           # policy updates already completed
    skip_micro_batches: int = 0     # prompt micro-batches already consumed
    stop_update: int = 0            # 0 = no stop
    resume_every: int = 0           # save a resumable checkpoint every k updates (0 = off)
    keep_every: int = 0             # save evaluation (HF) weights every k updates

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]], *, stop_update: int = 0,
                   resume_every: int = 0, keep_every: int = 0) -> "ResumePlan":
        state = state or {}
        return cls(
            start_update=int(state.get("update_step", 0) or 0),
            skip_micro_batches=int(state.get("consumed_micro_batches", 0) or 0),
            stop_update=int(stop_update or 0),
            resume_every=int(resume_every or 0),
            keep_every=int(keep_every or 0),
        )

    @classmethod
    def from_env(cls, state: Optional[Dict[str, Any]], keep_every: int = 0) -> "ResumePlan":
        return cls.from_state(
            state,
            stop_update=env_int("VDRA_STOP_UPDATE"),
            resume_every=env_int("VDRA_RESUME_EVERY"),
            keep_every=keep_every,
        )

    @property
    def already_done(self) -> bool:
        return bool(self.stop_update) and self.start_update >= self.stop_update

    def is_stop(self, update: int) -> bool:
        return bool(self.stop_update) and update >= self.stop_update

    def save_eval_weights(self, update: int) -> bool:
        """HF weights for evaluation: every keep_every updates and at the stop."""
        return (self.keep_every > 0 and update % self.keep_every == 0) or self.is_stop(update)

    def save_resume_state(self, update: int) -> bool:
        """Full resumable state: every resume_every, every eval save, and at the stop."""
        return ((self.resume_every > 0 and update % self.resume_every == 0)
                or self.save_eval_weights(update))


def capture_rng() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        out["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        import torch

        out["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            out["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return out


def restore_rng(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    try:
        import numpy as np

        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        pass
