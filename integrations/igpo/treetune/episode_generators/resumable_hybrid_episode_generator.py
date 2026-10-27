"""SPO's hybrid episode generator with a replay buffer that survives restarts.

SPO's OnPolicyReplayBuffer keeps unused tree edges for up to 8 iterations, in
memory only.  After a crash, treetune resumes the policy and optimizer but
would restart with an empty buffer, so the first iterations after a resume
would train on a different edge mix than an uninterrupted run.  This
generator persists the buffer after every iteration (one pickle per rank,
newest 3 kept) and reloads it when it resumes.  Registered as
"hybrid_episode_generator_resumable"; nothing else changes.
"""

from __future__ import annotations

import glob
import os
import pickle
import re
from pathlib import Path

from treetune.episode_generators.base_episode_generator import EpisodeGenerator
from treetune.episode_generators.hybrid_episode_generator import (
    HybridEpisodeGenerator,
    OnPolicyReplayBuffer,
)
from treetune.logging_utils import get_logger

logger = get_logger(__name__)


class PersistentOnPolicyReplayBuffer(OnPolicyReplayBuffer):
    def __init__(self, directory: Path, rank: int, keep: int = 3):
        super().__init__()
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.keep = int(keep)
        self._restored = False

    def _files(self):
        out = []
        for p in glob.glob(str(self.dir / f"after_iter_*_rank{self.rank}.pkl")):
            m = re.search(r"after_iter_(\d+)_rank", p)
            if m:
                out.append((int(m.group(1)), p))
        return sorted(out)

    def add_edges(self, new_edges, iteration):
        if not self._restored and not self.edges:
            older = [(it, p) for it, p in self._files() if it < iteration]
            if older:
                it, path = older[-1]
                with open(path, "rb") as f:
                    self.edges = pickle.load(f)
                logger.info(f"[vdra-resume] restored {len(self.edges)} replay-buffer edges saved after iteration {it}")
        self._restored = True
        super().add_edges(new_edges, iteration)

    def get_edges(self, iteration):
        sampled, removed = super().get_edges(iteration)
        path = self.dir / f"after_iter_{iteration}_rank{self.rank}.pkl"
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.edges, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)                       # atomic: a crash never leaves half a file
        for _, old in self._files()[: -self.keep]:
            try:
                os.remove(old)
            except OSError:
                pass
        return sampled, removed


@EpisodeGenerator.register("hybrid_episode_generator_resumable")
class ResumableHybridEpisodeGenerator(HybridEpisodeGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.replay_buffer_type == "on_policy":
            rank = getattr(getattr(self, "distributed_state", None), "process_index", 0)
            self.replay_buffer = PersistentOnPolicyReplayBuffer(
                Path(self.exp_root) / "replay_buffer", rank=rank
            )
