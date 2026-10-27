"""Configuration for VDRA (Value Dispersion-guided Rollout Allocation).

Defaults follow Table 4 of the Technical Supplement.  Every field is a plain
Python value so the same object can be built from jsonnet (treetune), Hydra
(verl / TreePO) or argparse (OpenRLHF / TreeRL).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict


@dataclass
class VDRAConfig:
    # --- dispersion proxy (Supp. Sec. E, "VDRA dispersion estimation") ---
    k0: int = 6                  # pilot branches per eligible node
    pilot_tokens: int = 96       # max tokens per pilot
    r: int = 6                   # likelihood-support samples per continuable pilot
    m: int = 32                  # inspected horizon (tokens) of each support sample
    eps_tail: float = 0.01       # tail coefficient, Eq. (9)
    eps_trunc: float = 0.02      # redundancy threshold on corrected TV
    # How to compare two *terminal* pilots (paper does not specify; see README):
    #   "answer": D=0 if both terminal pilots got the same verifier reward, else 1
    #   "one":    D=1 for every pair that involves a terminal pilot
    terminal_pair_mode: str = "answer"

    # --- allocation (Supp. Sec. E, "Queue-wise rollout allocation") ---
    n_min: int = 1
    n_queues: int = 32
    queue_capacity: int = 16
    queue_timeout_s: float = 1.0
    # "equal": Eq. (7) as in the paper.  "incoming": multiply C_v by
    # lambda_v = (n_parent - 1) / n_parent (the edge-weighted variant from the
    # rebuttal derivation).  Roots always get lambda = 1.
    weighting: str = "equal"

    # --- proxy mode ---
    # "tv":       the paper's tail-corrected TV proxy (default)
    # "constant": C_v = 1, no pilots, no pruning  -> must reproduce the baseline
    #             exactly (null test)
    # "raw_tv":   TV without tail correction and without redundancy cap
    proxy: str = "tv"
    use_redundancy_cap: bool = True

    # --- child construction ---
    # Paper: terminal pilots that already produced a verifier-scored response
    # are reused and count toward n_v; non-terminal pilots are discarded.
    reuse_terminal_pilots: bool = True

    # --- logging ---
    log_dir: str = ""            # where per-node JSONL rows are written ("" = off)

    def validate(self) -> "VDRAConfig":
        assert self.k0 >= 2, "k0 must be >= 2 to form pilot pairs"
        assert self.r >= 1 and self.m >= 1 and self.pilot_tokens >= 1
        assert 0.0 <= self.eps_tail <= 1.0
        assert 0.0 <= self.eps_trunc <= 1.0
        assert self.n_min >= 0
        assert self.n_queues >= 1 and self.queue_capacity >= 1
        assert self.weighting in {"equal", "incoming"}, self.weighting
        assert self.proxy in {"tv", "constant", "raw_tv"}, self.proxy
        assert self.terminal_pair_mode in {"answer", "one"}, self.terminal_pair_mode
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None) -> "VDRAConfig":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown VDRA config keys: {sorted(unknown)}")
        return cls(**d).validate()
