"""Value-dispersion proxy (Supp. Sec. B-E, Alg. 1 lines 8-15).

All functions here are pure: they take already-generated pilots, support
samples and cross log-likelihoods, and return distances, the coefficient C_v
and the redundancy-pruned pilot count k_v.  The I/O (sampling and scoring)
lives in `backend.py` (token-id, synchronous: vLLM engine, verl, OpenRLHF) and
in the treetune strategy (text, asyncio, vLLM HTTP server).

TV estimator.  For two continuation distributions P_i, P_j with a pooled
support Z = Z_i U Z_j (r samples drawn from each, i.e. from the mixture
(P_i + P_j)/2), the identity

    TV(P_i, P_j) = E_{z ~ (P_i+P_j)/2} | (p_i - p_j) / (p_i + p_j) |
                 = E_{z ~ (P_i+P_j)/2} | tanh( (log p_i(z) - log p_j(z)) / 2 ) |

gives the likelihood-ratio tanh estimator with the 1/2 TV convention built in.
It is unbiased for the inspected-horizon distance D_m and always in [0, 1].
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import VDRAConfig


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Pilot:
    """A short pilot continuation of an eligible node."""

    index: int
    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    finished: bool = False            # reached EOS / answer within pilot_tokens
    reward: Optional[float] = None    # verifier reward when finished
    sum_logprob: Optional[float] = None
    num_tokens: int = 0


@dataclass
class ProxyResult:
    coef: float                       # C_v (Alg. 1 line 13)
    k_v: int                          # pilots retained after redundancy pruning
    capped: bool                      # k_v < k0
    n_pilots: int
    n_continuable: int
    d_raw: Dict[Tuple[int, int], float]        # \hat D_m per pilot pair
    d_corr: Dict[Tuple[int, int], float]       # \tilde D_m per pilot pair
    raw_coef: float                   # C_v without tail correction (ablation)
    retained: List[int]               # indices of retained pilots
    # cheap heuristics computed from the same pilots (used by Exp 1)
    pilot_nll: float = float("nan")   # mean per-token NLL of sampled pilot tokens
    pilot_diversity: float = float("nan")  # 1 - mean pairwise trigram Jaccard
    tokens: Dict[str, int] = field(default_factory=dict)  # generated / scored token counts

    def to_row(self) -> Dict[str, object]:
        return {
            "C_v": self.coef,
            "C_raw": self.raw_coef,
            "k_v": self.k_v,
            "capped": self.capped,
            "n_pilots": self.n_pilots,
            "n_continuable": self.n_continuable,
            "pilot_nll": self.pilot_nll,
            "pilot_diversity": self.pilot_diversity,
            "mean_d_raw": _mean(self.d_raw.values()),
            "mean_d_corr": _mean(self.d_corr.values()),
            **{f"tokens_{k}": v for k, v in self.tokens.items()},
        }


def _mean(xs) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def tanh_tv(logp_i: Sequence[float], logp_j: Sequence[float]) -> float:
    """Mean |tanh((log p_i - log p_j)/2)| over pooled support samples."""
    if len(logp_i) != len(logp_j):
        raise ValueError("score vectors must align")
    if not logp_i:
        return 1.0
    acc = 0.0
    cnt = 0
    for a, b in zip(logp_i, logp_j):
        if a is None or b is None:
            continue
        if math.isinf(a) and math.isinf(b):
            # both impossible under the two prefixes: no information
            continue
        if math.isinf(a) or math.isinf(b):
            acc += 1.0
        else:
            acc += abs(math.tanh(0.5 * (a - b)))
        cnt += 1
    return acc / cnt if cnt else 1.0


def tail_correct(d: float, eps_tail: float) -> float:
    """Eq. (9): \\tilde D = \\hat D + (1 - \\hat D) eps_tail."""
    d = min(max(float(d), 0.0), 1.0)
    return d + (1.0 - d) * float(eps_tail)


def pairwise_distances(
    pilots: Sequence[Pilot],
    support_owner: Sequence[int],
    scores: Dict[Tuple[int, int], float],
    cfg: VDRAConfig,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Compute \\hat D and \\tilde D for every pilot pair.

    support_owner[s] = index of the pilot that generated support sample s.
    scores[(i, s)]   = log p(z_s | prefix + pilot_i), for continuable pilots i.
    Terminal pilots have no support (their continuation is a point mass on EOS).
    """
    by_idx = {p.index: p for p in pilots}
    support_of: Dict[int, List[int]] = {}
    for s, owner in enumerate(support_owner):
        support_of.setdefault(owner, []).append(s)

    d_raw: Dict[Tuple[int, int], float] = {}
    d_corr: Dict[Tuple[int, int], float] = {}
    idxs = sorted(by_idx)
    for i, j in itertools.combinations(idxs, 2):
        pi, pj = by_idx[i], by_idx[j]
        if pi.finished or pj.finished:
            # Exact distances; no tail correction is needed.
            if pi.finished and pj.finished and cfg.terminal_pair_mode == "answer":
                same = (pi.reward is not None and pj.reward is not None
                        and float(pi.reward) == float(pj.reward))
                d = 0.0 if same else 1.0
            else:
                d = 1.0
            d_raw[(i, j)] = d
            d_corr[(i, j)] = d
            continue
        pooled = support_of.get(i, []) + support_of.get(j, [])
        li = [scores.get((i, s)) for s in pooled]
        lj = [scores.get((j, s)) for s in pooled]
        d = tanh_tv(li, lj)
        d_raw[(i, j)] = d
        d_corr[(i, j)] = tail_correct(d, cfg.eps_tail)
    return d_raw, d_corr


# ---------------------------------------------------------------------------
# Coefficient and redundancy
# ---------------------------------------------------------------------------


def dispersion_coefficient(d: Dict[Tuple[int, int], float], k0: int) -> float:
    """Alg. 1 line 13: C_v = [k0 (k0-1)]^{-1} sum_{i<j} D_ij^2."""
    if k0 < 2:
        return 0.0
    return float(sum(v * v for v in d.values()) / (k0 * (k0 - 1)))


def redundancy_prune(
    indices: Sequence[int], d_corr: Dict[Tuple[int, int], float], eps_trunc: float
) -> List[int]:
    """Alg. 1 lines 14-15: build the graph with edges where \\tilde D <= eps_trunc
    and repeatedly remove a maximum-degree pilot (ties: highest index) until no
    edge remains.  Returns the retained pilot indices."""
    alive = set(indices)
    edges = {(i, j) for (i, j), v in d_corr.items() if v <= eps_trunc and i in alive and j in alive}
    while edges:
        deg: Dict[int, int] = {}
        for i, j in edges:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        victim = max(deg, key=lambda x: (deg[x], x))
        alive.discard(victim)
        edges = {(i, j) for (i, j) in edges if i != victim and j != victim}
    return sorted(alive)


# ---------------------------------------------------------------------------
# Cheap heuristics for Exp 1 (computed from pilots only)
# ---------------------------------------------------------------------------


def pilot_nll(pilots: Sequence[Pilot]) -> float:
    tot, n = 0.0, 0
    for p in pilots:
        if p.sum_logprob is not None and p.num_tokens > 0:
            tot += -float(p.sum_logprob)
            n += int(p.num_tokens)
    return tot / n if n else float("nan")


def _trigrams(ids: Sequence[int]) -> set:
    return {tuple(ids[k:k + 3]) for k in range(max(len(ids) - 2, 0))}


def pilot_diversity(pilots: Sequence[Pilot]) -> float:
    grams = [_trigrams(p.token_ids) for p in pilots]
    sims = []
    for a, b in itertools.combinations(grams, 2):
        if not a and not b:
            sims.append(1.0)
            continue
        sims.append(len(a & b) / max(len(a | b), 1))
    return 1.0 - (sum(sims) / len(sims)) if sims else float("nan")


# ---------------------------------------------------------------------------
# One-call assembly
# ---------------------------------------------------------------------------


def build_proxy(
    pilots: Sequence[Pilot],
    support_owner: Sequence[int],
    scores: Dict[Tuple[int, int], float],
    cfg: VDRAConfig,
    tokens: Optional[Dict[str, int]] = None,
) -> ProxyResult:
    k0 = len(pilots)
    d_raw, d_corr = pairwise_distances(pilots, support_owner, scores, cfg)
    coef = dispersion_coefficient(d_corr, max(k0, 2))
    raw = dispersion_coefficient(d_raw, max(k0, 2))
    idxs = [p.index for p in pilots]
    if cfg.use_redundancy_cap and cfg.proxy == "tv":
        retained = redundancy_prune(idxs, d_corr, cfg.eps_trunc)
    else:
        retained = list(idxs)
    k_v = len(retained)
    if cfg.proxy == "raw_tv":
        coef = raw
    return ProxyResult(
        coef=coef,
        k_v=k_v,
        capped=k_v < k0,
        n_pilots=k0,
        n_continuable=sum(1 for p in pilots if not p.finished),
        d_raw=d_raw,
        d_corr=d_corr,
        raw_coef=raw,
        retained=retained,
        pilot_nll=pilot_nll(pilots),
        pilot_diversity=pilot_diversity(pilots),
        tokens=dict(tokens or {}),
    )


def constant_proxy(k0: int) -> ProxyResult:
    """Null-test proxy: C_v = 1, no pruning."""
    return ProxyResult(
        coef=1.0, k_v=k0, capped=False, n_pilots=0, n_continuable=0,
        d_raw={}, d_corr={}, raw_coef=1.0, retained=list(range(k0)), tokens={},
    )
