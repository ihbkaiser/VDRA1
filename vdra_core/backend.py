"""Synchronous, token-id based proxy estimation for batched rollout engines.

Used by the TreePO (verl) and TreeRL (OpenRLHF) integrations and by the
node-bank builder.  All generation goes through one `generate_fn`, so the same
code works with:

  * an in-process `vllm.LLM`                      -> `from_vllm_llm(llm)`
  * verl's `self.inference_engine` (LLM wrapper)  -> `from_vllm_llm(engine)`
  * a Ray actor wrapping vLLM (TreeRL)            -> `from_ray_actor(actor)`

`generate_fn(list_of_prompt_token_ids, SamplingParams) -> list[RequestOutput]`.

Scoring uses vLLM `prompt_logprobs=0`, which returns, for every prompt
position, the log-probability of the actual prompt token under the raw model
(temperature is not applied).  Note that pilots and support samples are drawn
at the training temperature; see README "Known approximations".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import VDRAConfig
from .proxy import Pilot, ProxyResult, build_proxy, constant_proxy

GenerateFn = Callable[[List[List[int]], object], list]


def from_vllm_llm(llm) -> GenerateFn:
    """Adapter for `vllm.LLM` (and verl's inference_engine, which wraps it)."""

    def _gen(prompt_ids: List[List[int]], sp):
        try:
            return llm.generate(
                prompts=[{"prompt_token_ids": ids} for ids in prompt_ids],
                sampling_params=sp,
                use_tqdm=False,
            )
        except TypeError:
            # older vLLM signature
            return llm.generate(prompt_token_ids=prompt_ids, sampling_params=sp, use_tqdm=False)

    return _gen


def from_ray_actor(actor) -> GenerateFn:
    import ray  # noqa: WPS433 (optional dependency)

    def _gen(prompt_ids: List[List[int]], sp):
        return ray.get(actor.generate.remote(prompt_token_ids=prompt_ids, sampling_params=sp))

    return _gen


def _lp_value(x) -> Optional[float]:
    if x is None:
        return None
    return float(getattr(x, "logprob", x))


@dataclass
class SampleOut:
    token_ids: List[int]
    finished: bool
    finish_reason: str
    sum_logprob: Optional[float]
    num_tokens: int


@dataclass
class TokenBackend:
    generate_fn: GenerateFn
    temperature: float = 0.6
    top_p: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    max_model_len: int = 2048
    # count of generated / scored tokens, by category (for the cost tables)
    counters: Dict[str, int] = field(default_factory=dict)

    def _sp(self, **kw):
        from vllm import SamplingParams  # imported lazily so tests run without vLLM

        return SamplingParams(**kw)

    def _bump(self, key: str, n: int) -> None:
        self.counters[key] = self.counters.get(key, 0) + int(n)

    def sample(
        self,
        prefixes: Sequence[List[int]],
        n: int,
        max_tokens: int,
        category: str,
        want_logprobs: bool = True,
    ) -> List[List[SampleOut]]:
        """Draw n continuations of up to max_tokens for every prefix.
        Prefixes too long for the context get an empty list."""
        idx, prompts, caps = [], [], []
        for k, p in enumerate(prefixes):
            cap = min(int(max_tokens), self.max_model_len - len(p) - 1)
            if cap >= 1:
                idx.append(k)
                prompts.append(list(p))
                caps.append(cap)
        out: List[List[SampleOut]] = [[] for _ in prefixes]
        if not prompts:
            return out
        # vLLM takes one SamplingParams per call, so group prompts by cap.
        groups: Dict[int, List[int]] = {}
        for j, c in enumerate(caps):
            groups.setdefault(c, []).append(j)
        for cap, js in groups.items():
            sp = self._sp(
                n=int(n), temperature=self.temperature, top_p=self.top_p,
                max_tokens=int(cap), stop_token_ids=self.stop_token_ids,
                logprobs=0 if want_logprobs else None,
            )
            res = self.generate_fn([prompts[j] for j in js], sp)
            for j, ro in zip(js, res):
                samples = []
                for c in ro.outputs:
                    lps = None
                    if want_logprobs and getattr(c, "logprobs", None):
                        vals = []
                        for tok, d in zip(c.token_ids, c.logprobs):
                            v = d.get(tok) if isinstance(d, dict) else None
                            vals.append(_lp_value(v) if v is not None else 0.0)
                        lps = float(sum(vals))
                    reason = str(getattr(c, "finish_reason", "") or "")
                    samples.append(SampleOut(
                        token_ids=list(c.token_ids),
                        finished=(reason != "length"),
                        finish_reason=reason,
                        sum_logprob=lps,
                        num_tokens=len(c.token_ids),
                    ))
                    self._bump(f"gen_{category}", len(c.token_ids))
                out[idx[j]] = samples
        return out

    def score(self, pairs: Sequence[Tuple[List[int], List[int]]]) -> List[float]:
        """Sum log p(cont | prefix) for each (prefix, cont) pair."""
        if not pairs:
            return []
        sp = self._sp(n=1, temperature=1.0, max_tokens=1, prompt_logprobs=0)
        prompts = [list(p) + list(c) for p, c in pairs]
        keep = [k for k, pr in enumerate(prompts) if len(pr) < self.max_model_len]
        res = self.generate_fn([prompts[k] for k in keep], sp) if keep else []
        vals = [float("-inf")] * len(pairs)
        for k, ro in zip(keep, res):
            plen = len(pairs[k][0])
            plp = ro.prompt_logprobs or []
            tot = 0.0
            for pos in range(plen, len(prompts[k])):
                d = plp[pos] if pos < len(plp) else None
                tok = prompts[k][pos]
                v = d.get(tok) if isinstance(d, dict) else None
                tot += _lp_value(v) if v is not None else float("-inf")
            vals[k] = tot
            self._bump("scored_prompt", len(prompts[k]))
            self._bump("scored_cont", len(prompts[k]) - plen)
        return vals


def estimate_batch(
    prefixes: Sequence[List[int]],
    backend: TokenBackend,
    cfg: VDRAConfig,
    terminal_reward_fn: Optional[Callable[[int, List[int]], Optional[float]]] = None,
) -> Tuple[List[ProxyResult], List[List[Pilot]]]:
    """Proxy for a batch of eligible nodes, batching all engine calls.

    terminal_reward_fn(node_index, pilot_token_ids) -> reward, called for pilots
    that finished inside `pilot_tokens` (needed for terminal_pair_mode="answer").
    Returns (proxy results, pilots per node).
    """
    if cfg.proxy == "constant":
        return [constant_proxy(cfg.k0) for _ in prefixes], [[] for _ in prefixes]

    t0 = time.time()
    before = dict(backend.counters)
    pilot_samples = backend.sample(prefixes, cfg.k0, cfg.pilot_tokens, category="pilot")
    pilots: List[List[Pilot]] = []
    for v, samples in enumerate(pilot_samples):
        ps = []
        for i, s in enumerate(samples):
            rew = None
            if s.finished and terminal_reward_fn is not None:
                rew = terminal_reward_fn(v, s.token_ids)
            ps.append(Pilot(index=i, token_ids=s.token_ids, finished=s.finished, reward=rew,
                            sum_logprob=s.sum_logprob, num_tokens=s.num_tokens))
        pilots.append(ps)

    # support samples for every continuable pilot
    sup_prefix, sup_key = [], []
    for v, ps in enumerate(pilots):
        for p in ps:
            if not p.finished:
                sup_prefix.append(list(prefixes[v]) + p.token_ids)
                sup_key.append((v, p.index))
    sup = backend.sample(sup_prefix, cfg.r, cfg.m, category="support", want_logprobs=False)
    support: Dict[int, List[Tuple[int, List[int]]]] = {}   # node -> [(owner pilot, z)]
    for (v, i), zs in zip(sup_key, sup):
        for z in zs:
            support.setdefault(v, []).append((i, z.token_ids))

    # cross-scoring: every continuable pilot's prefix scores every support sample of the node
    pairs, keys = [], []
    for v, ps in enumerate(pilots):
        cont = [p for p in ps if not p.finished]
        for p in cont:
            pre = list(prefixes[v]) + p.token_ids
            for s, (_, z) in enumerate(support.get(v, [])):
                pairs.append((pre, z))
                keys.append((v, p.index, s))
    vals = backend.score(pairs)
    scores: Dict[int, Dict[Tuple[int, int], float]] = {}
    for (v, i, s), val in zip(keys, vals):
        scores.setdefault(v, {})[(i, s)] = val

    elapsed = time.time() - t0
    after = backend.counters
    batch_tokens = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    n_nodes = max(len(prefixes), 1)
    results = []
    for v, ps in enumerate(pilots):
        owners = [i for (i, _) in support.get(v, [])]
        per_node = {k: int(round(x / n_nodes)) for k, x in batch_tokens.items()}
        per_node["seconds_x1000"] = int(1000 * elapsed / n_nodes)
        results.append(build_proxy(ps, owners, scores.get(v, {}), cfg, tokens=per_node))
    return results, pilots
