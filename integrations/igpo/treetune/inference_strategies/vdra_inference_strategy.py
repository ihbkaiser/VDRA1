"""VDRA tree construction for treetune, implemented as described in the paper.

Registered as inference strategy type "vdra".  Drop-in replacement for SPO's
"hybrid" strategy: the episode generator (hybrid_episode_generator), edge
extraction, replay buffer, advantage and PPO loss are untouched, so R1 (SPO)
and R2 (VDRA) differ only in how many children each non-final node receives.

Per eligible node v (depth < max_depth - 1), following Alg. 1:
  1. k0 pilots of <= pilot_tokens tokens are sampled from v;
  2. every continuable pilot gets r support samples of <= m tokens;
  3. every continuable pilot's prefix scores every support sample of v
     (vLLM /completions with echo=True, one request per pair);
  4. pairwise tanh-TV, tail correction, C_v, redundancy cap k_v;
  5. (v, C_v, k_v) is submitted to a cross-prompt queue (32 x 16, 1 s timeout);
  6. on dispatch, terminal pilots are reused (counted toward n_v) and the
     remaining n_v - t_v children are sampled fresh with SPO's expander.
Nodes at the final expansion depth use SPO's uniform branch factor.

Two extra strategy types are registered:
  * "spo_logged": SPO's hybrid strategy plus the same per-node JSONL logging,
                  for budget audits of the baseline (no algorithmic change).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from treetune import logging_utils
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.inference_strategies.hybrid_inference_strategy import HybridInferenceStrategy
from treetune.inference_strategies.tree_inference import Node

from vdra_core import (
    AsyncQueueManager,
    JsonlLogger,
    Pilot,
    QueueItem,
    VDRAConfig,
    build_proxy,
    constant_proxy,
    incoming_weight,
)

logger = logging_utils.get_logger(__name__)


def _derived_seed(base: Optional[int], *parts: Any) -> Optional[int]:
    if base is None:
        return None
    h = hashlib.sha256(("|".join(str(p) for p in (base,) + parts)).encode()).hexdigest()
    return int(h[:8], 16)


def _num_tokens(node: Dict[str, Any], tokenize) -> int:
    n = node.get("num_tokens")
    if n is not None:
        return int(n)
    try:
        return len(tokenize(node.get("text", "")))
    except Exception:
        return 0


class _TreeLoggingMixin:
    """Per-node JSONL rows shared by SPO (logged) and VDRA."""

    def _log_paths(self) -> Dict[str, str]:
        base = Path(self.result_dir)
        return {
            "nodes": str(base / "vdra_nodes.jsonl"),
            "queues": str(base / "vdra_queues.jsonl"),
            "trees": str(base / "vdra_trees.jsonl"),
        }

    def generate(self, dataset):
        """Rotate logs left by an interrupted attempt of this iteration, so a
        resumed iteration does not double-count nodes, queues or trees."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for path in self._log_paths().values():
            if os.path.exists(path) and os.path.getsize(path) > 0:
                os.replace(path, f"{path}.interrupted-{stamp}")
        self._loggers = None
        return super().generate(dataset)

    def _get_loggers(self):
        if getattr(self, "_loggers", None) is None:
            p = self._log_paths()
            self._loggers = {k: JsonlLogger(v) for k, v in p.items()}
        return self._loggers

    def _tokenize_fn(self):
        tok = getattr(self, "tokenizer", None) or getattr(self.node_expander, "tokenizer", None)
        if tok is None:
            return lambda text: text.split()
        # treetune tokenizers are HuggingFace PreTrainedTokenizer subclasses
        return lambda text: tok(text, add_special_tokens=False).input_ids

    def _log_tree_summary(self, tree: Dict[str, Any], problem_id: Any, method: str) -> None:
        tokenize = self._tokenize_fn()
        main_tokens = 0
        n_edges = 0
        by_depth: Dict[int, List[int]] = {}

        def walk(node):
            nonlocal main_tokens, n_edges
            ch = node.get("children") or []
            if ch:
                by_depth.setdefault(int(node.get("depth", 0)), []).append(len(ch))
            for c in ch:
                n_edges += 1
                if not c.get("_vdra_reused_pilot"):
                    main_tokens += _num_tokens(c, tokenize)
                walk(c)

        walk(tree)
        self._get_loggers()["trees"].write({
            "method": method,
            "problem_id": problem_id,
            "main_rollout_tokens": main_tokens,
            "n_edges": n_edges,
            "children_per_node_by_depth": {str(k): v for k, v in by_depth.items()},
            "tree_construction_seconds": tree.get("tree_construction_seconds"),
            "vdra_aux_tokens": tree.get("vdra_aux_tokens"),
            "vdra_proxy_seconds": tree.get("vdra_proxy_seconds"),
            "vdra_queue_wait_seconds": tree.get("vdra_queue_wait_seconds"),
            "wall_time": time.time(),
        })


@InferenceStrategy.register("spo_logged", exist_ok=True)
class SPOLoggedInferenceStrategy(_TreeLoggingMixin, HybridInferenceStrategy):
    """SPO's hybrid strategy, unchanged, plus tree-level token logging."""

    async def _construct_tree(self, initial_prompt, max_depth=2, data_instance=None):
        tree = await super()._construct_tree(initial_prompt, max_depth, data_instance=data_instance)
        pid = (data_instance or {}).get("_treetune__idx")
        try:
            self._log_tree_summary(tree, pid, "spo")
        except Exception as exc:  # logging must not break training
            logger.warning("spo_logged: could not log tree: %r", exc)
        return tree


@InferenceStrategy.register("vdra", exist_ok=True)
class VDRAInferenceStrategy(_TreeLoggingMixin, HybridInferenceStrategy):
    def __init__(
        self,
        vdra: Optional[Dict[str, Any]] = None,
        vdra_score_concurrency: int = 64,
        vdra_score_timeout_seconds: float = 120.0,
        vdra_score_retry_attempts: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = VDRAConfig.from_dict(vdra)
        self.score_concurrency = int(vdra_score_concurrency)
        self.score_timeout = float(vdra_score_timeout_seconds)
        self.score_retries = int(vdra_score_retry_attempts)
        self._lp_client = None
        self._qm: Optional[AsyncQueueManager] = None
        self._loggers = None
        logger.info("VDRA config: %s", self.cfg.to_dict())

    # ------------------------------------------------------------------ helpers

    def _queue_manager(self) -> AsyncQueueManager:
        if self._qm is None:
            loggers = self._get_loggers()

            def on_dispatch(qid, items, alloc, budget, waited):
                loggers["queues"].write({
                    "queue_id": qid, "size": len(items), "budget": budget,
                    "allocated": int(sum(alloc)), "waited_s": waited,
                    "node_ids": [it.node_id for it in items],
                    "coefs": [it.coef for it in items],
                    "k_v": [it.k_v for it in items],
                    "capped": [it.capped for it in items],
                    "n_v": list(alloc),
                })

            self._qm = AsyncQueueManager(
                n_queues=self.cfg.n_queues,
                capacity=self.cfg.queue_capacity,
                timeout_s=self.cfg.queue_timeout_s,
                n_min=self.cfg.n_min,
                on_dispatch=on_dispatch,
            )
        return self._qm

    def _scorer(self):
        if self._lp_client is None:
            from treetune.ingpo.vllm_scorer import VLLMLogprobClient

            params = self.guidance_llm_lazy._params  # type: ignore[attr-defined]
            api_base = params.get("api_base")
            if api_base in (None, "none"):
                api_base = os.environ.get("APP_OPENAI_VLLM_API_BASE")
            if api_base is None:
                raise RuntimeError("VDRA needs the vLLM api_base for likelihood scoring")
            self._lp_client = VLLMLogprobClient(
                api_base=api_base,
                model=params.get("model"),
                api_key=params.get("api_key", "EMPTY"),
                timeout=self.score_timeout,
                max_concurrency=self.score_concurrency,
                retry_attempts=self.score_retries,
            )
        from treetune.ingpo.vllm_scorer import make_lp_scorer

        return make_lp_scorer(self._lp_client, self._tokenize_fn())

    async def _sample(self, prefix: str, depth: int, n: int, max_tokens: int, seed: Optional[int]):
        if n <= 0:
            return []
        return await self.node_expander._sample_node(
            prefix, depth, n, max_tokens=max_tokens, seed=seed
        )

    # ------------------------------------------------------------------ proxy

    async def _estimate(self, node: Node, depth: int, node_id: str, data_instance) -> tuple:
        cfg = self.cfg
        if cfg.proxy == "constant":
            return constant_proxy(cfg.k0), []
        t0 = time.time()
        prefix = node["full_text"]
        tokenize = self._tokenize_fn()
        raw = await self._sample(prefix, depth + 1, cfg.k0, cfg.pilot_tokens,
                                 _derived_seed(self.seed, node_id, "pilot"))
        pilots: List[Pilot] = []
        pilot_nodes: List[Dict[str, Any]] = []
        for i, pn in enumerate(raw):
            finished = pn.get("finish_reason") != "length"
            reward = None
            if finished:
                reward, _ = self.reward_function(
                    query=prefix, response=pn["full_text"], dataset_instance=data_instance
                )
            ids = tokenize(pn.get("text", ""))
            pilots.append(Pilot(index=i, token_ids=list(ids), text=pn.get("text", ""),
                                finished=finished, reward=reward,
                                sum_logprob=pn.get("sum_logprobs"),
                                num_tokens=int(pn.get("num_tokens") or len(ids))))
            pn["reward"] = reward
            pilot_nodes.append(pn)
        gen_pilot = sum(p.num_tokens for p in pilots)

        cont = [(p, pn) for p, pn in zip(pilots, pilot_nodes) if not p.finished]
        sup_lists = await asyncio.gather(*[
            self._sample(pn["full_text"], depth + 2, cfg.r, cfg.m,
                         _derived_seed(self.seed, node_id, "support", p.index))
            for p, pn in cont
        ])
        support: List[tuple] = []          # (owner index, text)
        gen_support = 0
        for (p, _), zs in zip(cont, sup_lists):
            for z in zs:
                support.append((p.index, z.get("text", "")))
                gen_support += _num_tokens(z, tokenize)

        scorer = self._scorer()
        jobs, keys = [], []
        for p, pn in cont:
            for s, (_, ztext) in enumerate(support):
                if not ztext:
                    continue
                jobs.append(scorer.score_one(pn["full_text"], ztext))
                keys.append((p.index, s))
        vals = await asyncio.gather(*jobs) if jobs else []
        scores = {k: float(v) for k, v in zip(keys, vals)}
        owners = [o for o, _ in support]
        tokens = {"gen_pilot": gen_pilot, "gen_support": gen_support, "score_requests": len(jobs)}
        res = build_proxy(pilots, owners, scores, cfg, tokens=tokens)
        res.tokens["seconds_x1000"] = int(1000 * (time.time() - t0))
        return res, pilot_nodes

    # ------------------------------------------------------------------ tree

    async def _construct_tree(self, initial_prompt: str, max_depth: int = 2,
                              data_instance: Optional[Dict[str, Any]] = None):
        t0_tree = time.time()
        cfg = self.cfg
        qm = self._queue_manager()
        loggers = self._get_loggers()
        problem_id = (data_instance or {}).get("_treetune__idx")
        aux = {"gen_pilot": 0, "gen_support": 0, "score_requests": 0}
        timing = {"proxy": 0.0, "wait": 0.0}

        tree = {
            "text": initial_prompt, "depth": 0, "full_text": initial_prompt,
            "stop_text": "aaa", "_request_object": data_instance, "leaf": False,
            "vdra_node_id": f"{problem_id}/r",
        }

        def grade(prefix: str, child: Dict[str, Any]) -> None:
            child["reward"], _ = self.reward_function(
                query=prefix, response=child["full_text"], dataset_instance=data_instance
            )
            child["leaf"] = True

        async def dfs(node: Node, prefix: str, depth: int, n_parent: Optional[int]) -> None:
            if depth == max_depth:
                node["reward"], _ = self.reward_function(
                    query=prefix, response=node["text"], dataset_instance=data_instance
                )
                node["leaf"] = True
                return
            node_id = node["vdra_node_id"]
            nominal = int(self.node_expander.branch_factor_strategy(node))
            final_depth = depth == max_depth - 1
            max_tokens = None if final_depth else self.M

            reused: List[Dict[str, Any]] = []
            if final_depth:
                n_v = nominal
                children = await self.node_expander.expand(
                    current_node=node, prefix=prefix, depth=depth, max_tokens=max_tokens,
                )
            else:
                t_p = time.time()
                res, pilot_nodes = await self._estimate(node, depth, node_id, data_instance)
                timing["proxy"] += time.time() - t_p
                for k in aux:
                    aux[k] += int(res.tokens.get(k, 0))
                lam = incoming_weight(n_parent) if cfg.weighting == "incoming" else 1.0
                coef = lam * res.coef
                if cfg.proxy == "constant":
                    # n proportional to sqrt(C) at the optimum, so C = nominal^2
                    # returns exactly the nominal branching (null test).
                    coef = float(nominal * nominal)
                item = QueueItem(node_id=node_id, coef=coef, k_v=res.k_v,
                                 nominal=nominal, capped=res.capped)
                t_w = time.time()
                n_v = await qm.submit(item)
                waited = time.time() - t_w
                timing["wait"] += waited

                if cfg.reuse_terminal_pilots and pilot_nodes:
                    terminal = [pn for pn in pilot_nodes if pn.get("finish_reason") != "length"]
                    rng = random.Random(_derived_seed(self.seed if self.seed is not None else 0, node_id, "reuse"))
                    rng.shuffle(terminal)
                    for pn in terminal[: max(n_v, 0)]:
                        c = dict(pn)
                        c["depth"] = depth + 1
                        c["leaf"] = True
                        c["_vdra_reused_pilot"] = True
                        reused.append(c)
                n_fresh = max(n_v - len(reused), 0)
                fresh = []
                if n_fresh > 0:
                    fresh = await self.node_expander.expand(
                        current_node=node, prefix=prefix, depth=depth,
                        max_tokens=max_tokens, branch_factor=n_fresh,
                    )
                children = reused + list(fresh)
                loggers["nodes"].write({
                    "problem_id": problem_id, "node_id": node_id, "depth": depth,
                    "nominal": nominal, "n_v": n_v, "lambda": lam,
                    "n_parent": n_parent, "n_reused_terminal_pilots": len(reused),
                    "n_children_built": len(children), "queue_wait_s": waited,
                    **res.to_row(),
                })

            for idx, c in enumerate(children):
                c["vdra_node_id"] = f"{node_id}/{idx}"
            node["children"] = children

            tasks = []
            for c in children:
                if c.get("_vdra_reused_pilot"):
                    if c.get("reward") is None:
                        grade(prefix, c)
                    continue
                if c.get("finish_reason") != "length":
                    grade(prefix, c)
                else:
                    c["leaf"] = False
                    tasks.append(asyncio.create_task(dfs(c, c["full_text"], depth + 1, len(children))))
            if tasks:
                await asyncio.gather(*tasks)

            rewards = [c["reward"] for c in children if c.get("reward") is not None]
            if rewards:
                node["reward"] = float(np.mean(rewards))
                node["reward_std"] = float(np.std(rewards))
            else:
                # n_v == 0 cannot happen with n_min >= 1; guard against a silent 0 value.
                logger.warning("VDRA node %s got no children; grading it as a leaf", node_id)
                node["reward"], _ = self.reward_function(
                    query=prefix, response=node["full_text"], dataset_instance=data_instance
                )
                node["reward_std"] = 0.0
                node["leaf"] = True

        await dfs(tree, initial_prompt, 0, None)
        tree["tree_construction_seconds"] = time.time() - t0_tree
        tree["vdra_aux_tokens"] = aux
        tree["vdra_proxy_seconds"] = timing["proxy"]
        tree["vdra_queue_wait_seconds"] = timing["wait"]
        try:
            self._log_tree_summary(tree, problem_id, "vdra")
        except Exception as exc:
            logger.warning("vdra: could not log tree: %r", exc)
        return tree
