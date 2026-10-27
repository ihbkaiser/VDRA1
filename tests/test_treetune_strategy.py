"""End-to-end test of the treetune VDRA tree builder with stubbed treetune
modules and a fake LLM.  Checks queue budgets across concurrently built trees,
terminal-pilot reuse, the constant-proxy null test and value backup."""

import asyncio
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "integrations/igpo/treetune/inference_strategies/vdra_inference_strategy.py"


def _install_stubs():
    if "treetune" in sys.modules and getattr(sys.modules["treetune"], "_vdra_stub", False):
        return

    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    tt = mod("treetune")
    tt._vdra_stub = True
    lu = mod("treetune.logging_utils")
    import logging

    lu.get_logger = logging.getLogger
    tt.logging_utils = lu
    mod("treetune.inference_strategies")
    base = mod("treetune.inference_strategies.base_inference_strategy")

    class InferenceStrategy:
        registry = {}

        @classmethod
        def register(cls, name, exist_ok=False):
            def deco(klass):
                cls.registry[name] = klass
                return klass
            return deco

    base.InferenceStrategy = InferenceStrategy
    hyb = mod("treetune.inference_strategies.hybrid_inference_strategy")

    class HybridInferenceStrategy:
        def __init__(self, node_expander, reward_function, result_dir, seed=None, M=600,
                     tokenizer=None, guidance_llm=None, **kw):
            self.node_expander = node_expander
            self.reward_function = reward_function
            self.result_dir = result_dir
            self.seed = seed
            self.M = M
            self.tokenizer = tokenizer
            self.guidance_llm_lazy = guidance_llm

    hyb.HybridInferenceStrategy = HybridInferenceStrategy
    ti = mod("treetune.inference_strategies.tree_inference")
    ti.Node = dict


def _load_strategy():
    _install_stubs()
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("vdra_strategy_under_test", STRATEGY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeExpander:
    """Each sampled segment is a list of pseudo-tokens; a segment 'finishes'
    with probability p_finish; the answer token decides the reward."""

    tokenizer = None

    def __init__(self, rng, p_finish=0.25):
        self.rng = rng
        self.p_finish = p_finish
        self.calls = []

    def branch_factor_strategy(self, node):
        return 6

    async def _sample_node(self, prefix, depth, n, max_tokens=None, seed=None):
        self.calls.append((depth, n, max_tokens))
        out = []
        for _ in range(n):
            await asyncio.sleep(0)
            L = 4 if max_tokens is None else min(4, max_tokens)
            toks = [self.rng.choice("abcdxy") for _ in range(L)]
            finished = max_tokens is None or self.rng.random() < self.p_finish
            text = " " + " ".join(toks) + (" ANS=" + self.rng.choice("01") if finished else "")
            out.append({"text": text, "full_text": prefix + text, "depth": depth,
                        "finish_reason": "stop" if finished else "length",
                        "num_tokens": L, "sum_logprobs": -1.5 * L})
        return out

    async def expand(self, current_node, prefix, depth, max_tokens=None, branch_factor=None):
        n = branch_factor if branch_factor is not None else self.branch_factor_strategy(current_node)
        return await self._sample_node(prefix, depth + 1, n, max_tokens=max_tokens)


class FakeScorer:
    def __init__(self, rng):
        self.rng = rng

    async def score_one(self, prefix, cont):
        # deterministic pseudo-likelihood depending on the last prefix token and the continuation
        h = hash((prefix[-6:], cont)) % 1000
        return -1.0 - h / 250.0


def reward_function(query, response, dataset_instance):
    return (1.0 if response.rstrip().endswith("ANS=1") else 0.0), False


def _build(mod, tmp_path, proxy, n_trees=8, seed=0):
    rng = random.Random(seed)
    exp = FakeExpander(rng)
    strat = mod.VDRAInferenceStrategy(
        vdra={"proxy": proxy, "queue_timeout_s": 0.05, "n_queues": 4, "queue_capacity": 16},
        node_expander=exp, reward_function=reward_function, result_dir=tmp_path, seed=7,
        M=600, tokenizer=None,
    )
    strat._tokenize_fn = lambda: (lambda text: text.split())
    strat._scorer = lambda: FakeScorer(rng)

    async def main():
        return await asyncio.gather(*[
            strat._construct_tree(f"Q{i}:", 3, data_instance={"_treetune__idx": i}) for i in range(n_trees)
        ])

    return asyncio.run(main()), exp


def _walk(node, depth=0, out=None):
    out = out if out is not None else []
    out.append((depth, node))
    for c in node.get("children", []) or []:
        _walk(c, depth + 1, out)
    return out


@pytest.fixture(scope="module")
def mod():
    return _load_strategy()


def test_constant_proxy_is_spo(mod, tmp_path):
    trees, exp = _build(mod, tmp_path, "constant")
    for t in trees:
        for d, node in _walk(t):
            ch = node.get("children")
            if ch:
                assert len(ch) == 6, "null test: every expanded node must get the nominal 6"
    # no pilot/support sampling happened
    assert all(n in (6,) for (_, n, _) in exp.calls)


def test_tv_proxy_respects_queue_budgets_and_backs_up_values(mod, tmp_path):
    trees, _ = _build(mod, tmp_path, "tv")
    queues = [json.loads(l) for l in open(tmp_path / "vdra_queues.jsonl")]
    assert queues, "no queue was dispatched"
    for q in queues:
        assert q["allocated"] == min(q["budget"], q["allocated"]) == q["budget"] or any(q["capped"])
        assert all(n >= 1 for n in q["n_v"])
    nodes = {r["node_id"]: r for r in map(json.loads, open(tmp_path / "vdra_nodes.jsonl"))}
    for t in trees:
        for d, node in _walk(t):
            ch = node.get("children") or []
            if not ch:
                assert node.get("leaf") and node.get("reward") in (0.0, 1.0)
                continue
            mean = sum(c["reward"] for c in ch) / len(ch)
            assert node["reward"] == pytest.approx(mean)
            if d < 2:
                row = nodes[node["vdra_node_id"]]
                assert len(ch) == row["n_v"]
                assert row["n_reused_terminal_pilots"] <= row["n_v"]
            else:
                assert len(ch) == 6          # final depth is uniform
    trees_rows = [json.loads(l) for l in open(tmp_path / "vdra_trees.jsonl")]
    assert len(trees_rows) == len(trees)
    assert all(r["vdra_aux_tokens"]["gen_pilot"] > 0 for r in trees_rows)


def test_queues_mix_prompts(mod, tmp_path):
    _build(mod, tmp_path, "tv", n_trees=12)
    queues = [json.loads(l) for l in open(tmp_path / "vdra_queues.jsonl")]
    mixed = [q for q in queues if len({nid.split("/")[0] for nid in q["node_ids"]}) > 1]
    assert mixed, "cross-prompt queueing never happened"
