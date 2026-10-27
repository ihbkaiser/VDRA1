#!/usr/bin/env python3
"""Build a frozen-policy node bank for Experiments 1 and 2 with offline vLLM.

Example (one checkpoint):
  python diagnostics/build_node_bank.py \
      --model experiments/vdra-s42/checkpoints/ckpt--iter_0100/hf_pretrained \
      --prompts data/diag256.jsonl --n_prompts 64 --seed 42 \
      --label R2_s42_step100 --out banks/R2_s42_step100.jsonl.gz

Per root: K=32 depth-1 children (segment tokens), each with R=32 completions;
E=8 non-terminal depth-1 children are "expanded" with their own K-pool of
depth-2 children, each with R completions.  The VDRA proxy and the cheap
heuristics are computed for every root and every expanded node with
vdra_core, so all Exp 1 / Exp 2 methods see exactly the same nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bank_io import save_bank  # noqa: E402
from vdra_core import VDRAConfig  # noqa: E402
from vdra_core import verifier  # noqa: E402
from vdra_core.backend import TokenBackend, estimate_batch, from_vllm_llm  # noqa: E402

DEFAULT_TEMPLATE = (
    "<｜begin▁of▁sentence｜><｜User｜>Solve the following math problem efficiently and clearly.  "
    "The last line of your response should be of the following format: 'Therefore, the final answer "
    "is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final "
    "number or expression that solves the problem. Think step by step before answering.\n\n"
    "{query}<｜Assistant｜><think>\n"
)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompts", required=True, help="jsonl with problem, answer, (level), (id)")
    ap.add_argument("--n_prompts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--refs", type=int, default=32)
    ap.add_argument("--expanded", type=int, default=8)
    ap.add_argument("--segment", type=int, default=600)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--template", default=None, help="file containing the question template with {query}")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=512, help="prompts per vLLM call")
    ap.add_argument("--vdra_config", default="{}", help="JSON overrides for VDRAConfig")
    return ap.parse_args()


def load_prompts(path: str, n: int, seed: int) -> List[Dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = random.Random(seed)
    # stratified by level when available, otherwise a plain random sample
    by_level: Dict[str, List[Dict]] = {}
    for r in rows:
        by_level.setdefault(str(r.get("level", "all")), []).append(r)
    for v in by_level.values():
        rng.shuffle(v)
    out: List[Dict] = []
    levels = sorted(by_level)
    while len(out) < min(n, len(rows)):
        for lv in levels:
            if by_level[lv] and len(out) < n:
                out.append(by_level[lv].pop())
    return out


def main():
    a = parse()
    from vllm import LLM
    from transformers import AutoTokenizer

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    template = open(a.template).read() if a.template else DEFAULT_TEMPLATE
    llm = LLM(model=a.model, tokenizer=a.tokenizer or a.model, max_model_len=a.max_model_len,
              enable_prefix_caching=True, gpu_memory_utilization=a.gpu_memory_utilization,
              tensor_parallel_size=a.tensor_parallel_size, seed=a.seed)
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else None
    backend = TokenBackend(generate_fn=from_vllm_llm(llm), temperature=a.temperature, top_p=a.top_p,
                           stop_token_ids=eos, max_model_len=a.max_model_len)
    cfg = VDRAConfig.from_dict(json.loads(a.vdra_config))
    rng = random.Random(a.seed)

    prompts = load_prompts(a.prompts, a.n_prompts, a.seed)
    root_ids = [tok(template.format(query=p["problem"]), add_special_tokens=False).input_ids for p in prompts]

    def chunks(xs, k):
        for i in range(0, len(xs), k):
            yield i, xs[i:i + k]

    def sample(prefixes, n, max_tokens, category):
        out = []
        for _, ch in chunks(prefixes, a.chunk):
            out.extend(backend.sample(ch, n, max_tokens, category=category, want_logprobs=False))
        return out

    def response_text(root_idx, ids_after_prompt):
        return tok.decode(ids_after_prompt, skip_special_tokens=True)

    def make_pool(prefix_list, owner_roots, base_lens):
        """Sample K segment children per prefix and R completions per non-terminal child."""
        seg = sample(prefix_list, a.pool, a.segment, "bank_segment")
        pools, comp_prefix, comp_key = [], [], []
        for k, (pre, children) in enumerate(zip(prefix_list, seg)):
            pool = []
            for j, c in enumerate(children):
                ids_after = pre[base_lens[k]:] + c.token_ids
                if c.finished:
                    r = verifier.score(response_text(owner_roots[k], ids_after), prompts[owner_roots[k]]["answer"],
                                       prompts[owner_roots[k]]["problem"])
                    pool.append({"finished": True, "reward": r, "refs": [r] * a.refs, "_ids": c.token_ids})
                else:
                    pool.append({"finished": False, "reward": None, "refs": [], "_ids": c.token_ids})
                    comp_prefix.append(pre + c.token_ids)
                    comp_key.append((k, j))
            pools.append(pool)
        comps = sample(comp_prefix, a.refs, a.max_model_len, "bank_completion")
        for (k, j), cp, outs in zip(comp_key, comp_prefix, comps):
            refs = []
            for o in outs:
                ids_after = cp[base_lens[k]:] + o.token_ids
                refs.append(verifier.score(response_text(owner_roots[k], ids_after), prompts[owner_roots[k]]["answer"],
                                           prompts[owner_roots[k]]["problem"]))
            while len(refs) < a.refs:            # context exhausted: score the truncated child as is
                refs.append(0.0)
            pools[k][j]["refs"] = refs
        return pools

    def term_reward_factory(prefix_list, owner_roots, base_lens):
        def fn(v, pilot_ids):
            ids_after = prefix_list[v][base_lens[v]:] + pilot_ids
            p = prompts[owner_roots[v]]
            return verifier.score(response_text(owner_roots[v], ids_after), p["answer"], p["problem"])
        return fn

    # depth-1 pools of the roots
    n = len(prompts)
    base = [len(r) for r in root_ids]
    root_pools = make_pool(root_ids, list(range(n)), base)
    root_proxy, _ = estimate_batch(root_ids, backend, cfg, term_reward_factory(root_ids, list(range(n)), base))

    # expanded depth-1 nodes and their depth-2 pools
    exp_prefix, exp_owner, exp_index = [], [], []
    for i, pool in enumerate(root_pools):
        cand = [j for j, c in enumerate(pool) if not c["finished"]]
        rng.shuffle(cand)
        for j in sorted(cand[: a.expanded]):
            exp_prefix.append(root_ids[i] + pool[j]["_ids"])
            exp_owner.append(i)
            exp_index.append(j)
    exp_base = [base[i] for i in exp_owner]
    exp_pools = make_pool(exp_prefix, exp_owner, exp_base)
    exp_proxy, _ = estimate_batch(exp_prefix, backend, cfg, term_reward_factory(exp_prefix, exp_owner, exp_base))

    roots = []
    for i, p in enumerate(prompts):
        for c in root_pools[i]:
            c.pop("_ids", None)
        roots.append({"prompt_id": str(p.get("id", p.get("unique_id", i))), "level": p.get("level"),
                      "proxy": root_proxy[i].to_row(), "pool": root_pools[i], "expanded": []})
    for k, (i, j) in enumerate(zip(exp_owner, exp_index)):
        for c in exp_pools[k]:
            c.pop("_ids", None)
        roots[i]["expanded"].append({"pool_index": j, "proxy": exp_proxy[k].to_row(), "pool": exp_pools[k]})

    header = {"label": a.label, "model": a.model, "seed": a.seed, "n_prompts": n, "pool": a.pool,
              "refs": a.refs, "expanded": a.expanded, "segment": a.segment, "max_model_len": a.max_model_len,
              "temperature": a.temperature, "vdra_config": cfg.to_dict(), "verifier": verifier.backend_name(),
              "counters": backend.counters, "seconds": time.time() - t_start}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_bank(a.out, header, roots)
    print(json.dumps({"out": a.out, "seconds": header["seconds"], "counters": backend.counters}))


if __name__ == "__main__":
    main()
