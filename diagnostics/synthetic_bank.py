#!/usr/bin/env python3
"""Synthetic node banks with known child values, for testing Exp 1 / Exp 2.

Root values and child dispersions are heterogeneous across prompts, so an
informative proxy has something to find.  `proxy_noise` controls how well the
fake C_v tracks the true dispersion (0 = perfect ranking).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_io import save_bank  # noqa: E402


def make_bank(n_roots=64, pool=32, refs=32, expanded=8, p_finish=0.08, proxy_noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    roots = []

    def child(p):
        if rng.random() < p_finish:
            r = float(rng.random() < p)
            return {"finished": True, "reward": r, "refs": [r] * refs, "p_true": r}
        return {"finished": False, "reward": None,
                "refs": [float(x) for x in (rng.random(refs) < p)], "p_true": float(p)}

    def fake_proxy(true_disp):
        c = true_disp * float(np.exp(rng.normal(0.0, proxy_noise)))
        return {"C_v": c, "C_raw": 1.05 * c, "k_v": 6, "capped": False,
                "pilot_nll": float(rng.gamma(2.0, 0.3)), "pilot_diversity": float(rng.random()),
                "tokens_gen_pilot": 576, "tokens_gen_support": 1152, "tokens_scored_prompt": 50000}

    for i in range(n_roots):
        v_root = rng.beta(1.5, 1.5)
        s_root = rng.choice([0.02, 0.1, 0.3])          # heterogeneous dispersion
        p1 = np.clip(v_root + rng.normal(0, s_root, size=pool), 0.0, 1.0)
        root_pool = [child(p) for p in p1]
        exp = []
        cand = [j for j, c in enumerate(root_pool) if not c["finished"]]
        rng.shuffle(cand)
        for j in sorted(cand[:expanded]):
            s_v = rng.choice([0.01, 0.08, 0.25])
            p2 = np.clip(root_pool[j]["p_true"] + rng.normal(0, s_v, size=pool), 0.0, 1.0)
            pool2 = [child(p) for p in p2]
            disp = float(np.var([c["p_true"] for c in pool2], ddof=1))
            exp.append({"pool_index": j, "proxy": fake_proxy(disp), "pool": pool2})
        root_disp = float(np.var([c["p_true"] for c in root_pool], ddof=1))
        roots.append({"prompt_id": str(i), "level": None, "proxy": fake_proxy(root_disp),
                      "pool": root_pool, "expanded": exp})
    return roots


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy_noise", type=float, default=0.5)
    a = ap.parse_args()
    save_bank(a.out, {"label": f"synthetic_s{a.seed}", "synthetic": True}, make_bank(seed=a.seed, proxy_noise=a.proxy_noise))
    print(a.out)
