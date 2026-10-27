import asyncio
import itertools
import math
import random

import pytest

from vdra_core import (
    AsyncQueueManager,
    Pilot,
    QueueItem,
    VDRAConfig,
    allocate_queue,
    build_proxy,
    dispersion_coefficient,
    incoming_weight,
    objective,
    redundancy_prune,
    solve_bounded,
    tail_correct,
    tanh_tv,
)


# ----------------------------------------------------------------------------- allocation

def brute_force(coefs, lower, upper, budget):
    target = min(budget, sum(upper))
    best, best_alloc = math.inf, None
    for alloc in itertools.product(*[range(l, u + 1) for l, u in zip(lower, upper)]):
        if sum(alloc) != target:
            continue
        val = objective(coefs, alloc)
        if val < best - 1e-12:
            best, best_alloc = val, alloc
    return best, best_alloc


@pytest.mark.parametrize("seed", range(40))
def test_greedy_is_optimal(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 5)
    coefs = [rng.choice([0.0, rng.random(), rng.random() * 5]) for _ in range(n)]
    lower = [1] * n
    upper = [rng.randint(1, 7) for _ in range(n)]
    budget = rng.randint(n, 6 * n)
    alloc = solve_bounded(coefs, lower, upper, budget)
    assert sum(alloc) == min(budget, sum(upper))
    assert all(l <= a <= u for a, l, u in zip(alloc, lower, upper))
    best, _ = brute_force(coefs, lower, upper, budget)
    assert objective(coefs, alloc) <= best + 1e-9


def test_equal_coefficients_give_uniform_split():
    items = [QueueItem(node_id=str(i), coef=0.37, k_v=6, nominal=6) for i in range(16)]
    assert allocate_queue(items, n_min=1) == [6] * 16


def test_constant_proxy_reproduces_nominal_branching_for_any_queue_size():
    for q in range(1, 17):
        items = [QueueItem(node_id=str(i), coef=1.0, k_v=6, nominal=6) for i in range(q)]
        assert allocate_queue(items, n_min=1) == [6] * q


def test_capped_node_residual_is_reassigned():
    items = [
        QueueItem("a", coef=10.0, k_v=2, nominal=6, capped=True),
        QueueItem("b", coef=1.0, k_v=6, nominal=6),
        QueueItem("c", coef=1.0, k_v=6, nominal=6),
    ]
    alloc = allocate_queue(items, n_min=1)
    assert alloc[0] == 2
    assert sum(alloc) == 18          # budget fully spent by the uncapped nodes
    assert alloc[1] == alloc[2] == 8


def test_neyman_proportionality_in_the_large_budget_limit():
    # optimum of sum C/n with sum n = B is n proportional to sqrt(C)
    coefs = [1.0, 4.0, 9.0]
    alloc = solve_bounded(coefs, [1, 1, 1], [10_000] * 3, 6000)
    ratios = [a / math.sqrt(c) for a, c in zip(alloc, coefs)]
    assert max(ratios) / min(ratios) < 1.01


def test_incoming_weight():
    assert incoming_weight(None) == 1.0
    assert incoming_weight(1) == 0.0
    assert incoming_weight(6) == pytest.approx(5 / 6)


# ----------------------------------------------------------------------------- proxy

def exact_tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def test_tanh_identity_is_exact_under_the_mixture():
    rng = random.Random(0)
    for _ in range(50):
        k = rng.randint(2, 6)
        p = [rng.random() for _ in range(k)]
        q = [rng.random() for _ in range(k)]
        p = [x / sum(p) for x in p]
        q = [x / sum(q) for x in q]
        mix = [(a + b) / 2 for a, b in zip(p, q)]
        expectation = sum(w * abs(math.tanh(0.5 * (math.log(a) - math.log(b))))
                          for w, a, b in zip(mix, p, q))
        assert expectation == pytest.approx(exact_tv(p, q), abs=1e-12)


def test_tanh_estimator_is_unbiased_by_monte_carlo():
    rng = random.Random(1)
    p = [0.5, 0.3, 0.2, 0.0]
    q = [0.1, 0.1, 0.4, 0.4]
    target = exact_tv(p, q)
    ests = []
    for _ in range(4000):
        zs = rng.choices(range(4), weights=p, k=6) + rng.choices(range(4), weights=q, k=6)
        li = [math.log(p[z]) if p[z] > 0 else -math.inf for z in zs]
        lj = [math.log(q[z]) if q[z] > 0 else -math.inf for z in zs]
        ests.append(tanh_tv(li, lj))
    assert sum(ests) / len(ests) == pytest.approx(target, abs=0.01)


def test_tv_bounds_value_gap_for_a_common_verifier():
    # Prop. 3 / Cor. 2 sanity: |E_P r - E_Q r| <= TV(P,Q) for r in [0,1]
    rng = random.Random(2)
    for _ in range(200):
        k = 5
        p = [rng.random() for _ in range(k)]
        q = [rng.random() for _ in range(k)]
        p = [x / sum(p) for x in p]
        q = [x / sum(q) for x in q]
        r = [rng.random() for _ in range(k)]
        gap = abs(sum(a * x for a, x in zip(p, r)) - sum(b * x for b, x in zip(q, r)))
        assert gap <= exact_tv(p, q) + 1e-12


def test_tail_correction_and_coefficient():
    assert tail_correct(0.0, 0.01) == pytest.approx(0.01)
    assert tail_correct(1.0, 0.01) == pytest.approx(1.0)
    assert tail_correct(0.5, 0.01) == pytest.approx(0.505)
    d = {(0, 1): 0.5, (0, 2): 1.0, (1, 2): 0.0}
    assert dispersion_coefficient(d, 3) == pytest.approx((0.25 + 1.0) / 6)


def test_redundancy_prune_removes_max_degree_first():
    # pilot 0 is near-identical to 1, 2 and 3; 1..3 are mutually distinct
    d = {(0, 1): 0.0, (0, 2): 0.0, (0, 3): 0.0, (1, 2): 0.5, (1, 3): 0.5, (2, 3): 0.5}
    assert redundancy_prune([0, 1, 2, 3], d, 0.02) == [1, 2, 3]
    # a chain 0-1-2: degrees 1,2,1 -> remove 1, then no edges
    d2 = {(0, 1): 0.0, (1, 2): 0.0, (0, 2): 0.9}
    assert redundancy_prune([0, 1, 2], d2, 0.02) == [0, 2]


def _fake_pilots(dists, rng, r=6, finished=None):
    """Build pilots whose support is drawn from their own categorical distribution."""
    finished = finished or [False] * len(dists)
    pilots, owners, samples = [], [], []
    for i, dist in enumerate(dists):
        pilots.append(Pilot(index=i, token_ids=[i, i, i, i], finished=finished[i], reward=1.0 if finished[i] else None,
                            sum_logprob=-2.0, num_tokens=4))
        if not finished[i]:
            for _ in range(r):
                samples.append(rng.choices(range(len(dist)), weights=dist, k=1)[0])
                owners.append(i)
    scores = {}
    for i, dist in enumerate(dists):
        if finished[i]:
            continue
        for s, z in enumerate(samples):
            scores[(i, s)] = math.log(dist[z]) if dist[z] > 0 else -math.inf
    return pilots, owners, scores


def test_identical_pilots_are_pruned_and_distinct_ones_kept():
    rng = random.Random(3)
    same = [0.25, 0.25, 0.25, 0.25]
    cfg = VDRAConfig()
    pilots, owners, scores = _fake_pilots([same] * 6, rng)
    res = build_proxy(pilots, owners, scores, cfg)
    assert res.k_v == 1 and res.capped
    assert res.coef == pytest.approx(15 * 0.01 ** 2 / 30)
    disjoint = [[1.0 if j == i else 0.0 for j in range(6)] for i in range(6)]
    pilots, owners, scores = _fake_pilots(disjoint, rng)
    res = build_proxy(pilots, owners, scores, cfg)
    assert res.k_v == 6 and not res.capped
    assert res.coef == pytest.approx(15 / 30)


def test_terminal_pilot_pairs():
    rng = random.Random(4)
    cfg = VDRAConfig(terminal_pair_mode="answer")
    pilots, owners, scores = _fake_pilots([[1.0]] * 3, rng, finished=[True, True, False])
    res = build_proxy(pilots, owners, scores, cfg)
    assert res.d_corr[(0, 1)] == 0.0      # same reward
    assert res.d_corr[(0, 2)] == 1.0      # terminal vs continuable


# ----------------------------------------------------------------------------- async queues

def test_async_queue_manager_dispatches_by_capacity_and_timeout():
    log = []

    async def main():
        mgr = AsyncQueueManager(n_queues=2, capacity=16, timeout_s=0.05, n_min=1,
                                on_dispatch=lambda qid, items, alloc, b, w: log.append((qid, len(items), alloc, b)))
        coros = [mgr.submit(QueueItem(str(i), coef=1.0, k_v=6, nominal=6)) for i in range(20)]
        return await asyncio.gather(*coros)

    res = asyncio.run(main())
    assert res == [6] * 20
    sizes = sorted(n for _, n, _, _ in log)
    assert sizes == [4, 16]
    for _, n, alloc, b in log:
        assert sum(alloc) == b == 6 * n


def test_async_queue_manager_reallocates_inside_a_queue():
    async def main():
        mgr = AsyncQueueManager(n_queues=1, capacity=4, timeout_s=10.0, n_min=1)
        coefs = [0.0, 0.0, 1.0, 1.0]
        return await asyncio.gather(*[mgr.submit(QueueItem(str(i), c, 6, 6)) for i, c in enumerate(coefs)])

    res = asyncio.run(main())
    assert sum(res) == 24 and res[0] == res[1] == 1 and res[2] + res[3] == 22


def test_nominal_squared_coefficients_reproduce_heterogeneous_nominal_branching():
    # the treetune null test uses C = nominal^2 so trees like (4, 6, 8) are reproduced exactly
    nominals = [4, 6, 8, 6, 4]
    items = [QueueItem(str(i), coef=float(b * b), k_v=6, nominal=b) for i, b in enumerate(nominals)]
    assert allocate_queue(items, n_min=1) == nominals
