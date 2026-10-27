import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
sys.path.insert(0, str(ROOT / "analysis"))

from bank_io import load_bank, save_bank  # noqa: E402
from synthetic_bank import make_bank  # noqa: E402
import exp1_proxy_quality as e1  # noqa: E402
import exp2_advantage_error as e2  # noqa: E402
import stats  # noqa: E402


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    p = tmp_path_factory.mktemp("bank") / "R2_s42_step100.jsonl.gz"
    save_bank(str(p), {"label": "R2_s42_step100"}, make_bank(n_roots=24, seed=1, proxy_noise=0.3))
    return str(p)


def test_bank_roundtrip(bank):
    header, roots = load_bank(bank)
    assert header["label"] == "R2_s42_step100" and len(roots) == 24
    assert all(len(c["refs"]) == 32 for r in roots for c in r["pool"])


def test_exp1_oracle_is_best_and_informative_proxy_helps(bank):
    res = e1.analyse_bank(bank, reps=60, capacity=16, nominal=6, seed=0, use_true=True)
    rec = {r["method"]: r["recovery"]["median"] for r in res["rows"]}
    assert rec["oracle"] == pytest.approx(1.0)
    assert rec["uniform"] == pytest.approx(0.0)
    assert rec["vdra"] > 0.3                      # proxy correlated with true dispersion
    assert rec["vdra"] > rec["random_perm"]       # information, not just non-uniformity


def test_exp2_measurement_matches_prop_1_2_prediction(bank):
    res = e2.analyse(bank, draws=40, capacity=16, nominal=6, seed=0, use_true=True)
    for r in res["rows"]:
        pred, meas = r["pred_d1_d2_leaf"]["median"], r["meas_d1_d2_leaf"]["median"]
        assert meas == pytest.approx(pred, rel=0.06), r["method"]


def test_exp2_reference_half_noise_correction_is_close_to_truth(bank):
    """Using the noisy reference half with the correction should land near the
    exact-target answer for the node-value MSE."""
    exact = e2.analyse(bank, draws=40, capacity=16, nominal=6, seed=0, use_true=True)
    noisy = e2.analyse(bank, draws=40, capacity=16, nominal=6, seed=0, use_true=False)
    for a, b in zip(exact["rows"], noisy["rows"]):
        assert b["mse_d1"]["median"] == pytest.approx(a["mse_d1"]["median"], abs=0.004)


def _fake_eval(tmp_path):
    rng = np.random.default_rng(0)
    runs = []
    evals = []
    benches = {"math500": 200, "aime24": 30}
    for pair, base_p, gain in (("spo", 0.5, 0.05), ("treepo", 0.45, 0.0)):
        for seed in (42, 123, 2026):
            for role in ("baseline", "hybrid"):
                rid = f"{pair}_{role}_{seed}"
                runs.append({"run_id": rid, "pair": pair, "role": role, "seed": seed})
                steps = list(range(0, 221, 25)) + ([220, 225, 230, 235, 240, 245, 250, 255, 260, 265] if role == "baseline" else [220])
                for st in sorted(set(steps)):
                    p = base_p + (gain if role == "hybrid" else 0) * st / 220 + 0.1 * st / 265
                    gh = st * (0.44 if role == "baseline" else 0.53)
                    for b, n in benches.items():
                        for i in range(n):
                            evals.append({"run_id": rid, "step": st, "gpu_hours": gh, "benchmark": b,
                                          "mode": "greedy", "problem_id": str(i),
                                          "correct": float(rng.random() < p), "k_correct": 0, "k": 1})
    with open(tmp_path / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "pair", "role", "seed"])
        w.writeheader()
        w.writerows(runs)
    with open(tmp_path / "eval.jsonl", "w") as f:
        for r in evals:
            f.write(json.dumps(r) + "\n")


def test_stats_pipeline_detects_a_real_gain(tmp_path):
    _fake_eval(tmp_path)
    out = tmp_path / "stats"
    subprocess.run([sys.executable, str(ROOT / "analysis/stats.py"), "--eval", str(tmp_path / "eval.jsonl"),
                    "--runs", str(tmp_path / "runs.csv"), "--out", str(out), "--reps", "400"], check=True)
    rows = list(csv.DictReader(open(out / "comparisons.csv")))
    got = {(r["pair"], r["regime"]): r for r in rows}
    assert float(got[("spo", "matched_updates")]["diff_pp"]) > 0
    assert "baseline step" in got[("spo", "matched_gpu_hours")]["notes"]
    assert all(0.0 <= float(r["p_holm"]) <= 1.0 for r in rows)
    subprocess.run([sys.executable, str(ROOT / "analysis/figures.py"), "--results", str(tmp_path),
                    "--out", str(tmp_path / "fig")], check=True)


def test_holm():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_auc_works_with_numpy_versions_without_trapezoid(monkeypatch):
    monkeypatch.delattr(stats.np, "trapezoid", raising=False)
    assert stats.auc([0, 1, 2], [0, 1, 0], 2) == pytest.approx(0.5)
