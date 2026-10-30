from rebuttal_igpo.rebuttal.sim_eq5_eq7 import (
    check_eq5_parent_identity,
    check_tau_identity,
    simulate_adaptive_plugin_bias,
    simulate_selection_bias,
)


def test_eq5_parent_identity_and_root_case():
    result = check_eq5_parent_identity(replicates=20000, seed=13)
    assert abs(result["empirical"] - result["expected"]) < 0.025
    root = check_eq5_parent_identity(n_parent=1, variances=(1.2,), replicates=1000, seed=13)
    assert root["empirical"] == 0.0
    assert root["expected"] == 0.0


def test_tau_identity_under_iid_sampling():
    result = check_tau_identity(replicates=20000, seed=11)
    assert abs(result["empirical"] - result["expected"]) < 0.02


def test_top_likelihood_candidate_reuse_shifts_reward_mean():
    result = simulate_selection_bias(replicates=20000, seed=17)
    assert result["top_mean"] > result["all_mean"]


def test_adaptive_plugin_estimate_is_optimistic_in_synthetic_case():
    result = simulate_adaptive_plugin_bias(replicates=20000, seed=19)
    assert result["predicted_objective"] < result["true_objective"]
