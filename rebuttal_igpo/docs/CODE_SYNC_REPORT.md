# Code synchronization report

**Source snapshot:** `/workspace/src/gear_source_rebuttal/gear`; it has no `.git` metadata. Hashes are SHA-1 Git blob IDs computed from snapshot files and compared with each IGPO ref tree. A path-level exact hash match is stronger than timestamp comparison; timestamps are used only to choose a nearby reference commit.

## Ref comparison

| IGPO ref | Commit | Identical shared paths | Different shared paths | Snapshot-only paths | IGPO-only paths |
| --- | --- | ---: | ---: | ---: | ---: |
| `origin/main` | `1ff79d0879e1c6d5cbb1d31916683366a5165453` | 592 | 69 | 2598 | 82 |
| `origin/fix/budget-allocation-theory` | `87e4061c998ebb3a348074a6f70b4599fb9f373e` | 592 | 69 | 2598 | 82 |
| `origin/agent/checkpoint-sweep-evals` | `5431bc61c43fd6875bacec0b1d6c6d035f593ce8` | 588 | 73 | 2598 | 82 |
| `origin/codex/aime24-eval-script` | `0151424ef1fbbbd6d1355145091d02ab1017aa28` | 582 | 76 | 2601 | 85 |
| `origin/codex/create-branch-budget-allocation` | `052f2b1b55e27e6e7ecd658370bbf171d4f67720` | 530 | 99 | 2630 | 660 |
| `origin/codex/find-bugs-in-igpo-algorithm-pipeline` | `73b289de011a682f26afceff1c30edc665e11b3b` | 559 | 74 | 2626 | 83 |
| `origin/codex/fix-loop-error-in-expansion` | `0717d630bcaa9302a31d869af834132bc6ec122e` | 559 | 79 | 2621 | 84 |
| `origin/codex/locate-evaluate.sh-code` | `94fe2a6dcc528c5f206afba362d3ad0124add2f5` | 591 | 70 | 2598 | 82 |
| `origin/refactor-code` | `2e591def83f2aba6a592ab1bc82aefd7dd099751` | 528 | 86 | 2645 | 84 |

## Date-based reference and synchronization result

The median mtime among source files (`treetune/`, `guidance/`, `scripts/`, and `tests/`, excluding `__pycache__`) is 2026-06-23 02:05:42 UTC. The nearest commit by absolute timestamp is `94fe2a6dcc528c5f206afba362d3ad0124add2f5` (2026-06-22 09:51:52 UTC; 0.68 days away), on `origin/codex/locate-evaluate.sh-code`. This is a date reference only.

No IGPO ref is an exact source match. By common-path hash counts, `origin/main` and `origin/fix/budget-allocation-theory` tie (592 identical, 69 changed); the date-nearest commit has 591 identical and 70 changed. The gear allocator, TV estimator, online queue, inference strategy, and gear episode generator are snapshot-only. The run configs/logs needed to identify the archived checkpoints' source code are absent; therefore no code ref is confirmed as the run source. The snapshot is used only for a code-level diagnostic candidate, with IGPO as comparison reference.

The snapshot has 3,259 files while IGPO main has 743 blobs. Snapshot-only files are dominated by 1,249 `data/` files (mostly Arrow), 1,030 `wandb/` files, caches, result figures, and the newer gear implementation. These data/results are not counted as evidence of code synchronization.

### Exact common-path differences from the date-nearest commit

| Path | Status |
| --- | --- |
| `PLAN.md` | differs |
| `README.md` | differs |
| `configs/algorithms/ppo.libsonnet` | differs |
| `configs/baselines/spo_tree_MATH.jsonnet` | differs |
| `configs/baselines/spo_tree_deepseekR1Qwen_MATH.jsonnet` | differs |
| `configs/debug.jsonnet` | differs |
| `configs/episode_generators/branch_factor_33333.jsonnet` | differs |
| `configs/episode_generators/branch_factor_4444.jsonnet` | differs |
| `configs/episode_generators/depth_2_W6.jsonnet` | differs |
| `configs/evaluation/cli_overrides.jsonnet` | differs |
| `configs/evaluation/math_benchmarks.libsonnet` | differs |
| `configs/polIter_smollm_135m_eval_MATH.jsonnet` | differs |
| `configs/tasks/aime24_inplace_no_answer_prefix.jsonnet` | differs |
| `configs/trainers/training_args.jsonnet` | differs |
| `guidance/library/_gen.py` | differs |
| `scripts/_common.sh` | differs |
| `scripts/aggregate_stats.py` | differs |
| `scripts/download_cached.sh` | differs |
| `scripts/evaluate.sh` | differs |
| `scripts/evaluate_long_cot.sh` | differs |
| `scripts/inspect_demos.py` | differs |
| `scripts/inspect_tree.py` | differs |
| `scripts/oracle_audit.py` | differs |
| `scripts/package_and_upload_data.sh` | differs |
| `scripts/run_ablations.sh` | differs |
| `scripts/run_all_models.sh` | differs |
| `scripts/run_baseline.sh` | differs |
| `scripts/run_exp1_sample_efficiency.sh` | differs |
| `scripts/run_exp2_prune_share_rate.sh` | differs |
| `scripts/run_exp3_overhead.sh` | differs |
| `scripts/run_exp_deep_tree.sh` | differs |
| `scripts/run_local_math_compare_3_9_27.sh` | differs |
| `scripts/run_local_math_spo_reward_variance.sh` | differs |
| `scripts/run_seeds.sh` | differs |
| `scripts/run_smoke.sh` | differs |
| `scripts/setup.sh` | differs |
| `scripts/start_vllm_server.sh` | differs |
| `scripts/start_vllm_server_named_params.sh` | differs |
| `scripts/tail_demos.sh` | differs |
| `scripts/train_debug.sh` | differs |
| `scripts/train_dpo_MATH.sh` | differs |
| `scripts/train_grpo_MATH.sh` | differs |
| `scripts/train_ppo_MATH.sh` | differs |
| `scripts/train_restem_MATH.sh` | differs |
| `scripts/train_rloo_GSM8K.sh` | differs |
| `scripts/train_rloo_MATH.sh` | differs |
| `scripts/train_spo_chain_MATH.sh` | differs |
| `scripts/train_spo_tree_MATH.sh` | differs |
| `scripts/train_vineppo_GSM8K.sh` | differs |
| `scripts/train_vineppo_MATH.sh` | differs |
| `tests/conftest.py` | differs |
| `tests/test_budget_allocation.py` | differs |
| `tests/test_budget_scheduler.py` | differs |
| `tests/test_local_value_share.py` | differs |
| `tests/test_log_prob_matrix.py` | differs |
| `tests/test_logging_helpers.py` | differs |
| `tests/test_math_competition_evaluation.py` | differs |
| `tests/test_refactor_layout_smoke.py` | differs |
| `tests/test_segment_index.py` | differs |
| `tests/test_thresholds.py` | differs |
| `tests/test_tree_stats.py` | differs |
| `tests/test_tv_distance.py` | differs |
| `tests/test_tv_estimators.py` | differs |
| `tests/test_vllm_launch_scripts.py` | differs |
| `tests/test_vllm_scorer.py` | differs |
| `treetune/episode_generators/__init__.py` | differs |
| `treetune/inference_strategies/__init__.py` | differs |
| `treetune/main.py` | differs |
| `treetune/runtime/policy_iteration_runtime.py` | differs |
| `treetune/trainers/ppo_trainer.py` | differs |

### Required implementation areas

| Area | Snapshot path | Relation to IGPO |
| --- | --- | --- |
| Allocator | `treetune/gear/budget_allocation.py` | Only in snapshot; IGPO has `treetune/ingpo/budget_allocation.py` with a different API and semantics. |
| Queue allocator | `treetune/gear/online_budget.py`, `treetune/gear/budget_scheduler.py` | Online queue is snapshot-only; the similarly named IGPO scheduler differs. |
| TV estimator | `treetune/gear/tv_estimators.py` | Snapshot-only; supports hierarchical/perplexity modes and TV-based candidate clustering. |
| Inference strategy | `treetune/inference_strategies/gear_inference_strategy.py` | Snapshot-only. |
| Episode edge extraction | `treetune/episode_generators/gear_episode_generator.py` | Snapshot-only. |
| Trainer loss/whitening | `treetune/trainers/ppo_trainer.py` | Shared path but content differs; inspect the run config for actual whitening. |
| On-policy iteration/checkpoint mapping | `treetune/runtime/policy_iteration_runtime.py`, `treetune/episode_generators/on_policy_episode_generator.py` | Shared paths; exact hashes and run iteration mapping are included in the detailed provenance. |
| Run configs/server scripts | `configs/gear_defaults.libsonnet`, `configs/polIter_deepseekR1Qwen7B_gear_tree_MATH.jsonnet`, `scripts/start_vllm_server.sh` | Gear config is snapshot-only; server script differs from IGPO main. Run-resolved values pending archive metadata. |

### IGPO main vs theory-fix branch

The fix branch differs from IGPO main in 11 paths: `README.md`, `configs/ingpo_defaults.libsonnet`, `configs/ingpo_overlay.libsonnet`, the allocator and scheduler modules/tests, TV estimator/tests, and `treetune/inference_strategies/ingpo_inference_strategy.py`. The fixed allocator is useful as a named counterfactual; it is not attributed to the gear runs unless the run metadata proves that code was used.
