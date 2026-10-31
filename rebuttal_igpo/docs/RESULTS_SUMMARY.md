# Results summary

This report separates measured outputs from missing run evidence. The archive does not establish that the archived checkpoints were trained with the settings used here. A3/A2/A4/A5 and X are controlled frozen-checkpoint probes; they are not replays of the unknown training run.

## A3 bank and allocator diagnostics

A3 collected 112 parent states from 16 pre-registered MATH validation prompts at iteration 100. The bank contains 12,029,177 continuation tokens, took 5,640.6 seconds wall time, and averaged 2,674 generated continuation tokens/s during generation calls. Its 7,345,343-byte compressed file remains local; the SHA-256 is recorded in `results/MANIFEST.md`.

Across the 112 parents, GEAR proxy variance and the independent `R_b` reward sample variance had Spearman correlation −0.148 (95% prompt-cluster bootstrap CI [−0.518, 0.259]). The interval is broad and includes zero; the per-parent plot is descriptive, with `R_b` still a finite-MC reference.

A2 perturbed the proxy `C` with log-normal noise and evaluated allocations using the independent `R_b` reference half. Across 16 prompt clusters and 50 repetitions per prompt, reallocated rollout share increased with noise; the held-out MC objective remained similar through σ=1 and increased at σ=2.

| Noise σ | Mean held-out `J_Rb` | 95% prompt-bootstrap CI | Rollout share reassigned |
|---:|---:|---:|---:|
| 0 | 0.1925 | [0.0807, 0.3488] | 0.0% |
| 0.25 | 0.1912 | [0.0749, 0.3419] | 4.1% |
| 0.5 | 0.1920 | [0.0710, 0.3437] | 8.0% |
| 1 | 0.2024 | [0.0813, 0.3435] | 15.4% |
| 2 | 0.2313 | [0.0864, 0.4032] | 28.6% |

A4 measured 672 candidate pairs. Spearman correlation between TV and absolute child-value gap was −0.0493; top-quartile overlap was 17.3%. This is descriptive: A4 reuses the candidate-child samples, so it is not an independent calibration test.

A5 compared controlled-bank allocation rules using `J_Rb`, not true policy-gradient error. The table gives estimates and 95% prompt-cluster bootstrap intervals. `gear_online_snapshot_defaults` uses source-snapshot defaults; it is not an as-trained replay. Under the candidate-bounded `capped_k` regime, the common feasible budget forces all rules to the same allocation.

| Budget regime | Uniform `J_Rb` | Exact Eq. (7) `J_Rb` | GEAR snapshot defaults `J_Rb` |
|---|---:|---:|---:|
| Uncapped | 0.1057 [0.0414, 0.1798] | 0.2229 [0.0891, 0.3806] | 0.1655 [0.0722, 0.2718] |
| Capped | 0.1057 [0.0423, 0.1786] | 0.1701 [0.0698, 0.2815] | 0.1203 [0.0472, 0.2055] |
| Candidate-bounded `capped_k` | 0.1057 [0.0428, 0.1751] | 0.1057 [0.0432, 0.1778] | 0.1057 [0.0478, 0.1763] |

Uniform had the lowest point estimate in the first two regimes on this bank, while intervals overlap. This does not establish superiority beyond this controlled sample. `L_node`, `L_edge`, and `L_heldout` are finite-MC proxies, and `L_heldout` is identical across allocation rules by construction.

## Figures

Reproducible vector PDFs and 300-DPI PNGs for the A2–A5 summaries are in `results/figures/`: [A2 noise sensitivity](../results/figures/fig_A2.pdf), [A3 proxy vs. held-out variance](../results/figures/fig_A3.pdf), [A4 TV/value calibration](../results/figures/fig_A4.pdf), and [A5 allocator comparisons](../results/figures/fig_A5.pdf). Generate them with `PYTHONPATH=/workspace/rebuttal_out/plotdeps:/workspace/VDRA1/.worktrees/rebuttal /venv/main/bin/python -m rebuttal_igpo.rebuttal.plot_results`. The numeric A3 plot source is `results/a3_iter100/analysis/a3_proxy_variance.csv`; it contains hashed prompt IDs and summary statistics only. An A1 figure is unavailable because the archived training trees and queues required for A1 are missing.

## A7: iteration-235 evaluation

Six sequential runs evaluated 16 samples per prompt at temperature 0.6/top-p 0.95, with the source pipeline's fixed seed 2746318213. The 100 prompts were the complete archived AIME24/AIME25/AMC23 splits (30/30/40). Paired percentile bootstrap intervals resample prompts and use 1,000 replicates.

| Dataset | Metric | SPO235 | VDRA235 | VDRA − SPO (95% paired prompt-bootstrap CI) |
|---|---|---:|---:|---:|
| AIME24 | avg@16 | 1.46% | 3.13% | +1.67 pp [0.00, 3.96] |
| AIME24 | Pass@16 | 3.33% | 16.67% | +13.33 pp [3.33, 26.67] |
| AIME25 | avg@16 | 0.63% | 2.29% | +1.67 pp [0.00, 3.96] |
| AIME25 | Pass@16 | 10.00% | 10.00% | 0.00 pp [0.00, 0.00] |
| AMC23 | avg@16 | 22.66% | 25.31% | +2.66 pp [−0.94, 6.41] |
| AMC23 | Pass@16 | 67.50% | 67.50% | 0.00 pp [−12.50, 15.00] |

Only AIME24 Pass@16 excludes zero in this unadjusted interval set; the other paired intervals include zero or are exactly zero. These are single-seed, 16-sample local diagnostics, not official benchmark scores. The SPO checkpoint contains no tokenizer; the VDRA235 tokenizer was used for both policies, and SPO tokenizer identity is unverified. Six runs took 797.36 seconds total on this RTX 3090. Compact hash-only prompt rows and bootstrap tables are in `results/a7_iter235/analysis/`; raw outputs and evaluator logs remain local.

## X: iteration sensitivity

Checkpoint-specific timed smokes passed for both iterations. Iteration 50 produced 42,722 visible tokens in 49.95 seconds (855.26 tokens/s), projecting 6.51 hours with a 1.5× margin. Iteration 150 produced 38,460 visible tokens in 46.53 seconds (826.55 tokens/s), projecting 6.72 hours. Their combined 13.23-hour projection fits the 25-hour window before the GPU cutoff.

At report preparation, the full iteration-50 bank is running and iteration 150 is queued behind it. No X outcome is included yet. The rate summaries are in `results/x_smoke/`; the iteration-specific commands and checkpoints are documented in `docs/EXPERIMENT_COMMANDS.md`.

## Blocked evidence and interpretation limits

- A0 inventory is complete, but training commit/settings and run provenance remain unresolved: the archives lack run configs, logs, stored trees, demos, merged edges, and evaluation outputs.
- A1 is blocked because the archived rollout trees, demos, queues, and merged edges needed for an as-trained allocation replay are absent.
- A6 is blocked because the paper/supplement and run-level timing/configuration artifacts are absent.
- No proxy result here is evidence of lower clipped-PPO gradient error. Local RTX 3090 wall times should not be compared with historical server GPU-hours.
- All probe results use one pre-registered 16-prompt subset and one seed. The MC reference is noisy, and the bootstrap clusters prompts rather than treating nodes as independent.
