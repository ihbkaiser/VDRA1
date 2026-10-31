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

| Budget regime | Uniform `J_Rb` | Exact Eq. (7) `J_Rb` | GEAR snapshot defaults `J_Rb` | Oracle (`R_a`) `J_Rb` |
|---|---:|---:|---:|---:|
| Uncapped | 0.1057 [0.0414, 0.1798] | 0.2229 [0.0891, 0.3806] | 0.1655 [0.0722, 0.2718] | 0.1043 [0.0400, 0.1792] |
| Capped | 0.1057 [0.0423, 0.1786] | 0.1701 [0.0698, 0.2815] | 0.1203 [0.0472, 0.2055] | 0.1008 [0.0385, 0.1785] |
| Candidate-bounded `capped_k` | 0.1057 [0.0428, 0.1751] | 0.1057 [0.0432, 0.1778] | 0.1057 [0.0478, 0.1763] | 0.1057 [0.0422, 0.1751] |

The independent-reference Oracle (`R_a` for allocation, `R_b` for evaluation) has the lowest point estimates in the uncapped and capped regimes; the prompt-cluster intervals overlap broadly. The candidate-bounded constraint forces identical feasible counts. These estimates do not establish superiority beyond this controlled sample. `L_node`, `L_edge`, and `L_heldout` are finite-MC proxies, and `L_heldout` is identical across allocation rules by construction.

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

Iteration 50 completed 112/112 parents in 5,853.9 seconds, collecting 13,327,123 continuation tokens (2,836 tokens/s during continuation calls). Its 8,440,187-byte raw bank remains local; hash and compact outputs are listed in `results/MANIFEST.md`.

For iteration 50, the GEAR proxy variance and independent `R_b` reward sample variance had Spearman correlation 0.175 (95% prompt-cluster bootstrap CI [−0.190, 0.440]); this is a noisy finite-MC reference. A2 again shows increasing allocation churn with proxy noise, alongside a modest increase in the held-out objective:

| Noise σ | Mean held-out `J_Rb` | 95% prompt-bootstrap CI | Rollout share reassigned |
|---:|---:|---:|---:|
| 0 | 0.3393 | [0.1707, 0.5119] | 0.0% |
| 0.25 | 0.3427 | [0.1760, 0.5380] | 2.8% |
| 0.5 | 0.3452 | [0.1794, 0.5102] | 5.7% |
| 1 | 0.3566 | [0.1834, 0.5428] | 11.1% |
| 2 | 0.3962 | [0.2132, 0.6018] | 21.7% |

A4 measured 672 pairs; TV vs. absolute child-value gap Spearman correlation was 0.0161 and top-quartile overlap was 26.2%. This reuses the fixed child samples and remains descriptive. A5 `J_Rb` point estimates are shown below; full intervals for all rules are in the published summary CSV. The `oracle` selects with `R_a` and is evaluated on `R_b`.

| Budget regime | Uniform | Exact Eq. (7) | GEAR snapshot defaults | Oracle (`R_a`) |
|---|---:|---:|---:|---:|
| Uncapped | 0.1762 | 0.4088 | 0.2943 | 0.1718 |
| Capped | 0.1762 | 0.2748 | 0.1953 | 0.1705 |
| Candidate-bounded `capped_k` | 0.1762 | 0.1762 | 0.1762 | 0.1762 |

Oracle has the lowest point estimate in the uncapped and capped regimes, but intervals overlap. `capped_k` constrains all rules to the same counts. These outcomes are finite-MC allocation diagnostics, not policy-gradient error or an as-trained replay. The rate summaries are in `results/x_smoke/`, and commands/checkpoint paths are in `docs/EXPERIMENT_COMMANDS.md`.

Iteration 150 also completed all 112 parents: 11,205,934 continuation tokens in 5,460.7 seconds (4,315.3 seconds of continuation calls; 2,597 tokens/s). Its 6,257,986-byte bank remains local; the SHA-256 and analysis file hashes are in `results/MANIFEST.md`. The GEAR proxy variance and independent `R_b` reward sample variance had Spearman correlation −0.342 (95% prompt-cluster bootstrap CI [−0.653, 0.062]).

| Noise σ | Mean held-out `J_Rb` | 95% prompt-bootstrap CI | Rollout share reassigned |
|---:|---:|---:|---:|
| 0 | 0.1919 | [0.0543, 0.3671] | 0.0% |
| 0.25 | 0.1953 | [0.0475, 0.3805] | 4.1% |
| 0.5 | 0.1968 | [0.0459, 0.3784] | 8.0% |
| 1 | 0.2034 | [0.0430, 0.3784] | 15.4% |
| 2 | 0.2290 | [0.0702, 0.4310] | 29.0% |

A4 measured 672 candidate pairs; TV vs. absolute child-value gap Spearman correlation was −0.0632 and top-quartile overlap was 17.9%. This is descriptive because A4 reuses the candidate samples. A5 `J_Rb` point estimates are below; full prompt-cluster intervals are in the analysis CSV.

| Budget regime | Uniform | Exact Eq. (7) | GEAR snapshot defaults | Oracle (`R_a`) |
|---|---:|---:|---:|---:|
| Uncapped | 0.1026 | 0.2217 | 0.1445 | 0.1001 |
| Capped | 0.1026 | 0.1667 | 0.1155 | 0.1014 |
| Candidate-bounded `capped_k` | 0.1026 | 0.1026 | 0.1026 | 0.1026 |

Oracle again has the lowest point estimates in uncapped and capped regimes, with broadly overlapping intervals. All rules receive identical counts under the candidate-bounded constraint. These are finite-MC frozen-checkpoint diagnostics, not as-trained replay or full clipped-PPO gradient error. Iterations 50 and 150 each used the same 16 pre-registered prompts and one seed; the prompt-cluster intervals are wide.

## Blocked evidence and interpretation limits

- A0 inventory is complete, but training commit/settings and run provenance remain unresolved: the archives lack run configs, logs, stored trees, demos, merged edges, and evaluation outputs.
- A1 is blocked because the archived rollout trees, demos, queues, and merged edges needed for an as-trained allocation replay are absent.
- A6 is blocked because the paper/supplement and run-level timing/configuration artifacts are absent.
- No proxy result here is evidence of lower clipped-PPO gradient error. Local RTX 3090 wall times should not be compared with historical server GPU-hours.
- All probe results use one pre-registered 16-prompt subset and one seed. The MC reference is noisy, and the bootstrap clusters prompts rather than treating nodes as independent.
