# Rebuttal experiment plan

All new generations use the iteration-100 archived checkpoint as a **controlled frozen-policy probe**, not as a replay of the unknown training run. Prompts are sampled without replacement from the local MATH validation dataset with Python seed 42; the first eight in that fixed 32-prompt sample form the smoke subset. The prompt template is `gear/configs/prompt_library/qwen_MATH.jsonnet`; sampling uses temperature 0.6, top-p 0.95 and max-new-tokens 1024. Individual request seeds are recorded in `controlled_prompt_manifest.json`. Run settings remain unknown; GEAR defaults used by any proxy are labeled source-snapshot defaults.

## Experiments

| ID | Hypothesis / estimand | Unit, checkpoint and protocol | Metrics and uncertainty | State / constraint |
| --- | --- | --- | --- | --- |
| A0 | Determine artifact presence, code correspondence and mechanism facts | Every extracted archive; snapshot vs all IGPO refs | Exact path/blob match and source citations | Completed as an evidence inventory; checkpoint provenance unresolved. |
| A6 | Reconcile compute and paper claims | Per run/row with matching demos, timing, paper and configs | Tokens, scored tokens, GPU seconds; classify each claim | Blocked: inputs absent; see `COMPUTE_AND_CONSISTENCY_AUDIT.md`. |
| A1 | Describe actual allocation behavior and replay counterfactuals | All saved VDRA trees, by run/stage/depth/queue | Counts, zero/one/>6 shares, Gini/HHI, utilization, queue balance, proxy objective | Blocked: no saved rollout trees/demos/edges. |
| T | Check Eq. (5)/(7) identities and failure modes | Seeded synthetic fixed trees; 20,000 replications | Empirical vs analytic expectations; synthetic selection effects | Completed; no checkpoint data; see `DERIVATION_EQ5_EQ7.md`. |
| S | Measure vLLM throughput and peak observed memory | 8 prompts × 6 completions, iteration 100, 1024-token cap, BF16 | Output tokens/s, per-call seconds, sampled GPU memory; raw completions omitted | Completed: 40,571 output tokens in 49.06 s (826.93 tokens/s), 20,218 MiB peak sampled memory. |
| A3 | Measure proxy-guided allocation on held-out policy outcomes | 16 prompts (first 16 IDs of fixed manifest), depths 0–1, iteration 100; 112 parents; independent `R_a=32` / `R_b=32` MC halves; 16 child continuations | `J`, regret, oracle-gap recovery, changed rollout share; prompt-clustered bootstrap | Running; 7/112 parents in the 2026-10-30 21:57 UTC checkpoint, preliminary 2,189 output tokens/s. |
| A2 | Quantify sensitivity to proxy noise | Repeated `C′=C exp(ε)`, `s∈{0,.25,.5,1,2}`, 50 repetitions on A3 bank | Allocation distance and held-out `Σ σ²_Rb/n`; prompt-clustered bootstrap | CPU command implemented; run after A3 completes. |
| A5 | Compare Eq. (5)-related value-error proxies under allocations | A3 parents and candidate children; fixed held-out child edges | `L_node`, weighted `L_edge`, fixed `L_heldout`; clustered bootstrap | CPU command implemented; run after A3 completes. |
| A4 | Test proxy calibration against held-out value dispersion | Candidate child pairs and A3 bank | Binned calibration, Spearman, top-quartile overlap, approximate noise bands | CPU command implemented; run after A3 completes. |
| A7 | Compare frozen-checkpoint evaluation | VDRA235 and matched SPO235 on AIME24/25 + AMC23; 16 samples, temp .6, top-p .95, seed 42, source verifier | avg@16, Pass@16 and paired prompt bootstrap | Requires evaluator compatibility, checkpoint tokenizer and measured fit before cutoff. |
| X | Check iteration sensitivity | Iterations 50/150 under A3/A5 protocol | Same metrics as A3/A5 | Run only after A7 if measured work fits; otherwise iteration 150 only or defer. |

## Interpretation limits

The archive does not establish training-time settings or training-tree behavior. No new probe result will be attributed to the archived training trajectory. `R_a` selects oracle allocations and `R_b` evaluates them; they must remain disjoint. Bootstrap resampling is by prompt, because several nodes share each prompt. RTX 3090 seconds describe this local diagnostic only. Lower `J`, `L_node`, or `L_heldout` is not evidence of lower full clipped-PPO gradient error or higher benchmark accuracy.
