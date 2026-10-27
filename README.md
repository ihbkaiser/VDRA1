# VDRA rebuttal experiments: code and run scripts

This package contains code and launch plans for the four experiments proposed for reviewer Pvw4. The intended matrix has six training configurations (R1–R6), each run with 3 seeds; it is a plan, not a claim that all runs have completed. Actual run status and evidence are tracked in [`codex/STATUS.md`](codex/STATUS.md).

| Exp | Question | What runs |
| --- | --- | --- |
| 1 | Is the TV proxy better than cheap heuristics? | Frozen-policy node banks from R1/R2; 8 allocation methods compared on J(n) = Σ σ²_MC / n_v |
| 2 | Does Eq. 7 reduce the Eq. 5 error? | The same banks; depth-3 trees rebuilt by resampling; measured edge-averaged advantage MSE |
| 3 | Does VDRA add value on top of TreePO and TreeRL? | R3 vs R4 and R5 vs R6, 3 seeds each |
| 4 | Do the gains survive compute matching? | Each baseline continued (or run long) to its hybrid's GPU-hours |

For current Phase D–F progress, B200 results, speed measurements, tests, Modal spend guard, and open experiment gates, see [`codex/STATUS.md`](codex/STATUS.md). Run the CPU regression suite with `python -m pytest -q tests`; its latest result is recorded in the status file. GPU evidence and CPU-only checks are reported separately.

---

## 1. What is in the box

```
vdra_core/                framework-free VDRA (used by all three integrations and the diagnostics)
  config.py               VDRAConfig: paper Table 4 defaults
  proxy.py                tanh likelihood-ratio TV estimator, tail correction (Eq. 9), C_v (Alg. 1 l.13),
                          redundancy pruning (Alg. 1 l.14-15), pilot-NLL / pilot-diversity heuristics
  resume.py               resume plan (skip / stop / save rules) and RNG capture, used by the TreeRL patch
  allocation.py           exact bounded solver for  min Σ C_v/n_v  (greedy marginal gain), upper bounds u_v,
                          incoming-edge weight λ = (n_p-1)/n_p
  queues.py               asyncio cross-prompt queues (32 x 16, 1 s timeout) for treetune
  backend.py              batched token-id sampling / cross-scoring on vLLM (offline LLM, verl engine, Ray actor)
  verifier.py             one math verifier for every run (treetune's grader when igpo is importable)
integrations/
  igpo/                   treetune strategy "vdra" (paper-faithful) + "spo_logged" (SPO + logging),
                          "hybrid_episode_generator_resumable" (replay buffer persisted across restarts),
                          configs, install.sh
  treepo/                 verl patch: tree_div_policy=vdra; custom reward; install.sh (pinned commit)
  treerl/                 OpenRLHF patch: VDRA fork allocation, local verifier, int EOS ids, full resume
                          (DeepSpeed state + data position + RNG), stop at a given update; install.sh (pinned commit)
diagnostics/              node-bank builder (vLLM), Exp 1, Exp 2, synthetic banks for testing
analysis/                 data split, leakage check, evaluator, GPU-hours, budget audit, statistics, figures
scripts/                  numbered bash drivers (see section 3); env.sh holds every setting
tests/                    CPU regression tests
```

### How VDRA was implemented

Your igpo repo implements VDRA as "InGPO budget-allocation mode". At `main` (1ff79d0) and on `fix/budget-allocation-theory` (87e4061), that code differs from the paper in several places: distance-to-value map, missing tail correction, support sampling, missing redundancy cap, per-tree rather than cross-prompt queues, and biased child selection on `main`.

Rather than patching `ingpo_inference_strategy.py`, `integrations/igpo/install.sh` creates branch `vdra-paper` from `fix/budget-allocation-theory` and adds a new strategy type `vdra` that follows Alg. 1 of the supplement:

1. 6 pilots × 96 tokens from every non-final node.
2. 6 support samples × 32 tokens per continuable pilot.
3. Every continuable pilot's prefix scores every support sample of the node (vLLM `/completions`, `echo=True`, through the existing `VLLMLogprobClient`).
4. Pairwise distance D = mean |tanh(Δlog p / 2)|, then D̃ = D + (1−D)·0.01, then C_v = Σ D̃² / (k0(k0−1)), then redundancy graph with threshold 0.02 and greedy max-degree removal, giving k_v.
5. Cross-prompt queues: insert into the fullest non-full queue, dispatch on capacity 16 or a 1 s timeout, B_q = 6·|Q|, exact solver with n_min = 1.
6. Terminal pilots are reused and count toward n_v. The remaining n_v − t_v children are sampled fresh with SPO's own expander. The final depth stays uniform.

Everything after tree construction is SPO's own code: `hybrid_episode_generator`, edge extraction, replay buffer, `rloo` advantage, whitening and the PPO loss. R1 runs the unchanged `hybrid` strategy under the name `spo_logged`, which only adds per-tree token logging. R1 and R2 therefore differ only in the number of children per node.

Config overlays in `configs/vdra/`:

- **`kl0`** sets KL to 0 for both R1 and R2. The DeepSeek MATH base config otherwise imports a 1e-4 KL loss.
- **`data_nodiag`** trains without the 256 diagnostic problems.
- **`model_path`** points every model reference at `MODEL_PATH`; the base configs hard-code `/workspace/storage-shared/...`.
- **`stop`** sets `early_stop_iteration`, so `num_iterations` (and the WarmupDecayLR horizon of 1000) is unchanged and continuations resume on the same schedule.
- **`variant`** merges `VDRA_VARIANT_JSON`, used for the null test and ablations.

### TreePO and TreeRL hybrids

- **TreePO (R4).** The hook is the per-root split of `total_divergence` in `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`, which is the function the worker actually calls. TreePO's budget transfer, fallback, early stop, subgroup advantage, dynamic sampling and DAPO loss are untouched. The proxy runs on the in-process vLLM engine. With `proxy=constant` the branch reproduces `fixed_avg` exactly.
- **TreeRL (R6).** The hook replaces `expansion_tasks * T` in `entropy_chain_local_manager.py` with a VDRA split of the same T·|forks| budget, in queues of up to 16 forks of one problem. Entropy fork selection, process rewards with the 1/√|L| reweighting, and the REINFORCE loss are untouched. `VDRA_LOCAL_VERIFIER=1` replaces TreeRL's three remote judge services with the shared verifier.

---

## 2. Setup

```bash
cp scripts/env.sh scripts/env.local.sh      # then edit: WORK, MODEL_PATH, CONDA_*, GPU_GROUPS, EVAL_SETS
bash scripts/00_setup.sh                    # clones igpo/TreePO/TreeRL, installs the integrations
bash scripts/01_prepare_data.sh             # diag split + exports + leakage check
bash scripts/02_unit_tests.sh               # CPU tests (about 1 minute)
```

You need three Python environments, because the frameworks pin different torch and vLLM versions. Set them in `env.local.sh` as `CONDA_IGPO`, `CONDA_TREEPO` and `CONDA_TREERL`.

- **igpo (R1, R2, banks, evaluation):** `pip install -r $IGPO_DIR/requirements.txt`.
- **TreePO (R3, R4):** follow `$TREEPO_DIR/recipe/treepo/README.md` (verl 0.3.1.dev, vLLM 0.8.5).
- **TreeRL (R5, R6):** `pip install -r $TREERL_DIR/requirements.txt`.

Download the evaluation sets with `python $IGPO_DIR/scripts/download_eval_datasets.py`. Then check that `EVAL_SETS` points at them; MATH-500 is assumed to be the `test` split of `data/math`, so verify that.

Every script accepts `DRY_RUN=1`, which prints commands without executing them. All of them were dry-run here.

---

## 3. Run order

| Phase | Command | GPUs | Gate / output |
| --- | --- | --- | --- |
| 0 | `GPUS=0,1 bash scripts/10_null_tests.sh` | 2–3 | `results/audit/null_igpo.json` → `"pass": true`; `budget_violations` = 0 for R2/R4/R6 |
| 0b | `GPUS=0,1 RUNS="R2 R4" bash scripts/12_resume_test.sh`; `GPUS=0,1,2 RUNS="R6" bash scripts/12_resume_test.sh` | 2–3 | `results/audit/resume_test.txt` → `PASS` for every framework |
| 1 | `RUNS="R1 R2 R3 R4" GPU_GROUPS="0,1 2,3 4,5 6,7" bash scripts/11_pilot.sh` then `RUNS="R5 R6" GPU_GROUPS="0,1,2 3,4,5" bash scripts/11_pilot.sh` | 8 | G1: `cv_n_v` ≥ 0.3 for hybrids; measured seconds per update; calibrate `TREEPO_WIDTH` and `TREERL_*` so main-rollout tokens per prompt are within ±15% of SPO |
| 2 | `PHASE=seed42 RUNS="R1 R2 R3 R4" bash scripts/21_run_matrix.sh` and `PHASE=seed42 RUNS="R5 R6" GPU_GROUPS="0,1,2 3,4,5" bash scripts/21_run_matrix.sh` | 8 | 6 runs to 220 updates |
| 2b | `bash scripts/40_build_banks.sh`, then `bash scripts/41_exp1.sh` and `bash scripts/42_exp2.sh` | 1 per bank | `results/exp1`, `results/exp2` |
| 3 | `PHASE=rest ...` (same two commands) | 8 | 12 more runs |
| 3b | `PAIR=spo SEED=<s> GPUS=0,1 bash scripts/30_compute_matched.sh` (also `PAIR=treepo`, and `PAIR=treerl` with 3 GPUs) | 2–3 | Baselines continued from their step-220 state to the hybrid's GPU-hours |
| 4 | `bash scripts/50_eval.sh` then `bash scripts/60_analysis.sh` | 1 per checkpoint | `results/stats/comparisons.csv`, `auc.csv`, `curves.csv`, `results/figures/E1–E4` |

Re-running any training command resumes it (section 4). `30_compute_matched.sh` continues every baseline, TreeRL included, from its step-220 state.

`12_resume_test.sh` (phase 0b) must run alone on a node, because it ends with `ray stop --force`.

| When | Command | Purpose |
| --- | --- | --- |
| any | `bash scripts/22_resume_all.sh` (status only), then `APPLY=1 FRAMEWORKS="treetune verl" bash scripts/22_resume_all.sh` | After a node failure: lists every run and resumes the incomplete ones with the parameters of their last launch |

---

## 4. Resuming training

Every training process can be stopped at any time and resumed with the same command.

| | treetune (R1, R2) | verl / TreePO (R3, R4) | OpenRLHF / TreeRL (R5, R6) |
| --- | --- | --- | --- |
| Resumable state every `RESUME_EVERY` (5) updates and at `STOP` | Native: weights + DeepSpeed optimizer + LR schedule; prompt order is seeded by iteration | Native: FSDP weights + optimizer + LR schedule + dataloader state (`trainer.resume_mode=auto`) | **Added**: DeepSpeed module + optimizer + LR schedule, plus update count, consumed prompt micro-batches, KL value and RNG (`ckpt/_actor_resume`) |
| Extra state | **Added**: SPO's on-policy replay buffer (stale edges kept up to 8 iterations) saved per iteration and rank, restored on resume | — | On resume the policy weights are pushed to the vLLM engines before the first rollout |
| Evaluation weights | Every `KEEP_EVERY` (25; 5 in continuations) and the newest | HF merge of every `KEEP_EVERY` step and `STOP`; older FSDP states pruned except the two newest and the resume point | HF save every `KEEP_EVERY` and at `STOP` |
| Stop at `STOP` | `early_stop_iteration` | `trainer.total_training_steps` | `VDRA_STOP_UPDATE` (dataset and shuffle no longer depend on `STOP`) |

**The wrapper in `20_train.sh` adds four things:**

- **Lock:** a second launch of a running run exits (`flock` on `RUN_DIR/.lock`).
- **Done-check:** a run already at `STOP` is skipped. This matters for verl, which would otherwise train one extra step after resuming at its last step.
- **Heartbeat:** written to `run_meta.jsonl` every 2 minutes.
- **Automatic restarts:** up to `MAX_RESTARTS` (3), `RESTART_DELAY` (60 s) apart, each resuming from the newest checkpoint.

`run_meta.jsonl` records every launch with its parameters, the update it resumed from, and its exit code. `22_resume_all.sh` reads these records.

**GPU-hours stay honest across crashes:**

- treetune keeps one timing record per iteration.
- For verl and OpenRLHF, time after a crashed launch's last checkpoint (work that was recomputed) and time between launches are excluded.

**Bug fixed in TreeRL.** Upstream `save_logs_and_checkpoints` used `tag` outside the branch that defines it, so `--save_ckpt` raised `NameError` on the first non-save step. Its own `_actor_global_stepN` resume skipped the wrong number of batches (updates, not micro-batches) and restored no optimizer state. Both are replaced.

`STOP` must be a multiple of `RESUME_EVERY` for treetune runs (it only checkpoints on multiples). The wrapper checks this, and `30_compute_matched.sh` rounds its targets.

---

## 4b. Evaluation during training (`EVAL_EVERY_N`)

With `EVAL_EVERY_N=25` (default in `env.sh`), `20_train.sh` trains in chunks:

```
eval step 0 (base model)  →  train 0→25  →  eval 25  →  train 25→50  →  eval 50  → … →  train 200→220  →  eval 220
```

At each pause the trainer has exited, so the run's own GPUs are free:

1. Leftover trainer processes (Ray workers, vLLM engines) are killed. Only processes tagged with this launch are touched, so other runs on the node are safe.
2. `gpu_hours.py` updates the run's GPU-hours.
3. `analysis/eval_pending.py` evaluates every checkpoint that has no results yet. It splits the benchmarks over the run's GPUs (largest first) and runs one shared evaluator (`eval_checkpoints.py`, same verifier for every framework) per GPU in parallel.
4. Training resumes from the checkpoint it just saved, using the same resume path as a crash restart.

The same code is used for all three frameworks. Each chunk is an ordinary launch that stops at the chunk end (`early_stop_iteration` / `total_training_steps` / `VDRA_STOP_UPDATE`), and the next one resumes.

- **Results:** one file per (run, step, benchmark, mode) in `results/eval/`, written atomically. Results are reused only when their sidecars match the checkpoint, exact dataset rows, tokenizer and verifier fingerprints, evaluation settings, row count, and result-byte digest. Legacy filename-only files are reevaluated. `scripts/50_eval.sh` fills in anything missing.
- **GPU-hours:** evaluation is logged as `eval_start`/`eval_end` in `run_meta.jsonl`, outside any training launch, so it never counts as training compute.
- **Failures:** an evaluation failure does not stop training. It is logged; re-run `50_eval.sh` later.
- **Settings:**
  - `EVAL_EVERY_N=0` turns pauses off (evaluate afterwards with `50_eval.sh`).
  - `EVAL_STEP0=0` skips the base model.
  - `EVAL_AT_END=1` evaluates when `STOP` is reached.
  - `SAMPLE_K` sets avg@k on AIME/AMC.
  - For R1/R2, `EVAL_EVERY_N` must be a multiple of `RESUME_EVERY`; the wrapper checks this.
- **Cost:** each pause adds one model reload (about 1–2 min) plus vLLM start-up for the next chunk. With `EVAL_EVERY_N=25` and `STOP=220` that is 9 pauses per run.
- **Continuations:** `30_compute_matched.sh` runs continuations with `EVAL_EVERY_N=0 EVAL_AT_END=1`, so the dense every-5 checkpoints are evaluated once at the end.

## 4c. One GPU per experiment (`scripts/exp/`)

Six scripts, one per experiment. Each one runs everything for its experiment on a single GPU.

| Script | Experiment | Default GPU |
| --- | --- | --- |
| `scripts/exp/R1_spo.sh` | SPO baseline (treetune) + compute-matched continuation | 0 |
| `scripts/exp/R2_spo_vdra.sh` | SPO + VDRA (treetune) | 1 |
| `scripts/exp/R3_treepo.sh` | TreePO baseline (verl) + compute-matched continuation | 2 |
| `scripts/exp/R4_treepo_vdra.sh` | TreePO + VDRA (verl) | 3 |
| `scripts/exp/R5_treerl.sh` | TreeRL baseline (OpenRLHF) + compute-matched continuation | 4 |
| `scripts/exp/R6_treerl_vdra.sh` | TreeRL + VDRA (OpenRLHF) | 5 |

```bash
bash scripts/00_setup.sh && bash scripts/01_prepare_data.sh && bash scripts/02_unit_tests.sh   # once
bash scripts/exp/launch_all.sh                 # all six in the background, GPUs 0-5
# or one at a time / another GPU / fewer seeds:
GPU=6 SEEDS="42" bash scripts/exp/R4_treepo_vdra.sh
tail -f work/runs/_logs/R4_gpu6.log
bash scripts/60_analysis.sh                     # when all six are done
```

**What each script does:**

- **Training and evaluation:** each seed in `SEEDS` is trained one after another to `STOP_STEP`. Training pauses every `EVAL_EVERY_N` updates to evaluate on the same GPU, then resumes.
- **Baselines (R1, R3, R5):** after their own seeds, they wait for the matching VDRA run with the same seed to finish (it checks every 10 min). Then the baseline continues to equal GPU-hours, and its continuation checkpoints are evaluated. Set `COMPUTE_MATCHED=0` to skip this.
- **Re-running:** a script resumes where it stopped. Finished seeds are skipped.
- **Failures:** a seed that fails after `MAX_RESTARTS` is reported, and the script moves on to the next seed.

**How each framework fits on one GPU.** Batch sizes are global, so the algorithm is the same as on several GPUs; only the speed changes. All six runs use 1 GPU, so GPU-hours stay comparable. `30_compute_matched.sh` refuses a pair whose runs used different GPU counts.

| | How it shares the GPU | Knobs if memory is short |
| --- | --- | --- |
| treetune (R1, R2) | Native: DeepSpeed with world size 1, with gradient accumulation up to the target batch; the vLLM server's memory share is set from free memory | `IGPO_VLLM_MEM=0.3` |
| verl (R3, R4) | Hybrid engine: the FSDP actor and vLLM rollout are colocated, and the KV cache is freed during the update. vLLM share is 0.4 on 1 GPU. No reference model (KL is off) | `VERL_VLLM_MEM=0.3`, `VERL_OFFLOAD=True` (CPU-offloads params and optimizer; slower) |
| OpenRLHF (R5, R6) | **Added to the patch:** actor, reference model and vLLM engine as fractional Ray GPUs (0.4 / 0.2 / 0.4). vLLM starts first so its memory profiling sees an empty GPU (0.35 of memory, max length 4096). Weights reach vLLM in 512 MB buckets through the Ray object store, because NCCL cannot connect two processes on one GPU | `TREERL_VLLM_MEM=0.25`, `TREERL_ADAM_OFFLOAD=1` |

The defaults assume an 80 GB GPU (A100/H100 80 GB). For a 1.5B model, the full-parameter Adam state is about 24 GB. On 40–48 GB GPUs, use the offload knobs.

**Before launching all six,** run one short check per framework:

```bash
GPU=0 SEEDS=42 STOP_STEP=10 EVAL_EVERY_N=5 COMPUTE_MATCHED=0 RUNS_DIR=work/runs_smoke bash scripts/exp/R6_treerl_vdra.sh
```

This checks memory and the colocated weight sync, which was tested here only with mocks. Repeat it for R2 and R4.

**Wall-clock.** The full matrix (6 experiments × 3 seeds × 220 updates, plus evaluation and continuations) runs as six sequential queues. Expect it to take several times longer per run than the multi-GPU layout. To cut wall-clock time with more GPUs, use `SEEDS="42"` on GPUs 0–5 and `SEEDS="123"` on GPUs 6–11. These use separate run directories, so they do not interfere.

## 5. Outputs and how they map to the rebuttal

| File | Content |
| --- | --- |
| `results/exp1/exp1_summary.csv` | Per bank × method: J, % reduction vs uniform, oracle-gap recovery (median and 95% CI), Spearman vs σ²_MC, agreement with oracle counts, auxiliary tokens per node |
| `results/exp2/exp2_summary.csv` | Per bank × method: own-tree and trained-edge L_adv by edge type, node-value MSE by depth, shared-frontier MSE, paired difference vs uniform, Prop. 1–2 prediction next to the measurement |
| `results/stats/comparisons.csv` | Per pair × regime (matched updates or matched GPU-hours): hierarchical-bootstrap difference in macro Pass@1 (pp, 95% CI), Holm-adjusted p, seeds won, mixed-model log-odds |
| `results/stats/auc.csv` | Areas under Pass@1-vs-steps and Pass@1-vs-GPU-hours over the common range |
| `results/audit/*.json` | Budget conservation, allocation headroom, share of n_v = 1, tokens per tree, null test |
| `results/gpu_hours/*.json` | Cumulative GPU-hours at every checkpoint (excluding evaluation and time between launches) |
| `results/leakage.json` | Exact and 13-gram overlap of the training set with every eval set and the diag split |

---

## 6. Known approximations and deviations (state these in the paper)

1. **Scoring temperature.** Pilots and support samples are drawn at the training temperature (0.6), but vLLM `prompt_logprobs` score at temperature 1. The tanh estimator therefore measures TV between temperature-1 continuation distributions over a temperature-0.6 support. The paper does not say which it used. Ask whoever wrote the supplement's "likelihood-ratio tanh estimator" sentence.
2. **Terminal pilots.** The paper does not say how to compare two pilots that both finished inside 96 tokens. Here D = 0 if their verifier rewards agree and 1 otherwise; set `terminal_pair_mode: "one"` to use D = 1 for every such pair.
3. **Pilot reuse in the hybrids.** TreePO and TreeRL hybrids do not reuse terminal pilots, because reuse would bypass those frameworks' own bookkeeping. Pilots are pure auxiliary cost there, which is conservative for VDRA.
4. **TreeRL queues.** Queues hold up to 16 forks of one problem rather than mixing prompts, because TreeRL's budget is per problem.
5. **Shared settings applied to TreePO and TreeRL.** These are not native:
    - temperature 0.6 and top-p 1.0 (native: 1.0/1.0 for TreePO, 1.2/0.95 for TreeRL);
    - 2,048-token context, giving 3 × 512-token TreePO segments (native 14 × 512);
    - lr 1e-6, KL 0, weight decay 0;
    - 32 prompts and 4 optimizer steps per update;
    - TreeRL uses `constant_with_warmup` instead of cosine, so runs of different lengths share a schedule. Its warmup is still 5% of each run's length.
    Method-defining parts are native: branching rules, fallback, budget transfer, reward shaping, advantage estimator, loss.
6. **Exp 2 reconstruction.** Roots draw only from the 8 expanded (non-terminal) depth-1 children, so terminal depth-1 children are excluded at the root level. Children are drawn without replacement from finite pools. The Prop. 1–2 prediction includes the finite-population factor and covers depth-1 → depth-2 → leaf edges.
7. **GPU-hours across frameworks** are only comparable within a pair.

---

## 7. A result from the synthetic tests you should know about before the real runs

`tests/test_diagnostics_and_analysis.py` checks Exp 2 against a synthetic bank where all child values are known. The measured edge-averaged error matches the Prop. 1–2 prediction to within about 0.5%, so the measurement is sound.

On that synthetic bank, however, allocating by σ² (even with the *oracle* σ²) gives a slightly *higher* edge-averaged error than uniform: 0.0283 vs 0.0258. The node-value MSE is also slightly higher (0.0084 vs 0.0079). The reason is that the inherited downstream noise τ̄² (leaf-level noise, p(1−p)/6) is as large as σ² there, and Eq. 7 ignores it. This is the reviewers' Eq. 5 vs Eq. 7 concern made concrete.

Whether the real banks behave the same way depends on how σ² compares with τ̄² for MATH reasoning trees, and Exp 2 will show it. If they do, the honest reframing is that VDRA's gains come from somewhere other than lower Eq. 5 error, or that the coefficient should include an estimate of τ̄². Do not promise Pvw4 that Exp 2 will confirm Eq. 7 until you have seen it.

---

## 8. What was and was not tested here

**Tested on CPU:**

- the solver is optimal (checked against brute force on 40 random instances);
- the tanh TV identity and its unbiasedness;
- the TV value-gap bound;
- tail correction and redundancy pruning;
- async queues (capacity, timeout, cross-prompt mixing);
- the full treetune tree builder with stubbed treetune modules and a fake LLM (null test, budgets, value backup);
- Exp 1 and Exp 2 on synthetic banks with known answers;
- the statistics pipeline on synthetic eval data;
- jsonnet configs evaluated against the real igpo configs;
- both upstream patches apply cleanly at the pinned commits;
- the resume machinery:
    - TreeRL's skip/stop/save logic reproduces an uninterrupted run after a crash;
    - run-status detection for all three frameworks;
    - crash-aware GPU-hours;
    - the wrapper restarting a crashing fake trainer, skipping finished runs and continuing to a larger `STOP`;
    - `22_resume_all.sh` job building;
    - the replay buffer surviving a restart;
- every bash driver dry-run.

The CPU suite does not load training models. GPU probes and partial Phase D/F attempts are tracked in [`codex/STATUS.md`](codex/STATUS.md); complete Phase F certification and the full matrix remain pending.

The most likely first-run problems:

- verl/TreePO Hydra keys (`+data.seed`, `custom_reward_function.*`) on that verl version;
- TreeRL's GPU-role layout with 3 GPUs;
- the treetune checkpoint name ↔ step mapping (`ckpt--iter_N` assumed to mean N completed updates);
- the TreeRL resume path on real DeepSpeed and Ray (run `12_resume_test.sh` first);
- whether the treetune episode generator passes the vLLM `api_base` into the strategy's guidance LLM params, as the existing InGPO code assumes.

Run Phase 0 and Phase 1 and read `results/audit/` before committing compute.
