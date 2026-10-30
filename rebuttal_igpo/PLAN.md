# VDRA rebuttal diagnostics in the `igpo` repo: implementation plan for Codex

> Credential-pattern literals are omitted from this published copy so the repository secret scan stays useful; the local pre-commit hook retains the full denylist.

**Your role:** senior ML research engineer (LLM RL, Monte Carlo estimation, experimental design).

**Task:** implement and run frozen-policy diagnostics for the VDRA rebuttal. The code under study is the source snapshot `gear_source_rebuttal` from the artifact archive. It may be synchronized with **https://github.com/Thanh124pav/igpo** (the InGPO / "budget allocation" code), which §1.2 must verify. Hardware: **one RTX 3090 (24 GB)** on a vast.ai instance with about 150 GB of disk.

**Deadline:** 2026-11-01 19:00 UTC+7. No new GPU data generation after 08:00 that day.

**Goal:** defensible measurements of what the trained allocator actually did and how its signal relates to MC value dispersion and to the advantage error of Eq. (5). Do not tune anything to favour VDRA.

---

## 0. Hard rules

1. **No retraining, nothing overwritten.**
   - No RL training and no optimizer states.
   - Never modify checkpoints, run directories, trees, demos or evaluation outputs.
   - All new code goes in `rebuttal/` and `tests/rebuttal/`; all outputs go in `$OUT=rebuttal_out/`.
   - Do not edit `treetune/` training code. Import from it.
2. **No fabricated numbers.** Every number must trace to a file you produced or read. If something is missing, say so.
3. **Rules fixed before results.**
   - Choose checkpoint, iteration and seed rules before looking at results.
   - Never select checkpoints, subsets, seeds or noise levels by outcome.
   - Report results that contradict the paper as clearly as supporting ones.
4. **Independent MC halves.**
   - Split MC references into R_a (used to choose Oracle allocations) and R_b (used to evaluate everything).
   - An MC estimate is a reference, not ground truth.
5. **Statistics and timing.**
   - Use a prompt-clustered bootstrap whenever several nodes share a prompt.
   - RTX 3090 timings are local profiling. Never compare them with historical server GPU-hours.
6. **Run autonomously, without waiting for approval.** The owner is away; nobody will answer questions. Never stop to ask. When a decision is needed, take the most reasonable option, write the decision and the reason to `EXPERIMENT_STATUS.md` → "Decisions", push, and keep going.
   - **Size long jobs from measurements.** Before each long GPU job, run the timed smoke test and write the measured rates and the projected time to `EXPERIMENT_STATUS.md`. Then launch the job automatically if it fits the time budget (§6 schedule, hard stop 2026-11-01 08:00 UTC+7). If it doesn't fit, shrink it (fewer prompts or nodes, smaller R, fewer depths) until it does, record the reduced size and the resulting loss of power, and launch that.
   - **Run long jobs in the background** (`nohup`/`setsid`, logs to file, resumable). Keep working on CPU tasks while they run, and check them at least hourly.
   - **On failure, keep moving.** If a step fails, debug it for a bounded time (about 45 min). If it still fails, mark it "Blocked: <exact reason>" in `EXPERIMENT_STATUS.md`, push, and move to the next item in the priority order. Come back to blocked items only when everything else is done.
   - **Never idle.** When an item can't run, do the next one. When everything that can run has finished, write `RESULTS_SUMMARY.md` and the final report.

---

## 0b. Results repository, credentials, and continuous pushing

All code, Markdown and small results produced by this plan are published continuously to **https://github.com/ihbkaiser/VDRA1** on the branch **`rebuttal`** only. Never push to `main` or any other branch. Two credential files are placed in the working copy's root: **`gh_token`** (GitHub) and **`hf_token`** (Hugging Face).

**The repository is public. Anything pushed is visible to everyone.** A token that reaches GitHub, even in one commit, must be treated as leaked.

### 0b.1 One-time setup (before the first commit)

```bash
cd $WORK && git clone https://github.com/ihbkaiser/VDRA1.git && cd VDRA1
# work only on branch `rebuttal`: track it if it exists on the remote, otherwise create it from main
if git ls-remote --exit-code --heads origin rebuttal >/dev/null; then
  git switch -c rebuttal --track origin/rebuttal
else
  git switch -c rebuttal origin/main        # first push below creates it on the remote
fi
# token files sit in the repo root but must never be tracked
for f in gh_token hf_token; do grep -qxF "$f" .gitignore || echo "$f" >> .gitignore; done
printf 'gh_token\nhf_token\n*.token\n.env\n' >> .git/info/exclude
git check-ignore -v gh_token hf_token            # both must be reported as ignored
# if a token file is somehow already tracked: untrack it locally (git rm --cached), never push, and log it (see "If any token is ever committed")
git ls-files | grep -E '(^|/)(gh_token|hf_token)$' && git rm --cached -q gh_token hf_token 2>/dev/null
chmod 600 gh_token hf_token
git config user.name  "codex-rebuttal-bot"      # or the owner's identity if instructed
git config user.email "codex-rebuttal-bot@users.noreply.github.com"
```

**Pre-commit guard.** Install `.git/hooks/pre-commit`. It must reject:
- any commit that stages `gh_token` / `hf_token`;
- any staged diff that matches `the configured credential-prefix denylist`;
- any staged file larger than 20 MB.

Test the hook once with a dummy staged file before relying on it.

**Push without exposing the token.** Use `GIT_ASKPASS` reading from the file. Do not put the token in the remote URL, `.git/config`, command lines, logs or shell history.

```bash
cat > .git/askpass.sh <<'EOF'
#!/bin/sh
case "$1" in Username*) echo "x-access-token" ;; *) cat "$(git rev-parse --show-toplevel)/gh_token" ;; esac
EOF
chmod 700 .git/askpass.sh
push() {
  [ "$(git branch --show-current)" = "rebuttal" ] || { echo "not on branch rebuttal - refusing to push"; return 1; }
  GIT_ASKPASS="$PWD/.git/askpass.sh" GIT_TERMINAL_PROMPT=0 git push -u origin rebuttal:rebuttal
}
```

**Hugging Face.** Use the token only inside processes, via `export HF_TOKEN="$(cat hf_token)"` inside the subshell that downloads or uploads. Never `echo` it, and never write it to logs or outputs. `huggingface-cli login --token "$(cat hf_token)"` is acceptable; it stores the token under `~/.cache/huggingface`, outside the repo.

### 0b.2 Layout inside VDRA1

The existing top level holds the earlier B200/Modal work: `analysis/`, `codex/`, `diagnostics/`, `integrations/`, `modal_app/`, `remote_artifacts/`, `scripts/`, `tests/`, `vdra_core/`. Leave it untouched unless a task requires a change there.

Put this plan's work under one new folder:

```
rebuttal_igpo/
  PLAN.md                 # a copy of this plan (keep it in sync if the plan is updated)
  STATUS.md               # = EXPERIMENT_STATUS.md, updated at every push
  docs/                   # ARCHIVE_INVENTORY, RUN_PROVENANCE, ARTIFACT_MATRIX, MECHANISM_AUDIT,
                          # DERIVATION_EQ5_EQ7, COMPUTE_AND_CONSISTENCY_AUDIT, EXPERIMENT_COMMANDS,
                          # REBUTTAL_EXPERIMENT_PLAN, RESULTS_SUMMARY
  rebuttal/               # the code of section 3 (import-only use of the igpo checkout)
  tests/                  # tests/rebuttal of section 6
  configs/                # section 5 server configs
  results/<experiment>/   # CSV, JSON summaries, figures (PNG/PDF), small logs
  results/MANIFEST.md     # every large output kept elsewhere: path, size, sha256, where stored
```

The igpo code itself stays in its own checkout. Record the igpo commit hash in `RUN_PROVENANCE.md`; do not vendor the whole repo.

### 0b.3 What goes to GitHub and what does not

- **Push:**
  - code, tests and configs;
  - all Markdown;
  - result CSV/JSON summaries;
  - figures;
  - status files;
  - small logs (under 5 MB each);
  - smoke-test measurements.
- **Never push:**
  - token files;
  - checkpoints;
  - `rebuttal_output.zip` or anything extracted from it;
  - node-bank `jsonl.gz` files over 20 MB;
  - vLLM caches;
  - raw tree dumps;
  - anything containing full model generations at scale.

  Large outputs stay on local disk and are listed in `results/MANIFEST.md` with their sha256. Do not upload them anywhere, and do not create new HF or GitHub repositories.
- **Before every push,** also check `git diff --cached --stat` for unexpected files.

### 0b.4 When to push

Commit and push:
- at every completed unit of work: each audit document, each module once its tests pass, each experiment's results, and each update to the plan;
- at least every 2 hours while long jobs run, with `STATUS.md` updated (running job, progress, measured rates, ETA).

**Each push:**
1. `git pull --rebase origin rebuttal` (skip it until the branch exists on the remote). Do not merge `main` into `rebuttal`.
   - If the rebase conflicts, `git rebase --abort`, then `git pull --no-rebase origin rebuttal` and resolve by keeping both versions (rename yours `*.codex.md` if needed). Never block on a conflict.
   - If a push fails (network, auth), keep committing locally, retry on the next push cycle, and record the failure in `STATUS.md`. A failed push never stops the experiments.
2. `pytest -q rebuttal_igpo/tests`. Push code only when the tests pass; status-only pushes may skip the tests.
3. Commit with a descriptive message: `A1: allocation statistics on ingpo-tree-666 iters 0-235`, `status: A3 bank 112/224 nodes`, and so on.
4. `push`.

**Never:** force-push, rewrite published history, push to `main` or any branch other than `rebuttal`, open pull requests, or push to other repositories.

**If any token is ever committed or pushed,**
- stop all further pushes for the rest of the session (rewriting history does not undo a leak in a public repository);
- write `ALERT_TOKEN_LEAK.md` in the local repo root with the commit hash and which token, and put the alert at the top of the final report so the owner revokes the token on waking;
- keep doing the experiments locally; only publishing stops.

If a token was committed locally but not pushed: remove it from the commit (`git reset --soft HEAD~1`, unstage, recommit) and continue normally.

**Before the first push,** remind the owner in `STATUS.md` that the repository is public. Pushing rebuttal results to a public, named repository during double-blind review may compromise anonymity. Do not change repository visibility yourself.

---

## 1. Step 0: instance, artifacts, code source, environment

**Instance:** vast.ai with an RTX 3090 and about **150 GB free on `/`**, which is enough for full extraction. Do not run any further disk-check or selective-extraction procedure. Record `df -h`, `free -g` and `nvidia-smi` once in `ARCHIVE_INVENTORY.md`, and keep a 10 GiB free-space floor.

### 1.1 Artifacts: full extraction, streamed

- **Source:** `rebuttal_output.zip` (39.4 GB, stored without compression) in the HF dataset `ihbkaiser/qwen25vl-3b-msd-checkpoints`. It contains six nested archives under `rebuttal_output/`:

  | Member | Size | Contents |
  | --- | --- | --- |
  | `vdra_iter_0235_final.tar.zst` | 26.3 GB | VDRA run: config, logs, demos, trees, final checkpoint |
  | `spo_iter_0235_matched.tar.zst` | 3.5 GB | SPO run, matched at iteration 235 |
  | `gear_source_rebuttal.tar.zst` | 0.4 GB | **source code snapshot** (§1.2) |
  | `vdra_iter_0150.tar.zst` | 3.5 GB | iteration-150 checkpoint |
  | `vdra_iter_0050.zip` | 2.8 GB | iteration-50 HF checkpoint |
  | `vdra_iter_0100.zip` | 2.8 GB | iteration-100 HF checkpoint |

- **Never store the outer zip.** For each member:
  - Read its byte range from the central directory with `remotezip` (HTTP range requests, Authorization header read from `hf_token`).
  - `*.tar.zst`: stream `curl -r <start>-<end> | zstd -d | tar -x -C $WORK/src/<member>/`. Write the member listing with sizes to `rebuttal_igpo/docs/listing_<member>.txt`. Skip optimizer and DeepSpeed states (`global_step*`, `*optim*`, `zero_*`, training-state `*.pt`).
  - `*.zip` (iterations 50/100): stream to `$WORK/tmp`, extract to `$WORK/src/<member>/`, delete the temp file.
- **Extraction order:**
  1. `gear_source_rebuttal`;
  2. `vdra_iter_0100`;
  3. `vdra_iter_0235_final`;
  4. `spo_iter_0235_matched`;
  5. `vdra_iter_0150`;
  6. `vdra_iter_0050`.

  Start the CPU work (§1.2, A0, A1, A6) as soon as the members it needs are on disk; don't wait for the rest.
- **Afterwards:**
  - `chmod -R a-w $WORK/src`.
  - Write `ARCHIVE_INVENTORY.md` with the tree (depth 3), sizes, checkpoints (weights, tokenizer and config present?) and stored trees per run (`temp_episodes/iteration__*/infer_results/*/traj_results_ds`), demos (`gear_demos/` or `ingpo_demos/`), `merged_edges`, `episodes/merged`, logs and evaluation folders.
  - Map each item to what the plan expects and list anything missing.

### 1.2 Code: `gear_source_rebuttal` is the primary source; igpo GitHub is the reference

The owner believes the snapshot `gear_source_rebuttal` is synchronized with https://github.com/Thanh124pav/igpo, but **this is not confirmed**. Establish it before relying on either:

1. **Identify the snapshot.**
   - Its top-level layout; whether it contains `.git` (if so, record `git log -1` and `git status`).
   - Whether its code uses `gear_*` / `vdra_*` names (the newer VDRA "gear" code, e.g. `treetune/gear/`, `vdra_core/`, `allocation_mode`, `use_redundancy_cap`, `allocation_proxy='vdra'`) or `ingpo_*` names (`treetune/ingpo/`, the older InGPO budget allocator).
2. **Compare it with igpo GitHub.** Clone igpo with all branches. Then:
   - for `main`, `origin/fix/budget-allocation-theory` and every other branch, plus the commit nearest the snapshot's file dates, compute `git diff --stat <ref> -- <paths>` against the snapshot (use `diff -rq` / `git diff --no-index` for a snapshot without `.git`);
   - report the closest ref and the exact differing files;
   - write `CODE_SYNC_REPORT.md` with a table: file → identical / differs (summary of the change) / only in snapshot / only in igpo. Cover at least the allocator, scheduler/queues, TV estimator, inference strategy, episode generator (edge extraction, filtering), trainer loss/whitening, and the configs used by the runs.
3. **Decide which code each run used.** Compare the field names and resolved settings in each run's `config.json` / log header (§1.3) with the snapshot and with the igpo refs.
   - The code whose parameter names and defaults reproduce the run's resolved config is the run's code.
   - If the snapshot matches, use it as the **primary code** for the audit (§2) and for all replays.
   - If it does not, use the matching igpo commit, and say so.
   - If neither matches exactly, use the closest one and list every mismatch in `RUN_PROVENANCE.md`.
4. **Keep the snapshot untouched.** It stays read-only. Import from it through `PYTHONPATH` or a copy in `$WORK/code/<name>` — the code under test — and never modify it. New code lives only in `rebuttal_igpo/rebuttal/`.
5. **§2 is a hypothesis about igpo `main`.** The mechanism table in §2 was written from igpo `main` and the fix branch. Re-verify every row against the code chosen in step 3. If that code is the newer gear/VDRA code, map each row to the gear equivalent: `vdra_core/allocation.py::allocate_rollouts`, `treetune/gear/budget_allocation.py::allocate_vdra_branch_factors`, `treetune/gear/online_budget.py`, `treetune/gear/tv_estimators.py`, `treetune/inference_strategies/gear_inference_strategy.py`, `configs/ablations/*`. Mark rows that do not apply.

### 1.3 Run provenance

For every run in `$WORK/src` (VDRA iterations 50/100/150/235 and SPO iteration 235, plus any other run directories found):
- read `config.json`, `metadata.json`, `dotenv.txt`, the log header and the file dates;
- record the fully resolved allocation settings (`gear_*` or `ingpo_*`): allocator mode, weight mode, candidate selection, `n_min`, cap, queues, TV settings, γ, pilots, `skip_near_leaf_expand`, root allocation;
- write `RUN_PROVENANCE.md`: run → code source (snapshot / igpo ref) → resolved settings → differences from the paper's description.

Note in particular whether the VDRA run used the pre-fix InGPO allocator (weights ∝ σ², floor rounding, top-logprob candidate reuse) or the VDRA bounded solver.

### 1.4 Environment

- **Use what is installed.** `/venv/main` already has torch with CUDA 12.8. Install vLLM matching that torch if possible; otherwise use a separate venv. Install with `--no-cache-dir`, then `pip cache purge`.
- **Install the run's code.** Use the requirements of the code chosen in §1.2, installed into that environment.
- **Verify:**
  - `torch.cuda.get_device_capability()` returns (8, 6);
  - BF16 works;
  - the repo's vLLM server script (`scripts/start_vllm_server.sh` in igpo or its gear equivalent) serves the iteration-100 checkpoint with prompt logprobs, as needed by the logprob scorer.
- Record versions and any deviation in `EXPERIMENT_STATUS.md`.

---

## 2. Facts to verify in code (A0, CPU)

These come from reading `main` and the fix branch. Verify each one, cite file and function, and record it in `ARTIFACT_MATRIX.md` / `MECHANISM_AUDIT.md`. Mark each "confirmed / different in run commit".

| # | Mechanism | Location | Behaviour on `main` (pre-fix) | Fix branch |
| --- | --- | --- | --- | --- |
| M1 | Variance proxy | `treetune/ingpo/tv_estimators.py::ConditionalTVEstimator`; `treetune/ingpo/budget_allocation.py::reward_variance_from_pair_tvs`, `simulation_lemma_gap` | Builds a cached logP(second | first) matrix and softmaxes each row over the sampled support. Pairwise TV = Σ\|p−q\|, halved if `tv_includes_half_factor`. Gap ψ = γ·TV/((1−γ)(1−γ+TV)). σ² = Σ_pairs 2ψ²/(2n(n−1)). With γ = 0.9, ψ can exceed 1, so σ² is not bounded by 0.25. | half factor default True |
| M2 | Pilots (hierarchical mode, `n_tv_estimates=8`) | `_generate_samples` | first_count = ⌈√8⌉ = 3 first-phase prefixes (`tv_subnode_max_tokens` tokens), up to 3 second-phase continuations each (`tv_second_phase_tokens`), at most 8 samples. First-phase prefixes that finished early are kept as candidates but get no second phase. | unchanged |
| M3 | Allocation weight | `allocate_branch_factors` | weight = √(σ⁴ − λ). With λ = 0 this is **σ²**, so n_v ∝ σ_v². The minimizer of Σσ²/n under Σn = B is n_v ∝ σ_v (Neyman), which is Eq. (7)'s optimum. | `allocation_weight_mode='std'` → √σ² |
| M4 | Rounding and budget loss | same | Floor rounding, leftover dropped (`underallocated_budget`). Nodes with weight 0 get 0 children when `n_min=0`. | largest remainder, `n_min=1`, uniform fallback |
| M5 | Upper bound / redundancy cap | same, plus `_construct_budget_allocated_tree` | **No upper bound.** n_v can exceed the default branch factor 6. Candidates are deduplicated only by identical prefix text, not by TV clustering. | unchanged |
| M6 | Queues | `treetune/ingpo/budget_scheduler.py::FlexibleBudgetScheduler.allocate` | Synchronous, per tree and depth: the frontier is split across `queue_count` queues (least-loaded). Queue budget = ⌊total_depth_budget·\|q\|/N⌋. Writes `node["ingpo_budget_queue_id"]`. | budget-conserving split |
| M7 | Children = reused pilots | `_construct_budget_allocated_tree` | Candidates are sorted by `sum_logprobs` descending and the **top-n are reused as children**; extra children are generated only if n_v > the number of candidates. Children are therefore not i.i.d. policy samples, and the value estimate is biased toward high-likelihood prefixes. | `candidate_selection='random'` |
| M8 | Root allocation | `_prepare_tree_construction_context` | With `root_allocation=true`, the depth-0 budget is allocated across all roots of the minibatch shard (log line "InGPO root_allocation allocated depth-0 budget…"). | – |
| M9 | Last expansion | `skip_near_leaf_expand` | Depth `max_depth−1` uses uniform B with no TV. | – |
| M10 | Advantages | `treetune/episode_generators/ingpo_episode_generator.py::extract_edges_from_tree` (+ `tree_episode_generator.py`) | A = child_reward − parent_reward (or a normalized variant). `zero_advantage_when_pruned`. Edges with all-zero PAV advantage may be dropped. | – |

**Also check:**
- What happens to a node allocated 0 children: its reward and value, and whether its incoming edge is emitted.
- Whether any batch whitening is applied in the trainer.
- How `traj_results_ds` iteration t maps to the checkpoint that generated it (`on_policy_episode_generator.py`, `latest_policy_path`).

**Stored artifacts to inventory per run:**
- `temp_episodes/iteration__XXXX/infer_results/process_YY/traj_results_ds`: HF dataset, column `_treetune__reasoning_tree`. Node fields include `ingpo_segment_id`, `ingpo_depth`, `ingpo_reward_variance`/`ingpo_sigma2`, `ingpo_allocated_branch_factor`, `ingpo_budget_weight`, `ingpo_budget_queue_id`, `ingpo_budget_candidates`, `ingpo_discarded_budget_candidates`, `ingpo_tv_logp_matrix`, `ingpo_tv_pair_count`, `ingpo_tv_support_size`. Tree fields include `ingpo_*_by_depth` and `ingpo_root_*`.
- `infer_results/merged_edges`: edges with `advantage`, `value`, `ingpo_action`.
- `episodes/merged`: the 512 training episodes, with `advantages`.
- `ingpo_demos/demos.jsonl`: per-tree stats; budget, timing and per-depth fields.
- `training_timing.jsonl`, `checkpoints/`, `evaluation/`.

Existing scripts to reuse: `scripts/inspect_tree.py`, `inspect_demos.py`, `aggregate_stats.py`, `oracle_audit.py`, `start_vllm_server.sh`, `evaluate.sh`, `evaluate_long_cot.sh`.

---

## 3. Code to write (`rebuttal/`, importing from the run's code chosen in §1.2)

```
rebuttal/
  io.py            # load trees (traj_results_ds), edges, episodes, demos, timing; normalize ingpo_* / gear_* / vdra_* fields to one schema
  alloc_replay.py  # allocators + bound regimes, CPU
  alloc_stats.py   # A1 from stored trees
  node_bank.py     # GPU: frozen-policy pilots + TV + MC references (resumable)
  metrics.py       # J(n), regret, gap recovery, Spearman, parent-value MSE, Eq.5 losses, clustered bootstrap
  cost.py          # tokens / GPU-seconds / VRAM per signal (operational protocol)
  eval_avgk.py     # A7 wrapper around the repo's evaluator
  sim_eq5_eq7.py   # synthetic checks of the identities
  cli.py           # one entry point: `python -m rebuttal.cli <subcommand> ...`
tests/rebuttal/    # CPU tests with tiny synthetic ingpo-format trees
```

### 3.1 `io.py`
- `iter_trees(run_dir, iters=None, max_per_iter=None)`: find every dataset under `temp_episodes/iteration__*/infer_results/**` that has `_treetune__reasoning_tree`. Yield `(iteration, shard, idx, tree)`.
- `walk(tree)`: flatten to node records with depth, parent, `n_children`, reward, and all `ingpo_*` / `gear_*` / `vdra_*` scalars (whichever the run's code wrote). Drop the large logp matrices (`ingpo_tv_logp_matrix`, `gear_tv_logp_matrix`) unless asked for them.
- `load_demos`, `load_timing`, `load_edges`, `load_episodes`. All of them stream; never load every iteration into RAM.

### 3.2 `alloc_replay.py`

A pure-Python, deterministic allocator interface: `allocate(coeffs, lower, upper, budget, rule) -> counts`.

| Rule | Definition |
| --- | --- |
| `as_trained` | Calls the allocator of **the run's code** (§1.2), with the run's resolved settings. For pre-fix InGPO this is `treetune.ingpo.budget_allocation.allocate_branch_factors` (∝ σ², floor rounding, budget loss); for gear/VDRA code it is `allocate_vdra_branch_factors` → `vdra_core.allocation.allocate_rollouts`. |
| `fixed_branch` | Fix-branch rule: √σ², largest remainder, `n_min=1`, uniform fallback. Re-implement it, or import from a second worktree. |
| `eq7_exact` | Exact minimizer of Σ C_v/n_v with integer bounds and Σn = B. Greedy marginal gain C/n − C/(n+1) is exact for this separable convex objective; test it against brute force on small cases. |
| `uniform` | Water-filling under the same bounds and budget. |
| `random_perm` | Shuffle another rule's counts within each queue, then repair bound violations and log the repairs. |
| `entropy`, `nll`, `raw_tv`, `vdra_lambda`, `oracle` | Coefficient choices fed to `eq7_exact`: entropy, NLL and raw TV (mean pairwise TV without ψ) come from the bank; `vdra_lambda` uses C_u·λ_u with λ_u = (n_{p(u)} − 1)/n_{p(u)}; `oracle` uses σ̂² from R_a. |

**Bound regimes:**
- `trained`: no upper bound; queues and budgets exactly as M6.
- `capped`: n_v ≤ default (6).
- `capped_k`: n_v ≤ min(6, k_v), where k_v is the number of distinct candidates.

### 3.3 `alloc_stats.py` (A1)

**Per run, per depth, per training stage** (iteration bins, e.g. 0–49, 50–149, 150–235):
- histogram and quantiles of n_v;
- fraction with n_v = 0, n_v = 1 and n_v > 6;
- Gini / HHI;
- budget utilization (allocated/requested, built/requested);
- floor loss per queue;
- queue sizes (from `ingpo_budget_queue_id`);
- children from reused pilots vs freshly generated (`len(candidates)` vs `allocated`, and `ingpo_discarded_budget_candidates`);
- the rank of the selected pilots by `sum_logprobs`, to quantify the M7 selection;
- edges in `merged_edges` / `episodes`, and the share with zero advantage.

**Counterfactual on stored C (CPU), per queue:**
- re-allocate the same stored `ingpo_reward_variance` with `fixed_branch` and `eq7_exact` under the same queue budget;
- report the share of rollouts the trained rule placed differently;
- report the proxy objective Σ C/n under each rule, to show how far the trained allocation was from the Eq. (7) optimum on its own signal.

**Outputs:** `allocation_statistics.csv`, `queue_level.csv`, `fig_A1.png` (3 panels: n_v by depth, budget loss / n_v = 0 share by stage, trained vs Eq. (7)-optimal distance).

### 3.4 `node_bank.py` (GPU, resumable)

**Inputs:** checkpoint, prompts, depths, number of prompts, seed, output path.

**Parent states:**
- Option (a): reuse stored tree nodes from the run at the iteration matching the checkpoint, after verifying the mapping in §2.
- Option (b): regenerate depth-1 states with the repo's `node_expander` and the run's segment settings.
- Default to (a) where it is valid. It is cheaper and is exactly the training distribution.

**Per parent:**
- **Proxy:** run the exact training proxy with `ConditionalTVEstimator` (mode, `n_tv_estimates`, horizons, `tv_includes_half_factor`, γ from the run config) through `make_lp_scorer(VLLMLogprobClient)` against a `scripts/start_vllm_server.sh` server. Store σ², the pair TVs, the logp matrix, the candidates, and the pilot entropy and NLL.
- **MC references:** R = 64 full continuations from the parent (32 + 32 halves), using the run's generation parameters. Score them with the task's own `reward_function`, the same function as training.
- **Child values (A4/A5):** for every candidate child (pilot prefix), R_c = 16 continuations to completion. These children double as the fixed held-out evaluation edges.

**Format:** stream `bank.jsonl.gz`, one record per node, and resume by node id. Log tokens/s, peak VRAM, seconds per pilot, per continuation and per scoring call.

### 3.5 `metrics.py`, `cost.py`
- J(n) = Σ σ̂²_{R_b}/n; % vs uniform; regret vs Oracle; Oracle-gap recovery; Spearman correlation with σ̂²_{R_b}; parent-value MSE (subsample n draws from R_b against the R_a mean).
- **Eq. (5) quantities:**
  - L_node = mean_v E_v;
  - L_edge = (1/M) Σ n_v E_v, where M = Σ n_v changes with the allocation;
  - L_heldout = mean over the fixed candidate-child edges of (Â − A_ref)².
- Prompt-clustered bootstrap with 1,000 replicates.
- **Operational cost per signal:** generated tokens, scored tokens and GPU-seconds. Pilots cost the same for entropy, NLL and TV, while scoring and second-phase tokens are TV-only. Uniform costs 0.

### 3.6 `eval_avgk.py` (A7)
- Wrap `scripts/evaluate_long_cot.sh` / `evaluate.sh`. Check whether sample-count overrides exist; branch `agent/checkpoint-sweep-evals` has "Allow eval sample count overrides". Prefer reusing that commit's mechanism.
- Protocol: 16 samples, temperature 0.6, top-p 0.95, fixed seeds, the repo's verifier. Report avg@16 and Pass@16 separately, with paired prompt-level bootstrap.

### 3.7 `sim_eq5_eq7.py`
Synthetic fixed trees with i.i.d. continuations. Numerically check:
- E[n_p S_p | C(p)] = ((n_p − 1)/n_p) Σ_{u ∈ C(p)} τ_u²;
- τ_u² = (σ_u² + τ̄_u²)/n_u;
- the decomposition term Σ_{u≠root} λ_u σ_u²/n_u.

Then show quantitatively how M7 (top-logprob pilot reuse) and adaptive allocation break them. Cover the root case and n_p = 1 (λ = 0).

---

## 4. Experiments (in order)

Each experiment section in `REBUTTAL_EXPERIMENT_PLAN.md` states: hypothesis, estimand, unit of analysis, checkpoint, prompts, protocol, baselines, constraints, seeds, references, metrics, CIs, runtime, VRAM, outputs and interpretation limits.

| ID | What | Compute | Depends on |
| --- | --- | --- | --- |
| A0 | §1–2: provenance, mechanism audit, artifact matrix | CPU, ~3–5 h | – |
| A6 | Compute and consistency audit from `demos.jsonl` (`variance_estimation_seconds` vs `expansion_seconds`, budget fields), `training_timing.jsonl`, and generated / scored tokens. Check the paper vs code/log items: Fig. 2a vs Table 1, Eq. 7 bound vs uncapped n > 6 (M5), GPU-hour span vs 1.20×, ε_tail, the AIME24/25 rows, 2.46 vs 2.48, DP vs greedy solver, rollout budget vs total sampling cost, L = 512, the OlympiadBench subset, **which commit and settings produced each Table 1/4 row**. Classify each item; never guess. | CPU, ~2–3 h | A0 |
| A1 | Allocation statistics + counterfactual (§3.3) on all stored trees of the VDRA runs | CPU, ~1–3 h (JSON parsing) | A0 |
| T | `sim_eq5_eq7.py` + `DERIVATION_EQ5_EQ7.md` (exact identity / identity under assumptions / local surrogate / unproven / needs experiment) | CPU, ~2–3 h | – |
| S | Smoke test: 8 prompts, depth 0–1, iteration-100 checkpoint. Measure rates and VRAM, write the projected times, then size and launch A3/A5 automatically (rule 6). | GPU, ~0.5 h | §3.4 |
| A3 | Bank: 32 prompts, depths 0–1 (~224 parents), iteration 100. Replay every allocator under the controlled protocol (same bounds and budget) and the operational protocol, in all three bound regimes. | GPU ~1–2 h + CPU | S |
| A2 | Noise on the A3 bank: C′ = C·exp(ε), ε ~ N(0, s²), s ∈ {0, 0.25, 0.5, 1, 2}, 50 repetitions. Report the measured curve. | CPU | A3 |
| A5 | Eq. (5) on the A3 bank + child values: L_node, L_edge, L_heldout for uniform / as_trained / fixed_branch / eq7_exact / vdra_lambda / oracle | GPU ~1.5–2.5 h | A3 |
| A4 | TV vs value gap per candidate pair: binned calibration, Spearman, top-quartile overlap, failure cases, reference-noise bands. Check that the verifier depends only on the completed text. | CPU on the A5 data | A5 |
| A7 | avg@16 / Pass@16 for SPO and VDRA at the iteration rule fixed in §0, on AIME24/25 + AMC23 | GPU ~1 h at 4K, ~3–5 h at 32K | starts automatically once A3/A5 are done, if it fits before the hard stop (else 4K only) |
| X | Iterations 50 and 150 for A3/A5 | GPU +5–8 h | starts automatically after A7, if it fits before the hard stop; otherwise run iteration 150 only, or skip and record it as deferred |

The GPU times are estimates for vLLM BF16 at about 3–5k generated tokens/s. Replace them with the measured rates from S. Without vLLM, multiply by 5–10.

**Interpretation boundaries:**
- A lower J, L_node or L_heldout is not proof of a lower full clipped-PPO gradient error.
- The as-trained allocator (M3/M4/M7) is not the Eq. (7) solver. Report the two separately, and never attribute fix-branch or `eq7_exact` results to the trained runs.

---

## 5. Configs for later server runs (write only; do not train)

**`configs/rebuttal/`**: a 2×2 grid at equal budget, using one commit (the fix branch) and one seed list:
- Uniform / no cap;
- Uniform + cap: water-filling that keeps the budget instead of dropping it;
- VDRA / no cap;
- VDRA + cap.

Plus:
- an "Eq. (7)-exact" variant (`eq7_exact` weights with largest-remainder rounding);
- a random-candidate vs top-logprob ablation (M7);
- a leave-one-out baseline variant for the advantage;
- a short README with the launch commands (`scripts/_common.sh` / `ingpo_run`).

Log rollouts and generated tokens per update in every cell.

---

## 6. Tests (CPU, `pytest tests/rebuttal`)
- `io` on a synthetic ingpo-format tree saved as an HF dataset and as JSON.
- `as_trained` reproduces the stored allocations (`ingpo_allocated_branch_factor` or `gear_allocated_branch_factor`) exactly on stored queues. This regression test checks that the replay is faithful. If it fails, spend up to about an hour finding out why (wrong commit, missing parameter, queue reconstruction). If it still fails, continue anyway: label every replay result "replay not verified against stored allocations (match rate X%)", report the match rate, and keep going.
- `eq7_exact` matches brute force on small cases.
- Water-filling, permutation feasibility, bootstrap, MC half independence, bank resume.
- `sim_eq5_eq7` identities within MC tolerance under the i.i.d. assumptions.

---

## 7. Deliverables (`rebuttal_out/` locally; published under `VDRA1/rebuttal_igpo/` per §0b)

- **Audit:** `ARCHIVE_INVENTORY.md`, `RUN_PROVENANCE.md`, `ARTIFACT_MATRIX.md`, `MECHANISM_AUDIT.md` (§2 table, confirmed per run)
- **Plan and status:** `REBUTTAL_EXPERIMENT_PLAN.md`, `EXPERIMENT_COMMANDS.md` (real paths; checkpoint, data, config, output, seed, batch, resume, footprint), `EXPERIMENT_STATUS.md` (Ready / Blocked + blocker / Running / Completed / Deferred, plus measured rates)
- **Analysis:** `DERIVATION_EQ5_EQ7.md`, `COMPUTE_AND_CONSISTENCY_AUDIT.md`
- **Results:** `allocation_statistics.csv`, `queue_level.csv`, bank files, `A2/A3/A4/A5` result CSVs + figures, `RESULTS_SUMMARY.md` (numbers with CIs and file pointers, measured evidence separated from planned work, contradicting results included)

## 8. Final response

Write it in Vietnamese; paths and commands in English. Include the URL of the last pushed commit, confirmation that `gh_token` / `hf_token` are untracked and appear nowhere in history (`git log --all -p | grep -E 'the configured credential-prefix scan'` returns nothing), and:

1. The §2 table, with confirmed / different for the run commit.
2. The top three experiments to run on the 3090.
3. The exact first command.
4. What must go to the server.
5. What can finish before the deadline.
6. What to defer to the camera-ready version.
