# Experiment Audit Report

**Date:** 2026-10-27  
**Auditor:** gpt-6.1-sol, ultra; fresh same-family reviewer, read-only, provisional  
**Project:** VDRA rebuttal experiments

## Overall Verdict: WARN

## Integrity Status: WARN

**Phase F certification: FAIL.** R1?R6 local dry runs pass command/config construction only. Earlier B200 attempts left incomplete durable evidence. The latest user-authorized R3-R6-only submission used profile `billionaireproject8` in workspace `kieusontung6`; R3 remained pending for 3,900 seconds without a container, then timed out. No GPU smoke ran, and the pending call was terminated. Current metered usage is $38.23260592 against a $37.95116494 workspace baseline; billed cost is $0 after credits. The $27 project-local additional cap remains active.

## Checks

### A. Ground Truth Provenance: PASS

Evaluation targets come from dataset answer fields or boxed dataset solutions in `analysis/datasets_util.py`. Missing, blank, empty-list, and `[None]` targets fail closed. Predictions are scored against those dataset targets by `analysis/eval_checkpoints.py`. The known exact and notation-equivalent train/evaluation overlaps remain disclosed in `codex/leakage.audit.json` and `codex/STATUS.md`; the benchmarks should not be described as entirely uncontaminated.

### B. Score Normalization: PASS
Benchmark outputs retain binary `correct`, raw `k_correct`, and `k`; the reviewer found no normalization by prediction extrema. The recovered IGPO null audit exactly reproduces the saved mismatch for IDs 2117 and 35. Recomputed R2-TV evidence shows complete queue-budget evidence and zero budget violations, but CV 0.1768768 remains below the declared 0.3 headroom threshold.
### C. Result Existence and Completeness: WARN
Recovered R1 and constant-proxy R2 tree logs reproduce the failed IGPO null equality for IDs 2117 and 35. Recovered R2-TV queue/node/tree logs reproduce 50 queues, 54 nodes, complete evidence, zero budget violations, and CV 0.1768768; the 0.3 headroom gate still fails. The remote volume contains an HF step-4 checkpoint and a roughly 3.55 GB weight file, but the file was not downloaded or fully hashed. The old R2 trainer was cancelled; later Modal calls reused artifacts without training. The strict validator rejects that run.

The Phase F remainder runner now starts a fresh R2-TV recovery in an isolated run root if no report passes the strict validator. It keeps R1/constant-R2 baselines under the earlier run tag and proceeds to R3?R6 only after successful trainer, timing, checkpoint, and queue-budget checks. The R2 recovery timeout is 10 minutes, exceeding the observed 209.02-second smoke. The end-to-end 55-minute reserve is $5.9788 including its $0.25 non-GPU cushion.

R3's first B200 trainer reported `rc=0`, but its Modal launcher failed with code 1 and no checkpoint, timing, or queue artifacts persisted. Its second B200 attempt ended in `KeyboardInterrupt` after 226.59 seconds. R4 records repeated launcher failures or interruptions without durable update evidence. R5 has no Phase F volume directory. R6 failed with launcher errors; the retained retry log shows an MCTS prompt-logprob OOM before update 1. A newer app did run R3, so the earlier statement that all deployment attempts were rejected before app start was incomplete. The aggregate report was not refreshed for the second R3 retry.

The current Phase D validator requires all five benchmarks with 16 unique rows each, row identities, sidecars, and result-byte digests. It recomputes the expected signature from the current checkpoint, exact dataset rows, tokenizer, verifier, and protocol for ordinary step 0/2/4 checks and separately for both ordinary and resumed step-4 evaluations. Step 0 is pinned to the base model even if a saved step-0 checkpoint exists, matching `eval_pending.py`. Regression tests cover changed checkpoint inventory, changed dataset answers, stale well-formed signatures, and the step-0 path.
### D. Dead Code and Output Persistence: PASS

Shared evaluation is wired through `scripts/20_train.sh`; `analysis/eval_checkpoints.py` writes result rows and sidecars. Phase F null-audit failures are retained in aggregate reports, and aggregate completion now requires every expected run label and rejects failed audit executions. Phase F disables benchmark evaluation, so its artifacts support smoke/instrumentation claims, not held-out accuracy claims. Both budgeted R3-R6 launcher scripts now parse `--help` before budget checks or Modal calls, with regression coverage for each.

### E. Scope: WARN

Evidence is limited to one model revision, seed 42, and tiny/partial smokes. No completed three-seed matrix, 20-update pilot, compute-matched study, or paired end-to-end B200 speedup is established. B200 vLLM telemetry supports a controlled `max_num_seqs=96` trial but does not establish a speedup or justify changing the shared default.

### F. Evaluation Type and Cache Provenance: WARN

- Benchmark evaluation: `real_gt`.
- TV, pilot-NLL, and diversity allocation quantities: model-derived/self-supervised proxies.
- Null-tree and resume comparisons: behavioral consistency checks, not independent benchmark performance.
- Fake-trainer/fake-evaluator and synthetic-bank tests: simulation/synthetic-only.
- Human evaluation: none evidenced.

Evaluation cache signatures bind checkpoint inventory, exact dataset rows, tokenizer asset fingerprints, active verifier implementation fingerprints, prompt, and settings. Sidecars bind row counts and result-byte digests. Phase D independently recomputes those expected signatures before accepting ordinary/resumed result files. Checkpoint identity remains an inventory fingerprint: files larger than 1 MiB use size/mtime rather than full content hashes.

## Local Verification
- Full local suite after the Phase F orchestration changes: `py -3.12 -m pytest -q tests` ? **186 passed, 1 skipped, 2 deprecation warnings**.
- Focused local checks after the Phase F recovery orchestration and detach changes: **29 passed, 2 deprecation warnings** across `tests/test_phase_c_paths.py`, `tests/test_phase_f_runner_cli.py`, and `tests/test_phase_d.py`.
- R1?R6 dry-run preflight emitted `PREFLIGHT_OK` for all six; this covers command/config construction only.
- The recovered null and R2-TV audits recompute exactly from the saved raw logs. The earlier updated 55-minute budgeted runner blocked before invoking `modal run`.
## Action Items
- Obtain B200 scheduler availability and verify workspace headroom before retrying Phase F; the latest R3 call timed out in the scheduler before container allocation.
- Rerun the R3-R6-only smokes when B200 capacity is available, preserve raw outputs, then resolve or explicitly retain the failed IGPO null-equality and R2-TV headroom gates.
- Run a matched Phase D IGPO pair at `max_num_seqs=96` against the fixed 64 setting, then compare speed and verify parity before changing the shared default.
- Keep Phase D signature-parity checks, 20-update pilots, compute-matched comparisons, and the full matrix pending until artifacts prove completion.
## Claim Impact

- Dataset-based benchmark grading and non-prediction-derived accuracy: **supported**.
- ?All R1?R6 runs work?: **unsupported**; all six local dry runs pass preflight, but GPU evidence is incomplete.
- ?IGPO and VDRA are allocation-equivalent under the null?: **unsupported**; recovered raw trees reproduce the failed equality gate.
- ?R2-TV meets the headroom gate?: **unsupported**; recovered raw queues reproduce zero budget violations, but CV 0.1769 is below threshold.
- ?End-to-end training speedup achieved?: **unsupported**; existing component microbenchmarks do not replace a matched B200 result.
- Comprehensive three-seed or compute-matched conclusions: **unsupported**; planned runs remain incomplete.

## Post-audit operational update - 2026-10-27

- The user approved a $28 total additional cap for profile `billionaireproject8`; the project guard now has a $26.60 safety stop. Metered usage remains $87.10290662, or $19.93099707 above the recorded baseline.
- The IGPO one-update profile path now accepts `max_num_seqs` and an isolated run tag. `py -3.12 -m pytest -q tests/test_phase_d.py` passes 17 tests with two deprecation warnings.
- A seq64 profile passed the project-local $1.2916 reserve check, but Modal rejected it before app start with `Workspace ... has exceeded its spend limit`. No task, update, checkpoint, or additional usage resulted. The October billing report remains $31.86206706; the current workspace limit is not exposed by the CLI.
- The shared billing guard has no verified monthly budget for this profile and returns `UNKNOWN`; the old $36 figure belonged to other profile evidence and was removed. The 64/96 speed comparison, Phase D pairs, and Phase F GPU confirmation remain pending.

## User-approved R3-R6-only budget update - 2026-10-27

- The latest user approval sets the project-local additional cap to $27, with a $25.65 hard stop. At the prior workspace baseline, the live $19.93099707 additional meter left enough for the $4.9372 R3-R6-only reserve; it did not cover the $5.9788 reserve for the full R2 recovery plus R3-R6 path.
- Added an R3-R6-only option to the Phase F budget runner and Modal entrypoint. It skips R2-TV recovery, reports partial scope rather than full Phase F certification, and requires R3-R6 smoke completion plus R4/R6 budget audits. Focused tests pass 34/34. A later launch under the new workspace was accepted, but the R3 B200 call stayed pending until the coordinator's 3,900-second timeout; no GPU container started.

## Phase F scheduler follow-up - 2026-10-27

- The CPU setup on the new workspace completed successfully. Its manifest records the pinned model revision, the accepted known leakage decision, and all 11,244 unchanged training rows. A failed preparation attempt exposed that the checker did not read Verl's nested `reward_model.ground_truth`; local support and an explicit `--allow-exact-overlaps` mode were added. The final setup used `registered_existing_exports_without_rebuild`, so it did not exercise the updated checker path.
- The R3-R6-only app passed the local gate at $37.99260592 metered with a $4.9372 reserve. R3 call `fc-01M5Y8KHS88RDDV9VPJ6NKFYBT` remained pending without a container for 3,900 seconds. Modal timed out the coordinator, the R3 call was terminated, and the app was stopped with no active containers.
- Final billing summary: $38.23260592 metered and $0 billed after credits. No training update, checkpoint, queue log, or GPU smoke artifact was produced. Launcher log: `remote_artifacts/phasef20261027r3r6a/launcher.log`.
- Detailed call, setup, and billing evidence: `.aris/traces/experiment-audit/2026-10-27_run01/followup_phase_f_b200_queue_timeout.md`.

## R3-R6 local mock preflight - 2026-10-27

- A targeted local `DRY_RUN=1` pass emitted `PREFLIGHT_OK` for R3-R6 and constructed the corresponding TreePO/TreeRL trainer commands. R5/R6 used the configured co-located vLLM memory settings and off/VDRA allocation modes.
- The dry-run path prints commands without executing the trainers. This checks local launch plumbing only; it is not GPU execution evidence and does not change the Phase F FAIL certification.
- Evidence: `codex/dry_runs/phase_f_r3_r6_mock_20261027.sh`, `.log`, and `.exit_code.txt`.