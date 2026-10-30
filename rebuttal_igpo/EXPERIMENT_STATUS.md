# Experiment status

## Status

Running — A0, the CPU analysis helpers, synthetic T checks, and the vLLM smoke are complete. The source GEAR estimator passes a one-parent GPU integration check. A3's pre-registered 16-prompt/112-parent bank is collecting in the tracked foreground session (PID 23238, started 2026-10-30 21:50 UTC); output is outside Git at `/workspace/rebuttal_out/a3_iter100/`.

## Decisions

- `WORK=/workspace`, `OUT=/workspace/rebuttal_out`, and the published checkout is `/workspace/VDRA1`. Raw checkpoints, datasets, vLLM caches, tokenizer overlay and generated banks stay outside Git.
- The plan uses experiment IDs (A0, A6, A1, T, S, A3, A2, A5, A4, A7, X); IDs are tracked directly.
- All five checkpoint archives lack training configs, metadata, run logs, trees, demos, merged edges/episodes, timing and evaluation outputs. The 247 parseable W&B files in the source snapshot describe DeepSeek-R1-Distill-Qwen 1.5B/7B experiments, not these archived Qwen2 checkpoints. Training commit and resolved settings remain unknown; no source default is attributed to the archived checkpoints.
- New GPU results are **controlled frozen-checkpoint probes**, not replays of archived training. The iteration-100 checkpoint, local MATH validation data, `qwen_MATH.jsonnet` prompt template, Python seed 42, temperature 0.6, top-p 0.95 and 1024-token continuation cap were fixed before the smoke. Thirty-two prompt IDs were sampled without replacement; the first 8 formed S. For A3/A5 the scope was fixed at the first 16 IDs from that manifest before collecting MC outcomes. Prompt IDs, hashes and request seeds are in `controlled_prompt_manifest.json`.
- The source Qwen MATH config uses a 100-token segment horizon; the probe uses that value for regenerating depth-1 states because the archived run's segment settings are missing. It is a controlled source-config choice, not a recovered run setting.
- The probe's TV estimator uses the archived GEAR snapshot implementation with explicit settings: γ=0.9, hierarchical K=16 (4 first-phase candidates), 120/60-token horizons, half-factor enabled. This tests the snapshot estimator only; it does not establish that the archived checkpoints used those settings.
- Source MATH reward extraction/grading is used with the snapshot's `math_inplace_no_answer_prefix` task behavior. Unfinished responses receive 0, matching the Qwen config. The full `MATHRewardFunction` module cannot import without the training-only DeepSpeed dependency; its wrapper behavior was reproduced around the imported source `MATH` task methods and checked on a known answer.
- A3/A5 uses 16 prompts, 112 parents (root plus six depth-1 nodes per prompt), disjoint `R_a=32` / `R_b=32` seed streams per parent, and 16 continuations per TV candidate. Scope was reduced from 32 prompts using compute only: the end-to-end integration smoke ran at about 333 generated tokens/s for small n=2 batches; 32 prompts project to roughly 25 hours before safety margin and over 37 hours with a 1.5× margin, beyond the cutoff. Sixteen prompts project to roughly 12.9 hours before margin and 19.3 hours with margin. This reduces prompt-level power; no subset was selected by outcomes.
- A7 protocol remains fixed: archived VDRA235 and matched SPO235; AIME24/AIME25/AMC23; 16 samples; temperature 0.6; top-p 0.95; seed 42; source verifier. Run it only if evaluator compatibility is confirmed and measured time fits before 2026-11-01 01:00 UTC.

## Setup and A0 audit — complete

- Environment snapshot at 2026-10-30 20:42 UTC: root filesystem had 148 GB available; memory had 25 GB available; GPU was an idle RTX 3090 with 24,576 MiB.
- The HF API lists `rebuttal_output.zip` at 39,381,522,342 bytes; range access succeeded. All six nested archives streamed and extracted; the outer ZIP was not saved. `/workspace/src` is read-only, and training-state `.pt` files were excluded.
- The VDRA checkpoints contain weights, tokenizer and model config; SPO has weights and model config but no tokenizer. Checkpoint config identifies a Qwen2 causal LM (`model_type=qwen2`, 28 layers, hidden size 1536).
- No rollout trees, demonstrations, merged edges/episodes, run-level configs/logs, timing or evaluation outputs were present. See `docs/ARCHIVE_INVENTORY.md`, `docs/ARTIFACT_MATRIX.md` and `docs/RUN_PROVENANCE.md`.
- `gear_source_rebuttal` has no Git metadata. Snapshot/IGPO comparisons are in `docs/CODE_SYNC_REPORT.md`; no IGPO ref contains the snapshot's Gear/VDRA modules. The nearest-date ref is `origin/codex/locate-evaluate.sh` at `94fe2a6dcc528c5f206afba362d3ad0124add2f5`, but it is not an exact match.
- Code-level GEAR mechanism audit is complete in `docs/MECHANISM_AUDIT.md`; all run-specific confirmations remain unverified because run configs and trees are absent.
- The existing unrelated repository suite does not collect: six modules import the missing optional `modal` SDK. No unrelated Modal package was installed.

## CPU implementation and verification — complete

- Added `rebuttal/` helpers for streaming artifact reads, allocation replay, A1 summaries, cost accounting, metrics and prompt-clustered bootstrap, resumable node banks, evaluation wrapping, CLI and synthetic Eq. (5)/(7) checks.
- `allocate_gear_as_trained` matches the supplied snapshot allocator on synthetic queues for floor-only, remainder and fallback paths. No stored-tree allocation match rate is computable because the archives contain no trees.
- `pytest -q rebuttal_igpo/tests`: **27 passed** on 2026-10-30.
- A0 is complete as an evidence inventory. A1 is **Blocked: no stored trees/demos/edges**. A6 is **Blocked: no paper/supplement or matching run logs/timings**. Details are in `docs/COMPUTE_AND_CONSISTENCY_AUDIT.md`.
- T synthetic checks and `DERIVATION_EQ5_EQ7.md` are complete. The simulation values are labelled synthetic and are saved at `/workspace/rebuttal_out/eq5_eq7_simulation.json`.

## Environment and GPU smoke

- Main PyTorch environment remains `2.11.0+cu128`; BF16 CUDA matmul and RTX 3090 capability `(8,6)` passed. Relevant GEAR-source dependencies were installed into `/venv/main` without cache. The source snapshot remains unchanged.
- vLLM 0.20.1 and Torch 2.11.0+cu129 are isolated in `/workspace/vllm-venv`. The model loaded on the RTX 3090 with BF16 and 4096 max context. The first CLI attempt used an unsupported logging flag; it was replaced by the supported `--no-enable-log-requests --disable-uvicorn-access-log` flags.
- The checkpoint's LlamaTokenizerFast metadata triggered the Transformers warning about the Mistral regex. The relevant upstream discussion documents the tokenization mismatch and `fix_mistral_regex=True` correction ([Hugging Face discussion](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84)). A temporary tokenizer overlay under `/workspace/rebuttal_out/tokenizer_iter100_fixed` enables that flag; the checkpoint files remain untouched. This is an explicit inference-time correction; archived training-time tokenizer behavior is unknown.
- S completed: 8 prompts × 6 completions = 48 completions, 40,571 output tokens, 49.06 seconds, **826.93 output tokens/s**, peak sampled GPU memory **20,218 MiB** (93 memory samples). Full per-prompt measurements are in `results/smoke/smoke_iter100.json`; the 32-prompt ID/hash/seed manifest is in `results/smoke/controlled_prompt_manifest.json`.
- One-parent source GEAR proxy integration completed: the snapshot `ConditionalTVEstimator` returned 16 pilot pairs and 4 candidates; it generated 1,440 pilot tokens, scored 94,048 prompt tokens over 256 prompt-logprob requests, and completed in about 14 seconds wall time. A one-root bank smoke with 2 samples per MC half and 2 child continuations completed in 44.54 seconds: 6 generation requests and 12,288 output tokens. All 4 MC rewards in this tiny smoke were zero; this is an integration check only and is not an A3 outcome.
- Conservative A3/A5 projection: 112 parents × 128 full continuations (64 MC references + 4 × 16 candidate continuations), capped at 1,024 tokens each, plus TV pilots. At the measured low-batch rate of about 333 tokens/s, this is ~12.4 hours generation; 112 measured proxy calls add ~0.45 hours. A 1.5× safety margin gives **about 19.3 hours**, within the remaining time before the hard cutoff. Track hourly; stop or shrink if observed progress diverges materially and resume through the gzip bank offset.
- The first detached shell launch exited before producing a bank and left an empty log; it was not treated as completed work. The same deterministic command is now running as a tracked tool session, with stdout/stderr in `/workspace/rebuttal_out/a3_iter100.log`.
- First measured A3 checkpoint at 2026-10-30 21:55 UTC: 5/112 parents in 268.8 seconds, 581,277 MC/child output tokens plus 7,200 proxy-pilot tokens, and 578,756 prompt-logprob scored tokens. Preliminary end-to-end generated throughput is about 2,189 tokens/s (includes proxy pilots; five-node sample only); current sampled GPU memory was 20,908 MiB. At this pace the remaining bank projects to about 1.6 hours plus setup; retain the original 19.3-hour 1.5× conservative bound until more nodes confirm the rate.
- CPU analysis commands for A2/A4/A5 are implemented and targeted tests pass (30 passed). A2 uses prompt-cluster bootstrap and independent R_b variance; A4 reports child-pair TV/value calibration and approximate MC noise bands; A5 emits equal-budget allocation comparisons and documents the MC split-half limits.

## Experiment queue

| ID | State | Evidence / blocker |
| --- | --- | --- |
| A0 | Completed with provenance unresolved | Inventory, code sync and mechanism audit docs |
| A6 | Blocked | No paper/supplement or run-level timing/config/log files |
| A1 | Blocked | No saved trees, demos, queue records or merged edges |
| T | Completed | Synthetic result plus derivation/assumption note |
| S | Completed | 826.93 tokens/s; 20,218 MiB peak sampled GPU memory |
| A3 | Running | 7/112 parents at the 21:57 UTC checkpoint; first-five-node preliminary 2,189 output tokens/s; tracked PID 23238 |
| A5 | Ready after A3 | CPU command implemented; uses A3's held-out MC and candidate child values |
| A2 | Ready after A3 | CPU noise-sensitivity command implemented |
| A4 | Ready after A3 | CPU TV/value calibration command implemented |
| A7 | Pending compatibility check | Evaluator protocol fixed; run only if dependencies and cutoff allow |
| X | Deferred pending A7 | Iteration 150 only if time remains; otherwise camera-ready |

## Publication

- The destination repository is public; publishing rebuttal code/results can compromise double-blind anonymity. Repository visibility was not changed.
- Local commits `f8e28ee625fc1c1c759702c83246dc8743018f37` and `5c16176` contain the audit, CPU helpers, configs, smoke summary, tests, and A2/A4/A5 analyses. Push attempts to `origin/rebuttal` were rejected with HTTP 403: `Permission to ihbkaiser/VDRA1.git denied to ihbkaiser`; the branch remains absent and no remote commit was created. Continue locally and retry at the next publication checkpoint.
- `gh_token` and `hf_token` are ignored, mode 600, untracked and absent from `git log --all -p`; the documented history scan returns no matches. Before each push, staged files are checked for credentials, files over 20 MiB and raw listings/tree dumps. No credential was printed or staged.
