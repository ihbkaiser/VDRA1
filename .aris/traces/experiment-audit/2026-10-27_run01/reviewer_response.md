# Fresh reviewer response

Reviewer: `gpt-6.1-sol`, ultra, same-family and provisional. Agent path: `/root/phase_f_integrity_postfix_audit`.

## Overall result

Overall integrity verdict: **WARN**. Phase F certification remains **FAIL**. No fabricated benchmark ground truth or prediction-derived accuracy normalization was found.

## A — Ground truth: PASS

- Targets come from dataset answer fields or boxed dataset solutions in `analysis/datasets_util.py:15–34,58–82`; predictions are graded against `p["answer"]` in `analysis/eval_checkpoints.py:338`.
- Missing, blank, empty-list and `[None]` targets fail closed. Valid fallback answer fields remain supported.
- Contamination remains disclosed: four OlympiadBench exact overlaps, three diagnostic overlaps, and one notation-equivalent diagnostic overlap. These datasets cannot be described as entirely uncontaminated.

## B — Normalization: PASS

- Benchmark results retain binary `correct`, raw `k_correct`, and `k`; no prediction-extrema denominator was found.
- Allocation CV is an allocation diagnostic with raw histograms. The reviewer recomputed R2-TV's histogram: 54 allocations, sum 313, CV 0.17687681195739843. This matches `remote_artifacts/phase_f_r2_tv_verified_20261027/report.json` and fails the declared 0.3 headroom threshold.
- TV, pilot NLL, and diversity are model-derived allocation proxies, named and logged as such.

## C — Results and completeness: WARN

- Initial artifacts record R1 and constant-proxy R2 as complete, R2-TV timed out, and overall failed. IGPO null equality fails: problem IDs 2117 and 35 have different tree signatures. This establishes failed equality, not an allocator budget violation.
- The saved R2-TV report claims zero budget violations and complete queue evidence but fails headroom. The local bundle contains only `report.json`, `run_meta.jsonl`, and `training_timing.jsonl`; raw queues/trees and checkpoint contents are absent, so the audit claim and checkpoint integrity cannot be independently recalculated.
- Timing records iteration 0, global step 2, and 209.018 seconds. It does not establish two policy iterations. A successful 69.69-second invocation reused the saved artifact and had no new training-start event; it does not prove resumed training.
- The reuse validator still checks checkpoint-directory existence and accepts any timing row with global step >= 2. This limit is documented in `codex/STATUS.md`.
- Archived paired R1 metrics duplicate updates 1/2; current parsing rejects duplicates. R2 only completed base-model evaluation before its training failure.
- Closed during this review: Phase D requires five benchmarks × 16 unique rows with row identities, sidecars and output digests; Phase F checks every required run label and rejects failed audit executions. Remaining prospective caveat: Phase D checks that sidecar signatures are well-formed hashes but does not recompute them against the current dataset/checkpoint/config signature. No extant completed certification demonstrates stale-file acceptance.

## D — Dead paths/output persistence: PASS

- Shared evaluation is wired through `scripts/20_train.sh`; evaluator metrics and sidecars are written by `analysis/eval_checkpoints.py`.
- Null audit failures are retained in aggregate reports. Phase F disables benchmark evaluation, so those artifacts support smoke/instrumentation claims, not held-out accuracy improvements.

## E — Scope: WARN

- Evidence is limited to one model revision, seed 42, tiny smoke settings, and partial execution. No completed three-seed matrix, 20-update pilot, compute-matched study, or paired end-to-end B200 speedup is established.
- README scope wording now identifies the matrix as a plan; STATUS distinguishes saved report claims, missing artifacts, local preflight and uncompleted GPU gates.
- Documentary stale wording identified earlier in `analysis/eval_pending.py`, `README.md:39`, and `README.md:290` has since been corrected.

## F — Evaluation type/cache provenance: WARN

- Benchmark evaluation: `real_gt`. TV/NLL/diversity allocation quantities: model-derived/self-supervised proxy. Null-tree and resume comparisons: behavioral consistency checks, not independent benchmark performance. Fake-trainer/fake-evaluator and synthetic-bank tests: simulation/synthetic-only. No human evaluation is evidenced.
- Cache signatures bind exact dataset rows, prompt, settings, checkpoint inventory, tokenizer asset hashes, and verifier fingerprint. Verifier fingerprints cover active scorer/helper source and applicable package versions; result sidecars bind row counts and result-byte digests. Saved rows record backend and verifier fingerprint.
- Checkpoint identity remains an inventory fingerprint, not a full weight-content digest: files above 1 MiB use size/mtime rather than content hashing. The separate Phase D final-signature comparison caveat remains open.

## Supported claim

Partial seed-42 B200 smoke reports produced training, timing, and allocation summaries; IGPO null equality fails, R2-TV misses headroom, raw evidence is incomplete, and Phase F remains uncertified.
