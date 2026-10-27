# Phase D signature follow-up review

Date: 2026-10-27
Reviewer: `gpt-6.1-sol`, ultra, same-family and provisional
Agent path: `/root/phase_f_integrity_postfix_audit`

## Result

The Phase D evaluation-signature freshness gap identified in the initial review is closed. The reviewer compared Phase D's recomputed inputs with the shared evaluator and `analysis/eval_pending.py` for ordinary and resumed evaluations.

- Phase D checks the expected five benchmarks at 16 rows each and compares each result's signature to a signature recomputed from the current evaluation inputs.
- Step 0 uses the base model, matching `eval_pending.py`, even when a step-0 checkpoint directory is present.
- Resume validation computes separate expected signatures for the reference and resumed step-4 evaluations.
- Inputs match the shared evaluator for datasets/limit, checkpoint discovery, tokenizer, greedy decoding settings, verifier backend/fingerprint, and applicable protocol settings. `gpu_hours` is excluded from the signature payload; `SAMPLE_K=2` does not activate average-at-k below step 100.
- Regression coverage includes stale signatures, checkpoint configuration changes, dataset answer changes, and the step-0 path. The project suite passes 179 tests with 1 skipped.

## Remaining findings

This review closes a local evaluation-integrity gap; it does not certify Phase F. R1-R6 passed local command/config dry-run preflight only. The Modal workspace spend limit rejected the next deployment before an app started, so fresh GPU updates and live run/equivalence evidence remain unavailable. Existing gates still include failed IGPO null equality, R2-TV headroom below threshold, and missing raw queue/tree logs and checkpoint contents. Checkpoint identity is still an inventory fingerprint, not a full weight-content digest.

Reviewer classification remains A PASS, B PASS, C WARN, D PASS, E WARN, F PASS; overall WARN; Phase F certification FAIL.
