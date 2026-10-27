# Phase F reuse-gate follow-up

Date: 2026-10-27

## Evidence and change

The archived R2-TV bundle records a training start for R2/seed 42 with stop and chunk stop 2, then a `modal_container_end` with `InputCancellation`. Its report references a later short container call marked complete. The bundle has a timing row at global step 2 but does not locally include the checkpoint payload, raw queue/tree logs, or a successful trainer `end` record. A complete later container call alone did not prove that the training process completed.

`modal_app/phase_c.py` now reuses this artifact only when the run metadata contains a successful R2/seed-42 trainer end (`rc=0`) for the expected two-step chunk, the report's matching Modal container call completed after that trainer end, the timing data ends at step 2 with nonzero stage timings, and the step-4 HF checkpoint contains a config and a nonempty model-weight file. The archived cancelled attempt therefore fails the stricter reuse gate and will need a fresh R2-TV run when spend headroom is available.

Regression evidence: the cancellation-with-later-container and empty-checkpoint cases both failed against the old validator. The focused Phase C path suite then passed 8 tests. The full project suite passed 181 tests with 1 skipped (2 protobuf deprecation warnings).

## Current Modal state

The active `billionaireproject8` meter remains $87.10290662; no training tasks are active, and the latest R3-R6 deployment was rejected by the workspace spend limit before app start. The separate `anh-nd210020` profile reports $25.29359805 while its local guard remains configured at $25; no GPU job was charged to it.

Phase F remains uncertified. The IGPO null-equality and R2-TV headroom gates remain failed, and live R1-R6 confirmation plus raw queue/checkpoint evidence remain outstanding. No paired end-to-end B200 speedup has been measured.
