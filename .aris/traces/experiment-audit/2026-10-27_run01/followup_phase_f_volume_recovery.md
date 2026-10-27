# Phase F volume-recovery follow-up

**Date:** 2026-10-27 (Asia/Bangkok)  
**Method:** read-only Modal volume listing/download, saved run metadata review, and deterministic local audit recomputation. No GPU deployment was started in this follow-up.

## Evidence recovered

- Source volume: `vdra-work/mock/phase_f_null_1629e2315f`.
- Local evidence package: `remote_artifacts/phase_f_r2_tv_volume_recovery_20261027/`.
- R1 and constant-proxy R2 tree logs reproduce the existing null audit exactly. Allocation equality fails for IDs 2117 and 35.
- R2-TV queue, node, and tree logs reproduce the saved budget audit exactly: 50 queues, 54 nodes, complete queue evidence, zero budget violations, CV 0.17687681195739818, budget gate true, headroom gate false.
- The remote R2-TV volume has an HF step-4 checkpoint config and a nonempty ~3.55 GB weight file, plus actor checkpoint state. Weight contents were not downloaded or content-hashed. The R2 trainer metadata records `InputCancellation`; later Modal containers completed without a new trainer start. The strict reuse gate rejects this artifact as a successfully completed training run.

## Remaining method status

- **R3:** First B200 attempt (06:58) has trainer `rc=0` but Modal launcher exit 1; approximate container cost $0.6364. No checkpoint, timing, or episode/queue directory persisted. Second B200 attempt (07:12) ended in `KeyboardInterrupt` after 226.59 seconds; approximate container cost $0.3934. The app stopped later after its local client disconnected. The aggregate report was last modified at 07:04 and does not include the second retry.
- **R4:** Remote run metadata shows repeated launcher failures, cancellation, or keyboard interruption. No durable update/checkpoint/queue evidence is present.
- **R5:** `null_R5_tv` is absent from the Modal volume; there is no Phase F trainer run.
- **R6:** Remote metadata shows launcher failures. The retained R6 retry log records an OOM in MCTS prompt-logprob before update 1.
- **Local preflight:** R1?R6 dry-run commands all emitted `PREFLIGHT_OK`; this establishes command/config construction only.

## Spend boundary and conclusion

The live profile meter is $87.10290662, $19.93099707 above the $67.17190955 baseline. The authorized additional cap is $20, and its 95% safety stop is $19; current usage is past that stop. `modal billing summary` reports $0 billed after credits, but the guard uses metered usage. Modal also previously rejected an app at the workspace spend limit. A fresh guard-only check for the 45-minute R3-R6 reserve projected $24.87 additional usage against the $19 local stop and blocked before Modal deployment. No additional GPU launch was made. Phase F remains uncertified, and no paired end-to-end B200 speedup is established.

The profile-specific local budget guard was aligned from $27 to the authorized $20 cap. The last recorded full CPU suite passed 183 tests with 1 skipped before this follow-up; no tests were rerun for this documentation and budget-constant update.
