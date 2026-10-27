# Phase F recovery orchestration and speed follow-up

**Date:** 2026-10-27 (Asia/Bangkok)

## Changes

- The R2-TV strict validator now accepts either a fresh completed run or a previously saved run, while still requiring the actual R2/seed-42 start, successful trainer end, matching successful Modal container end, global step 2 timing, a nonempty HF checkpoint weight file, and a successful queue-budget audit. A `reused_saved_run_path` marker alone no longer qualifies a run.
- `phase_f_remaining_smoke` now runs a fresh isolated R2-TV recovery when no artifact passes that validator. It refuses to overwrite a nonempty unverified root, reads the R1/constant-R2 null baselines from the separate existing run tag, saves the recovery audit before proceeding, and starts R3?R6 only after the recovery passes every gate.
- Increased R2-TV recovery timeout to 10 minutes. The recorded R2-TV smoke took 209.02 seconds to reach step 2.
- Updated the guarded R3?R6 runners to generate fresh alphanumeric run tags, accept an explicit baseline run tag, and reserve all 55 GPU minutes (10 R2, 10 R3, 15 R4, 10 R5, 10 R6) plus $0.25 non-GPU cushion: $5.9788 total.
- Added Modal's `--detach` option to budgeted Phase D and Phase F GPU launchers so local-client disconnection does not cancel the remote app.

## Verification and limits

- Focused local checks: `py -3.12 -m pytest -q tests/test_phase_c_paths.py tests/test_phase_f_runner_cli.py tests/test_phase_d.py` ? **29 passed, 2 deprecation warnings**. The full project suite then passed: `py -3.12 -m pytest -q tests` ? **186 passed, 1 skipped, 2 deprecation warnings**.
- Current guard-only check reports $19.93 additional meter and blocks the $5.73 GPU reserve plus $0.25 cushion before deployment; projected additional usage is $25.91 against a $19.00 local stop.
- No Modal deployment or GPU run was attempted. Phase F GPU completion and the matched `max_num_seqs=96` speed comparison remain unverified.
