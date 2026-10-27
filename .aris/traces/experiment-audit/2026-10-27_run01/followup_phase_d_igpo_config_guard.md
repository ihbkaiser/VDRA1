# IGPO R2 config guard follow-up - 2026-10-27

- The retained R2 effective config and runtime overlay from the B200 attempt show `enable_prefix_caching=true`, although the R2 config intended false.
- Current `scripts/20_train.sh` selects `configs/episode_generators/no_prefix_caching.jsonnet` for R2. That Jsonnet file sets `enable_prefix_caching: false`; current `work/igpo/scripts/_common.sh` leaves this field to the selected config instead of emitting a later override.
- Added `test_r2_no_prefix_cache_config_is_not_overridden_by_runtime_overlay` in `tests/test_phase_d.py`. The focused suite passes: 18 passed, two protobuf deprecation warnings. The test validates source/config wiring; Jsonnet is unavailable locally (`_jsonnet` and CLI not installed), so it does not produce a rendered effective config.
- The corrected effective config remains unvalidated on B200. Modal profile `billionaireproject8` remains blocked by the workspace spend limit; no deployment was attempted during this follow-up.

- Extended local verification: the current Bash helper generated 64- and 96-sequence runtime overlays; Jsonnet 0.22 merged each with the archived full R2 config and checked-in no-prefix overlay. Both resolve `enable_prefix_caching=false`; only `max_num_seqs` differs. Evidence: `codex/phase_b_results/igpo_seq64_96_config_preflight_20261027.json`. This remains a config preflight, not a B200 speed result.
