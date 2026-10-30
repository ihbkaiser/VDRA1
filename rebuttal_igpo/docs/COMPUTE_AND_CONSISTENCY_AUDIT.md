# Compute and consistency audit (A6)

## Result

**Blocked for empirical reconciliation:** the extracted archives have no `demos.jsonl`, `training_timing.jsonl`, run-level configs/logs, rollout trees, merged episodes, or evaluation folders. The source snapshot's 247 parseable offline W&B runs describe different DeepSeek-R1-Distill-Qwen runs and cannot be assigned to the archived Qwen2 checkpoints. The paper and supplement were not supplied with the archive. No paper-vs-run mismatch is inferred from these missing inputs.

The following items are explicitly unresolved rather than guessed:

| Requested check | Evidence needed | Status |
| --- | --- | --- |
| Fig. 2a vs Table 1 | Paper definitions, data and evaluation outputs | Blocked: paper/supplement and matching outputs absent |
| Eq. 7 bound vs uncapped `n > 6` (M5) | Resolved run config and stored tree allocations | Blocked: both absent |
| GPU-hour span vs 1.20× | Per-run timing plus hardware / accounting definition | Blocked: matching timing files and paper context absent |
| `ε_tail` | Paper definition and run configuration | Blocked: paper/supplement and run config absent |
| AIME24/25 rows and 2.46 vs 2.48 | Evaluation samples, aggregation and paper table | Blocked: matching evaluation outputs and paper absent |
| DP vs greedy solver | Solver code at run commit and resolved run metadata | Blocked: run commit unknown; fix-branch reference is not proven to have generated checkpoints |
| Rollout budget vs total sampling cost | Training demos, timing and generated/scored token counts | Blocked: all matching run records absent |
| `L = 512` | Paper definition and resolved training/evaluation config | Blocked: paper and matching config absent |
| OlympiadBench subset | Dataset ID, selection manifest and evaluation outputs | Blocked: no evaluation folder/manifest |
| Commit/settings behind each Table 1/4 row | Run metadata, logs and table provenance | Blocked: no matching logs/configs; checkpoint provenance is unresolved |

## Inputs inspected

Archive presence/absence is inventoried in `ARCHIVE_INVENTORY.md` and `ARTIFACT_MATRIX.md`. Run-to-code uncertainty is detailed in `RUN_PROVENANCE.md`. The source-vs-IGPO comparisons are in `CODE_SYNC_REPORT.md`. This audit can be reopened if the paper/supplement or matching run artifacts become available.
