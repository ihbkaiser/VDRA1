# Experiment status

## Status

Running — setup and A0 provenance/artifact audit.

## Decisions

- `WORK=/workspace`, `OUT=/workspace/rebuttal_out`, and the public repository checkout is `/workspace/VDRA1`. This follows the plan's sibling `src/` layout and keeps raw data outside Git.
- The plan uses experiment IDs (A0, A6, A1, T, S, A3, A2, A5, A4, A7, X), while the executing-plans helper expects `Task N` headings. I will track those IDs directly in the SDD ledger and create one brief per plan ID; execution order remains §4.

## Setup

- Snapshot captured 2026-10-30 20:42 UTC: `df -h /` showed 148G available; `free -g` showed 25G available; `nvidia-smi` showed an idle RTX 3090 with 24,576 MiB.
- The HF dataset API lists `rebuttal_output.zip` at 39,381,522,342 bytes; range access with the provided token succeeded.
- The `VDRA1` `rebuttal` branch was absent, so the local branch was created from `origin/main` at `8c4624e66b9255b3bfc75eeb344e957c6095efdc`.
- `gh_token` and `hf_token` are ignored; working copies have mode 600. No push has occurred.
- Installed `remotezip`, `zstandard`, and `datasets` in `/venv/main`; torch was not intentionally changed.
