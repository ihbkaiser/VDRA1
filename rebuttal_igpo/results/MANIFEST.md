# Results manifest

Large generated artifacts stay under `/workspace/rebuttal_out` or the read-only `/workspace/src`; they are not uploaded or committed. Add the final byte size and SHA-256 for each generated bank before closing its experiment.

| Artifact | Path | Status / contents |
| --- | --- | --- |
| Archived input checkpoints and datasets | `/workspace/src/<archive-member>/` | Read-only inputs; extracted from the supplied archive; not new result files and never publish. |
| vLLM help logs | `/workspace/rebuttal_out/vllm_help*.txt` | Local environment diagnostics only. |
| Eq. (5)/(7) synthetic result | `/workspace/rebuttal_out/eq5_eq7_simulation.json`; SHA-256 `3490065d83a7c88c8f9e19d8d00ef44bbcd292c938bf25a462692c36f9a767d4`; 786 bytes | Small deterministic Monte Carlo summary; published copy at `results/synthetic/eq5_eq7_simulation.json`. |
| Controlled prompt manifest | `/workspace/rebuttal_out/controlled_prompt_manifest.json`; SHA-256 `b67ecb4d00c6b59191c9d9f145885b3c1b6ba0b0785612270ccfc9d9a95dca58`; 8,435 bytes | 32 selected MATH validation row IDs, prompt hashes, seeds and protocol; no prompt text or generated completions. Published copy under `results/smoke/`. |
| vLLM smoke summary | `/workspace/rebuttal_out/smoke_iter100.json`; SHA-256 `d6e767f7a26a0c15b1436b004e63427cdb976e0c34a10b8086e5529f22b24fe3`; 4,684 bytes | Summary metrics, per-prompt completion counts/timings and finish reasons; no completion text. Published copy under `results/smoke/`. |
| Fixed tokenizer metadata overlay | `/workspace/rebuttal_out/tokenizer_iter100_fixed/` | Two tokenizer files; symlinked tokenizer JSON and a separate config with `fix_mistral_regex=true`; checkpoint remains unchanged. |
| One-root integration smoke bank | `/workspace/rebuttal_out/probe_integration_smoke.jsonl.gz`; SHA-256 `956d38b8e4f266866feff45cb310af076c7710ec7afe595674b8a3fa45541ecd`; 14,491 bytes | One root; `R_a=R_b=2`, child MC=2; all four observed rewards were zero. Integration evidence only; not a statistical result and not published. |
| One-root integration summary | `/workspace/rebuttal_out/probe_integration_smoke.jsonl.gz.summary.json`; 935 bytes | Wall time, token count, prompt ID and reduced smoke protocol; local only. |
| Future A3/A5 bank | `/workspace/rebuttal_out/a3_iter100/` | Not started; resumable `.jsonl.gz` + `.offset` when produced; hash/size to be recorded. |

No bank or scaled generations are currently staged for publication.
