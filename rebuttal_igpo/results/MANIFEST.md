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
| A3 frozen-policy bank | `/workspace/rebuttal_out/a3_iter100/bank.jsonl.gz` plus `.offset`, `.progress.json`, `.summary.json` | Complete: 112/112 parents; 7,345,343 bytes; SHA-256 `4a86ea455e35fe68502f04c7d59a380479496d43a8cd110ac0cf9bfa1ac3a3ea`. Raw generations remain outside Git. |

No bank or scaled generations are currently staged for publication.

| A3/A2/A4/A5 compact analyses | `rebuttal_igpo/results/a3_iter100/analysis/` | Published CSV/JSON only; no generated text. Files (bytes; SHA-256): `a2_noise_per_prompt.csv` (5750; `8cbdc580f852bacc0112b4680c9f6ea4da20eb0f92cefeed25bf86b47ac4ed70`); `a2_noise_sensitivity.csv` (1279; `27e2424392def68372ace415f7ea808d47b31070793d54d7e48133391c05819d`); `a4_tv_value_pairs.csv` (90006; `21e96cfcf6a0bf389e17cdea2dc19b762f55cf96c2c3ab9252c5c56b5efe8089`); `a4_tv_value_summary.json` (1624; `db687232d08327691b77d7ce48899f1e51cd23974e98a337057f15840ff80c3b`); `a5_allocation_per_prompt.csv` (161603; `1467ba566c9ca250503620defd6e550fe8fe0bfb051f1bef145edba80ea56d86`); `a5_allocation_summary.csv` (11841; `5252297b43508cbe954758cc0d49dd6724f61e081e631e7d0ff432d211c9d760`); `a5_heldout_edges.csv` (42657; `5f8b78d784ea17ae1a37fc10be80f521055effca9cf5daa1d48bb9cec51aba7d`) |

| A7 timed-smoke summaries | `rebuttal_igpo/results/a7_smoke/` | `vdra_rate.json` (397 bytes; SHA-256 `b8e5c9e018a4ee42fa9e95a972d416e57902954bbc243af9a57f919db4bab430`); `spo_rate.json` (398 bytes; SHA-256 `2964ba2aa38f7c93649d8771c8d979b47a768ee879e38d43e8d14a563146bb2b`). The 10-prompt outputs and evaluator logs remain local under `/workspace/rebuttal_out/a7_smoke/`. |
