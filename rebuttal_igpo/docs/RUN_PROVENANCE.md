# Run provenance

## Evidence and conclusion

The archive has five directories named as VDRA iterations 50, 100, 150, 235, and matched SPO iteration 235. Each directory contains an HF checkpoint but no run `config.json` (outside the HF model config), `metadata.json`, `dotenv.txt`, stdout/stderr log, generation dataset, or timing record. The HF model config identifies the checkpoint as a Qwen2 causal LM (`model_type: qwen2`, 28 layers, hidden size 1536); it does not identify the training algorithm or its resolved settings.

The source snapshot's 247 parseable W&B offline runs were decoded from their local `.wandb` event records. They describe Gear/SPO and evaluation runs using DeepSeek-R1-Distill-Qwen 1.5B or 7B models. None names the archived Qwen2 checkpoint family or supplies a unique link to any of the five archive directories. These logs are useful for auditing the snapshot's other experiments but cannot establish provenance for these checkpoints.

Therefore the exact training commit and resolved VDRA/SPO settings **cannot be identified from the supplied artifacts**. The Gear snapshot remains the primary *code candidate* for implementing diagnostics because it is the supplied source snapshot; this is not a finding that these checkpoints were trained with it. The IGPO GitHub refs are comparison references only. No run setting is imputed from Gear defaults, unrelated W&B runs, or checkpoint iteration names.

## Per-run inventory

| Run label | Archive checkpoint | Model config | Training code source | Resolved allocation settings | Provenance status |
| --- | --- | --- | --- | --- | --- |
| VDRA iteration 50 | `vdra_iter_0050/checkpoints/ckpt--iter_0050--epoch_1.00--step_0200/hf_pretrained` | Qwen2, 28 layers, hidden size 1536 | Unknown; Gear snapshot is candidate only | Unknown | Archive contains weights and tokenizer; no run config/logs/trees |
| VDRA iteration 100 | `vdra_iter_0100/checkpoints/ckpt--iter_0100--epoch_1.00--step_0400/hf_pretrained` | Same | Unknown; Gear snapshot is candidate only | Unknown | Archive contains weights and tokenizer; no run config/logs/trees |
| VDRA iteration 150 | `vdra_iter_0150/checkpoints/ckpt--iter_0150--epoch_1.00--step_0600/hf_pretrained` | Same | Unknown; Gear snapshot is candidate only | Unknown | Archive contains weights and tokenizer; no run config/logs/trees |
| VDRA iteration 235 | `vdra_iter_0235_final/checkpoints/ckpt--iter_0235--epoch_1.00--step_0940/hf_pretrained` | Same | Unknown; Gear snapshot is candidate only | Unknown | Archive contains weights, tokenizer and chat template; no run config/logs/trees |
| Matched SPO iteration 235 | `spo_iter_0235_matched/checkpoints/ckpt--iter_0235--epoch_1.00--step_0940/hf_pretrained` | Same | Unknown | Not applicable without run config | Archive contains weights only; no tokenizer or run metadata |

## Settings required but unavailable

For every run, allocator mode, weight mode, candidate selection, `n_min`, cap, queue count and timeout, TV mode and sample count, token horizons, γ, pilot settings, `skip_near_leaf_expand`, root allocation, trainer whitening, generation temperature/top-p, and verifier/reward settings remain unverified. Iteration-to-generation checkpoint mapping is also unverified. The checkpoint names alone are not enough to establish those settings.

## Impact on downstream work

- Static mechanism results in `MECHANISM_AUDIT.md` describe the code snapshot only.
- The stored-tree allocation regression, A1, A6 training-cost reconciliation, and run-faithful A3/A5/A7 replays cannot be performed from this archive.
- Any new frozen-checkpoint probe must be labeled as a separate controlled diagnostic with explicit chosen settings; it cannot be described as a reproduction of the archived VDRA run.
