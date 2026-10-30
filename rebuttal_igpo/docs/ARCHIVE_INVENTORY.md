# Archive inventory

**State:** all six members fully extracted. The outer `rebuttal_output.zip` was not saved locally; members were streamed through HTTP range requests from the HF dataset. Each member has an `.extract-complete` marker, and `$WORK/src` is read-only after extraction.

## Instance snapshot

Captured once on 2026-10-30 20:42 UTC as requested:

| Command | Result |
| --- | --- |
| `df -h /` | 150G filesystem; 2.4G used; 148G available (2%) |
| `free -g` | 31G total; 25G available; 7G swap |
| `nvidia-smi` | NVIDIA GeForce RTX 3090, 24,576 MiB, driver 580.159.03, CUDA driver capability 13.0, idle |

PyTorch in `/venv/main`: `2.11.0+cu128`, CUDA runtime `12.8`, `torch.cuda.is_available() == True`, device capability `(8, 6)`. A CUDA BF16 256×256 matrix multiplication returned finite results.

## Outer HF archive

HF dataset: `ihbkaiser/qwen25vl-3b-msd-checkpoints`; API-reported `rebuttal_output.zip` size: 39,381,522,342 bytes. All nested entries are stored without outer compression. The requested range-based listing was read with `remotezip`; the sizes below are exact `ZipInfo.file_size` values.

| Extraction order | Member | Stored bytes | Destination | State |
| ---: | --- | ---: | --- | --- |
| 1 | `gear_source_rebuttal.tar.zst` | 393,996,450 | `/workspace/src/gear_source_rebuttal` | Extracted; no `.git`; 3,259 files in its `gear/` subtree |
| 2 | `vdra_iter_0100.zip` | 2,810,056,269 | `/workspace/src/vdra_iter_0100` | Extracted |
| 3 | `vdra_iter_0235_final.tar.zst` | 26,281,342,640 | `/workspace/src/vdra_iter_0235_final` | Extracted; 10 regular files after training-state exclusions |
| 4 | `spo_iter_0235_matched.tar.zst` | 3,541,541,507 | `/workspace/src/spo_iter_0235_matched` | Extracted; 6 regular files |
| 5 | `vdra_iter_0150.tar.zst` | 3,544,537,789 | `/workspace/src/vdra_iter_0150` | Extracted; 8 regular files |
| 6 | `vdra_iter_0050.zip` | 2,810,046,083 | `/workspace/src/vdra_iter_0050` | Extracted; 8 regular files; temporary ZIP deleted |

For tar members, `listing_<member>.txt` records the archive listing and sizes. Extraction excludes `global_step*`, `*optim*`, `zero_*`, and training-state `*.pt`. ZIP members are listed before extraction; temporary ZIP files are removed after extraction. The source/artifact tree is made read-only after all six members finish.

## Snapshot source tree

The source snapshot lives at `/workspace/src/gear_source_rebuttal/gear`. It has no `.git` metadata. It contains the newer `treetune/gear/` and `gear_*` names; there is no `vdra_core/` directory in this snapshot. It also contains logs, cached data and result files, so it is not a code-only export. The tracked project files and run-specific resolved settings are being reconciled in `CODE_SYNC_REPORT.md` and `RUN_PROVENANCE.md`.

## Run artifact map

All five run archives contain only checkpoint files (plus tiny custom checkpoint metadata). They do **not** include the run config, metadata, dotenv, logs, generation trees, demos, merged edges, merged episodes, timing or evaluation results. A search of the full extracted tree found no `temp_episodes/`, `traj_results_ds`, `merged_edges`, `episodes/merged`, `gear_demos/`, `ingpo_demos/`, `training_timing.jsonl`, `metadata.json` or run-level `dotenv.txt`.

| Run | Checkpoint | Weight bytes | Tokenizer | HF model config | Stored trajectories/results |
| --- | --- | ---: | --- | --- | --- |
| VDRA iter 50 | `checkpoints/ckpt--iter_0050--epoch_1.00--step_0200/hf_pretrained` | 3,554,222,037 | `tokenizer.json`, `tokenizer_config.json` | Present | Missing |
| VDRA iter 100 | `checkpoints/ckpt--iter_0100--epoch_1.00--step_0400/hf_pretrained` | 3,554,222,037 | `tokenizer.json`, `tokenizer_config.json` | Present | Missing |
| VDRA iter 150 | `checkpoints/ckpt--iter_0150--epoch_1.00--step_0600/hf_pretrained` | 3,554,222,037 | `tokenizer.json`, `tokenizer_config.json` | Present | Missing |
| VDRA iter 235 | `checkpoints/ckpt--iter_0235--epoch_1.00--step_0940/hf_pretrained` | 3,554,222,037 | `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Present | Missing |
| SPO iter 235 | `checkpoints/ckpt--iter_0235--epoch_1.00--step_0940/hf_pretrained` | 3,554,222,037 | No tokenizer files in this archive | Present | Missing |

The checkpoints are Qwen2-family causal LMs (`model_type: qwen2`, 28 layers, hidden size 1536); they are not Qwen2.5-VL model configs despite the HF dataset name. The final VDRA checkpoint archive additionally contains an `actor/latest` pointer. No training-state `*.pt` files survived extraction; the tar filters excluded `*.pt`, `global_step*`, `*optim*`, and `zero_*`.

`gear_source_rebuttal/gear/wandb/wandb` contains 247 parseable W&B offline runs. Their embedded configs describe Gear/SPO runs against DeepSeek-R1-Distill-Qwen 1.5B/7B models, not the archived Qwen2 checkpoint; therefore those logs cannot be assigned to these five checkpoints. No run provenance is asserted from them. See `RUN_PROVENANCE.md` and `ARTIFACT_MATRIX.md`.

Full nested archive listings are retained under `docs/listing_*.txt`; per-run extracted file trees are recorded in `docs/tree_*.txt`. Raw checkpoints and source data remain local and outside Git.
