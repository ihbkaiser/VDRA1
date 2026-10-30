# Artifact matrix

| Required evidence | Present? | Evidence path / note | Consequence |
| --- | --- | --- | --- |
| Source code snapshot | Yes | `/workspace/src/gear_source_rebuttal/gear`; 3,259 files, no `.git` | Code-level audit possible; commit identity unavailable |
| IGPO reference checkout | Yes | `/workspace/igpo`; all listed remote branches cloned | Snapshot-to-ref hash comparison possible |
| VDRA iteration 50 weights | Yes | `vdra_iter_0050/.../hf_pretrained/pytorch_model.bin` (3,554,222,037 bytes) | Frozen inference possible |
| VDRA iteration 100 weights | Yes | `vdra_iter_0100/.../hf_pretrained/pytorch_model.bin` (3,554,222,037 bytes) | Frozen inference possible |
| VDRA iteration 150 weights | Yes | `vdra_iter_0150/.../hf_pretrained/pytorch_model.bin` (3,554,222,037 bytes) | Frozen inference possible |
| VDRA iteration 235 weights | Yes | `vdra_iter_0235_final/.../hf_pretrained/pytorch_model.bin` (3,554,222,037 bytes) | Frozen inference possible |
| Matched SPO iteration 235 weights | Yes | `spo_iter_0235_matched/.../hf_pretrained/pytorch_model.bin` (3,554,222,037 bytes) | Frozen inference possible; tokenizer absent in that archive |
| HF model `config.json` | Yes | All five checkpoint dirs; `model_type=qwen2` | Architecture identified, training provenance not identified |
| Tokenizer | Partial | Present for four VDRA checkpoints; absent for SPO | Shared tokenizer may be reused only after matching hashes and confirming same base model |
| Run `config.json` / resolved config | No | Only model architecture config is present | Cannot fix actual training allocation settings |
| `metadata.json`, `dotenv.txt`, run log header | No | No matching files in extracted tree | Cannot connect code revision or runtime settings to runs |
| Stored rollout trees / `traj_results_ds` | No | No `temp_episodes/`, `traj_results_ds`, or equivalent run tree dataset | A1 and exact stored-tree replay blocked |
| Demos / allocation records | No | No `gear_demos/`, `ingpo_demos/`, or `demos.jsonl` | No allocation distribution, pilot selection, or queue utilization statistics |
| Merged edges / episodes | No | No `merged_edges` or `episodes/merged` | A5 from training edges and zero-advantage rate unavailable |
| Training timing / generated tokens | No | No `training_timing.jsonl` or equivalent | A6 runtime reconciliation blocked |
| Evaluation outputs | No | No run evaluation directory in checkpoint archives | Cannot reconstruct published run metrics from archive |
| W&B logs | Partial, unrelated | 247 parseable W&B event files under source snapshot; embedded configs show DeepSeek-R1-Distill-Qwen 1.5B/7B | Not linked to archived Qwen2 checkpoints; do not use for their provenance |
| Paper/supplement tables for A6 reconciliation | No | No paper/supplement PDF or source table supplied in this checkout or archive | Table/Fig discrepancies listed in the plan cannot be adjudicated |

The raw source archive listings and extracted file trees stay local under `docs/`; they are ignored by Git because they are raw inventory dumps. The compact summary above is the published audit record.
