# Reproduction commands and paths

## Environment and immutable inputs

```bash
WORK=/workspace
OUT=/workspace/rebuttal_out
REPO=/workspace/VDRA1/.worktrees/rebuttal
GEAR=/workspace/src/gear_source_rebuttal/gear
IGPO=/workspace/igpo
CKPT=/workspace/src/vdra_iter_0100/checkpoints/ckpt--iter_0100--epoch_1.00--step_0400/hf_pretrained
MATH=/workspace/src/gear_source_rebuttal/gear/data/math/validation
```

The source snapshot and checkpoints under `/workspace/src` are read-only. Generated files and large banks go only under `/workspace/rebuttal_out`; no full generations or banks are committed.

## CPU verification and synthetic checks

```bash
cd "$REPO"
pytest -q rebuttal_igpo/tests
/venv/main/bin/python -m rebuttal_igpo.rebuttal.sim_eq5_eq7 > "$OUT/eq5_eq7_simulation.json"
```

## vLLM smoke server

The tested isolated environment is `/workspace/vllm-venv` with vLLM 0.20.1 and Torch 2.11.0+cu129. Bind to loopback; this smoke service is not exposed through Caddy or an external port.

```bash
/workspace/vllm-venv/bin/vllm serve "$CKPT" \
  --tokenizer /workspace/rebuttal_out/tokenizer_iter100_fixed \
  --served-model-name iter100 --host 127.0.0.1 --port 18000 \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.82 \
  --max-num-seqs 32 --seed 42 --generation-config vllm \
  --no-enable-log-requests --disable-uvicorn-access-log
```

After `curl -fsS http://127.0.0.1:18000/health` succeeds, run the pre-registered 8-prompt smoke. It requests six completions per prompt and writes summary-only measurements and the 32-prompt ID/hash/seed manifest:

```bash
cd "$REPO"
/venv/main/bin/python -m rebuttal_igpo.rebuttal.smoke_vllm \
  --dataset-dir "$MATH" --output "$OUT/smoke_iter100.json" \
  --base-url http://127.0.0.1:18000/v1 --seed 42 --prompts-total 32 \
  --smoke-prompts 8 --samples-per-prompt 6 --max-tokens 1024
```

Measured rates and the estimated full-bank wall time must be entered in `EXPERIMENT_STATUS.md` before starting any long GPU job. If the model cannot be served, record the exact error and do not substitute a different model.

## Later A3/A5 collection

Target checkpoint: `$CKPT`; prompt data: `$MATH`; prompt and generation configuration: `rebuttal/configs/eval_generation_0.6_0.95.jsonnet` plus the fixed prompt manifest. Output root: `$OUT/a3_iter100/`. The bank writer uses gzip members and an offset sidecar for restart. The records must keep independent `R_a` and `R_b` seeds, child sample seeds, prompt IDs, verifier version and allocator/bound configuration.

The integrated collector uses the archived GEAR `ConditionalTVEstimator`, the source MATH task answer extractor/grader and a resumable gzip bank. A one-parent run with `--mc-per-half 2 --child-mc 2 --root-only` verified the writer and reward callback; it is an integration check, not a result. The measured low-batch output rate gives a conservative 19.3-hour projection for the fixed 16-prompt scope including 1.5× margin. Run in a persistent foreground session and monitor at least hourly. The first detached-shell attempt in this environment exited without output; the tracked foreground session is the active run.

```bash
cd "$REPO"
env PYTHONUNBUFFERED=1 PYTHONPATH="$GEAR" /venv/main/bin/python -m rebuttal_igpo.rebuttal.gear_probe \
  --dataset-dir "$MATH" \
  --manifest "$OUT/controlled_prompt_manifest.json" \
  --output "$OUT/a3_iter100/bank.jsonl.gz" \
  --tokenizer /workspace/rebuttal_out/tokenizer_iter100_fixed \
  --seed 42 --prompt-count 16 --segment-tokens 100 --max-new-tokens 1024 \
  --mc-per-half 32 --child-mc 16 \
  > "$OUT/a3_iter100.log" 2>&1
```

The job saves a durable `.offset` sidecar, an atomically replaced `.progress.json` checkpoint, a JSON progress line after each completed parent, and deterministic per-node seeds; rerunning the same command resumes completed nodes. Keep the bank local and record its final size and SHA-256 in `results/MANIFEST.md`.

## A7 evaluation wrapper

The wrapper is `rebuttal_igpo/rebuttal/eval_avgk.py`. IGPO `main` does not accept a sample-count override; use a detached worktree at `origin/agent/checkpoint-sweep-evals` commit `5431bc6` (`/workspace/igpo-eval-compat`), which adds `--num-samples` and its Jsonnet override. The selected config's task metrics return `pass@16` and `avg@16`; no unsupported `--metrics` flag is passed. The 30 AIME24, 30 AIME25 and 40 AMC23 rows exist under the read-only archive. Their `data/` links in the detached worktree point to those saved datasets. The SPO235 checkpoint has no tokenizer files; use the VDRA235 tokenizer directory for both policies so the pair shares one tokenizer. VDRA iteration 100 and VDRA235 have identical 151,665-entry tokenizer vocabularies, and SPO235's model config matches the VDRA235 Qwen2 config, but SPO tokenizer identity is unverified. This is a documented comparison limitation, not a recovered training setting. The wrapper sets its working directory to the evaluator checkout and writes to the requested output directory.

Run only after A3/A5 and if measured time fits before the GPU cutoff:

```bash
cd "$REPO"
export PYTHONPATH="$REPO"
export PATH=/workspace/vllm-venv/bin:/venv/main/bin:$PATH
EVAL_SCRIPT=/workspace/igpo-eval-compat/scripts/evaluate.sh
EVAL_CONFIG=/workspace/igpo-eval-compat/configs/qwen1_5b_base_for_MATH_eval.jsonnet
SHARED_TOKENIZER=/workspace/src/vdra_iter_0235_final/checkpoints/ckpt--iter_0235--epoch_1.00--step_0940/hf_pretrained
/venv/main/bin/python -m rebuttal_igpo.rebuttal.eval_avgk \
  --script "$EVAL_SCRIPT" --config "$EVAL_CONFIG" \
  --model /workspace/src/vdra_iter_0235_final/checkpoints/ckpt--iter_0235--epoch_1.00--step_0940/hf_pretrained \
  --datasets aime24,aime25,amc23 --output-dir "$OUT/a7_iter235/vdra" \
  --samples 16 --seed 42 --tokenizer "$SHARED_TOKENIZER" --context-length 4096 \
  --generation-overlay "$REPO/rebuttal_igpo/configs/eval_generation_0.6_0.95.jsonnet"
/venv/main/bin/python -m rebuttal_igpo.rebuttal.eval_avgk \
  --script "$EVAL_SCRIPT" --config "$EVAL_CONFIG" \
  --model /workspace/src/spo_iter_0235_matched/checkpoints/ckpt--iter_0235--epoch_1.00--step_0940/hf_pretrained \
  --datasets aime24,aime25,amc23 --output-dir "$OUT/a7_iter235/spo" \
  --samples 16 --seed 42 --tokenizer "$SHARED_TOKENIZER" --context-length 4096 \
  --generation-overlay "$REPO/rebuttal_igpo/configs/eval_generation_0.6_0.95.jsonnet"
```

The process environment uses `/venv/main` for the DeepSpeed evaluator (Torch 2.11.0+cu128) and `/workspace/vllm-venv` for the VLLM subprocess (Torch 2.11.0+cu129). `deepspeed==0.19.9` is installed with optional ops disabled; the PATH order below selects its executable from the main venv while `python3` in the VLLM server script resolves to `/workspace/vllm-venv/bin/python3`. CPU-only DeepSpeed launch, VLLM server help, dataset loading and wrapper dry-run passed. The source runtime hard-codes inference pipeline seed `2746318213` in `PolicyIterationRuntime.run_evaluation`; `--seed 42` sets the server/global seed but does not replace that pipeline seed. Both policies use the same fixed pipeline seed, which must be reported as the effective generation seed. Do not treat the wrapper's command construction test as evidence the evaluation ran.

## A2/A4 bank analyses

Run only after the A3 bank has completed and its `.summary.json` exists. A2 groups the 112 parent rows by prompt, allocates `6 × parent_count` samples with `eq7_exact`, lower bound 1 and no cap, perturbs proxy `C` by lognormal noise, and evaluates each allocation using unbiased sample variances from the disjoint `R_b` half. The reported 95% intervals bootstrap 16 prompt clusters. A4 compares each recorded pair TV with the absolute gap between the corresponding fixed candidate-child reward means and reports normal-approximation reference-noise bands; child samples are reused for this descriptive calibration, not an independent test set. A5 compares allocation rules under the same per-prompt feasible budget across `trained`, `capped`, and `capped_k` bounds. Its source-default GEAR comparator uses λ=0.005, `n_min=2`, largest remainder and uniform fallback from the snapshot; these are not recovered training settings. `L_node`/`L_edge` are MC-half value-error proxies, while `L_heldout` is a fixed candidate-edge split-half disagreement (identical across allocation rules), not ground truth. The candidate-bounded regime can reduce the common budget below 6 per parent.

```bash
cd "$REPO"
/venv/main/bin/python -m rebuttal_igpo.rebuttal.cli bank-a2 \
  "$OUT/a3_iter100/bank.jsonl.gz" "$OUT/a3_iter100/analysis" --seed 42
/venv/main/bin/python -m rebuttal_igpo.rebuttal.cli bank-a4 \
  "$OUT/a3_iter100/bank.jsonl.gz" "$OUT/a3_iter100/analysis"
/venv/main/bin/python -m rebuttal_igpo.rebuttal.cli bank-a5 \
  "$OUT/a3_iter100/bank.jsonl.gz" "$OUT/a3_iter100/analysis"
```
