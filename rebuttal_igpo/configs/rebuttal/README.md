# Later-server ablation specifications

These files define the requested equal-budget 2×2 design plus diagnostic ablations. They are protocol specifications for the rebuttal allocator adapter; **they are not yet executable Jsonnet overlays**. The audited IGPO reference is commit `87e4061c998ebb3a348074a6f70b4599fb9f373e` (`origin/fix/budget-allocation-theory`). Its current constructor exposes neither uniform allocation nor upper-bound caps, candidate-selection choice, nor an advantage-baseline switch. Passing descriptive fields through Jsonnet would not change the implementation and could silently run the wrong treatment.

Before a later server run, implement and unit-test an adapter on that server commit that consumes every field in `scenarios.json`, logs actual allocated and built rollouts plus generated/scored tokens per update, and rejects unknown treatment fields. Use the same prompt IDs, seed list `[42]`, fixed per-depth rollout budget, verifier and model commit in every cell. Do not interpret the fix-branch allocator as the archived VDRA training code.

The branch's existing launch entry point is `/workspace/igpo/scripts/train_ingpo_tree_deepseekR1Qwen_MATH.sh`, which sources `scripts/_common.sh` and calls `ingpo_run`. Once the adapter is integrated and a training run is separately authorized, the invocation shape is:

```bash
cd /workspace/igpo
source scripts/_common.sh
APP_DIRECTORY=/workspace/rebuttal_out/server_runs APP_SEED=42 \
  ingpo_run "rebuttal-<scenario>-seed42" \
  "/workspace/igpo/configs/polIter_deepseekR1Qwen_ingpo_tree_MATH.jsonnet,/workspace/igpo/configs/episode_generators/branch_factor_666.jsonnet,<adapter-overlay.jsonnet>"
```

This command invokes the training iteration loop; it was not run. The example uses the IGPO config's DeepSeek model because that is the config family on the fix branch, not because it matches the archived Qwen2 checkpoints.
