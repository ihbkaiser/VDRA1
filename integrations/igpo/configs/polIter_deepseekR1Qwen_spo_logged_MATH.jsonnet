// R1: SPO-tree on MATH with DeepSeek-R1-Distill-Qwen-1.5B, unchanged algorithm,
// plus per-tree token logging ("spo_logged" strategy) for the budget audit.
(import 'polIter_deepseekR1Qwen_spo_tree_MATH.jsonnet')
+ (import 'vdra/logprobs_on.jsonnet')
+ (import 'vdra/kl0.jsonnet')
+ {
  episode_generator+: {
    // identical to SPO's hybrid generator, plus a replay buffer that survives restarts
    type: 'hybrid_episode_generator_resumable',
    inference_strategy+: {
      type: 'spo_logged',
    },
  },
}
