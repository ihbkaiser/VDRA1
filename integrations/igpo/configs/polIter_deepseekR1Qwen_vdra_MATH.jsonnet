// R2: VDRA on MATH with DeepSeek-R1-Distill-Qwen-1.5B.
// Identical to the SPO-tree config (R1) except for the inference strategy.
(import 'polIter_deepseekR1Qwen_spo_tree_MATH.jsonnet')
+ (import 'vdra/vdra_defaults.libsonnet')
+ (import 'vdra/logprobs_on.jsonnet')
+ (import 'vdra/kl0.jsonnet')
+ {
  episode_generator+: {
    type: 'hybrid_episode_generator_resumable',
    inference_strategy+: {
      type: 'vdra',
    },
  },
}
