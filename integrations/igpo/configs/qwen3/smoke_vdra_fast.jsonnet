// Reduced inference budget for the Modal wiring smoke only. The research
// configs continue to use the full VDRA defaults from vdra_defaults.libsonnet.
{
  episode_generator+: {
    inference_strategy+: {
      M: 128,
      vdra+: {
        k0: 2,
        pilot_tokens: 32,
        r: 2,
        m: 8,
        n_queues: 4,
        queue_capacity: 8,
        queue_timeout_s: 0.1,
      },
    },
  },
}
