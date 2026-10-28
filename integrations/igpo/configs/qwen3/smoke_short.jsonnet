// Short generation and batch sizes for the one-update Modal wiring smoke only.
{
  num_iterations: 1,
  num_episodes_per_iteration: 16,
  episode_generator+: {
    dataset_num_samples_per_iteration: 4,
    inference_strategy+: {
      node_expander+: {
        program_kwargs+: { max_tokens: 256 },
      },
    },
  },
  trainer+: {
    // With per_device_train_batch_size=1, whitening each microbatch can turn a
    // constant per-episode advantage into exactly zero and suppress the mock
    // update. Keep raw advantages for this wiring smoke only; research runs
    // continue to use the full PPO config with whitening enabled.
    params+: {
      whiten_advantages: false,
    },
    general_training_args+: {
      target_train_batch_size: 8,
      per_device_train_batch_size: 1,
    },
  },
}
