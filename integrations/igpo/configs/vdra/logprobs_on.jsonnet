// Ask vLLM for sampled-token logprobs so every node carries num_tokens and
// sum_logprobs (used for token accounting and the pilot-NLL heuristic).
// Sampling itself is unchanged.
{
  episode_generator+: {
    inference_strategy+: {
      node_expander+: {
        program_kwargs+: { logprobs: 1 },
      },
    },
  },
}
