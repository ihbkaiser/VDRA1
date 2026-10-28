// One-update Modal smoke shape: two branches at each of the same three depths.
{
  episode_generator+: {
    inference_strategy+: {
      M: 600,
      max_depth: 3,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 2 },
          { depth: 1, branch_factor: 2 },
          { depth: 2, branch_factor: 2 },
        ],
      },
    },
  },
}
