// Evaluation-only generation override. The evaluator's overlay merge applies
// this to all selected inference pipelines without changing the base config.
{
  inference_pipelines: [
    pipeline + {
      inference_strategy+: {
        node_expander+: {
          program_kwargs+: {
            temperature: 0.6,
            top_p: 0.95,
          },
        },
      },
    }
    for pipeline in super.inference_pipelines
  ],
}
