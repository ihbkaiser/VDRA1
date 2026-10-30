// The archived AIME24 DatasetDict uses title-cased source columns.
// Keep the source MATH task and verifier; only map its field names.
{
  inference_pipelines: [
    if pipeline.inference_name == 'aime24_test' then
      pipeline + {
        task+: {
          problem_field: 'Problem',
          solution_field: 'Solution',
        },
      }
    else pipeline
    for pipeline in super.inference_pipelines
  ],
}
