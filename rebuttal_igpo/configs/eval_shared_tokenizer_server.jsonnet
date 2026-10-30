// Use the A7 tokenizer for both checkpoints, including SPO235 whose archive
// lacks tokenizer files. The custom launcher injects APP_EVAL_TOKENIZER into
// the vLLM server itself as well as the evaluator's tokenizer objects.
{
  // The benchmark-only config does not select a runtime; the source entrypoint
  // requires this dispatch key before it can start an evaluation.
  runtime_type: 'policy_iteration',
  evaluation_vllm_server: {
    script_path: std.extVar('APP_VLLM_SHARED_TOKENIZER_SERVER'),
  },
}
