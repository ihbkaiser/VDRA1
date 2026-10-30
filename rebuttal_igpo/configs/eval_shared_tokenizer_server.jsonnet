// Use the A7 tokenizer for both checkpoints, including SPO235 whose archive
// lacks tokenizer files. The custom launcher injects APP_EVAL_TOKENIZER into
// the vLLM server itself as well as the evaluator's tokenizer objects.
{
  evaluation_vllm_server: {
    script_path: std.extVar('APP_VLLM_SHARED_TOKENIZER_SERVER'),
  },
}
