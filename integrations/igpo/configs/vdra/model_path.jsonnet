// Point every model reference at VDRA_MODEL_PATH (the base configs hard-code
// /workspace/storage-shared/models/DeepSeek-R1-Distill-Qwen-1.5B).
local p = std.extVar('APP_VDRA_MODEL_PATH');
{
  episode_generator+: {
    initial_model_name_or_path: p,
    inference_strategy+: { guidance_llm+: { model: p, tokenizer_name: p } },
    value_estimation_inference_strategy+: { guidance_llm+: { model: p, tokenizer_name: p } },
  },
  tokenizer+: { hf_model_name: p },
  trainer+: {
    actor_model+: { hf_model_name: p },
    reference_model+: { hf_model_name: p },
  },
  inference_pipelines: [
    pipeline + {
      inference_strategy+: {
        node_expander+: {
          tokenizer+: { hf_model_name: p },
        },
        guidance_llm+: {
          model: p,
          tokenizer_name: p,
        },
      },
    }
    for pipeline in super.inference_pipelines
  ],
}
