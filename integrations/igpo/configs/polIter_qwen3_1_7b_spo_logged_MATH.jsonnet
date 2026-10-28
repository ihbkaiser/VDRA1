// Qwen3-1.7B thinking-mode SPO. The model and prompt overlays are shared with
// the VDRA arm so the pair differs only in its tree allocation strategy.
(import 'polIter_deepseekR1Qwen_spo_logged_MATH.jsonnet')
+ (import 'model_overrides/qwen3_1_7b.jsonnet')
+ (import 'qwen3/prompt_MATH.jsonnet')
