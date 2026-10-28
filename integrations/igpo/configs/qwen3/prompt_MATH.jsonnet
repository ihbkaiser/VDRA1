local qwen3_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} top_k={top_k} max_tokens={max_tokens} seed={seed} logprobs={logprobs} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
local qwen3_question_template = "<|im_start|>user\nSolve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes), where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.\n\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n";

{
  episode_generator+: {
    question_template: qwen3_question_template,
    inference_strategy+: {
      question_template: qwen3_question_template,
      node_expander+: {
        program: qwen3_iid,
        program_kwargs+: {
          temperature: 0.6,
          top_p: 0.95,
          top_k: 20,
        },
      },
    },
    value_estimation_inference_strategy+: {
      question_template: qwen3_question_template,
    },
  },
  inference_pipelines: [
    pipeline + {
      question_template: qwen3_question_template,
      inference_strategy+: {
        question_template: qwen3_question_template,
        node_expander+: {
          program: qwen3_iid,
          program_kwargs+: {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
          },
        },
      },
    }
    for pipeline in super.inference_pipelines
  ],
}
