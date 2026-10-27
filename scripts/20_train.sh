#!/usr/bin/env bash
# Train one configuration of the matrix, resuming automatically.
#
#   GPUS=0,1 bash scripts/20_train.sh <RUN> <SEED> [STOP] [NAME]
#
# RUN:  R1 SPO | R2 VDRA | R3 TreePO | R4 TreePO+VDRA | R5 TreeRL | R6 TreeRL+VDRA
# STOP: policy updates to reach (default STOP_STEP=220)
# NAME: run directory under RUNS_DIR (default <RUN>_s<SEED>)
#
# Resume behaviour (all three frameworks):
#   * a resumable checkpoint (weights + optimizer + LR schedule + data position) is
#     written every RESUME_EVERY updates and at STOP;
#   * evaluation weights are kept every KEEP_EVERY updates and at STOP;
#   * re-running the same command continues from the newest resumable checkpoint;
#     running it with a larger STOP continues training (compute-matched baselines);
#   * a run that crashes is restarted up to MAX_RESTARTS times by this wrapper;
#   * a run that already reached STOP is skipped; a run that is already running
#     (lock held) is skipped.
# Extra knobs (env):
#   VDRA_VARIANT_JSON='{"proxy":"constant"}'   VDRA overrides (null test, ablations)
#   KEEP_EVERY=<n> RESUME_EVERY=<n> MAX_RESTARTS=<n>
#   EVAL_EVERY_N=<n>   pause training every n updates, evaluate the new checkpoints on
#                      this run's GPUs (shared evaluator, one benchmark set per GPU),
#                      then resume training; 0 = no in-training evaluation
#   EVAL_AT_END=1      evaluate all pending checkpoints when STOP is reached
#   EVAL_STEP0=1       also evaluate the base model as step 0 (first pause)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RUN="${1:?RUN}"; SEED="${2:?SEED}"
STOP="${3:-${STOP_STEP}}"
NAME="${4:-${RUN}_s${SEED}}"
GPUS="${GPUS:-0,1}"
KEEP_EVERY="${KEEP_EVERY:-${SAVE_STEPS:-${SAVE_EVERY}}}"
RESUME_EVERY="${RESUME_EVERY:-5}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
VDRA_VARIANT_JSON="${VDRA_VARIANT_JSON:-{\}}"
RUN_DIR="${RUNS_DIR}/${NAME}"
mkdir -p "${RUN_DIR}" "${RUNS_DIR}/_cfg"
export CUDA_VISIBLE_DEVICES="${GPUS}"
case "${RUN}" in R1|R2) FRAMEWORK=treetune;; R3|R4) FRAMEWORK=verl;; R5|R6) FRAMEWORK=openrlhf;;
  *) echo "unknown RUN ${RUN}" >&2; exit 2;; esac
EVAL_EVERY_N="${EVAL_EVERY_N:-0}"
EVAL_AT_END="${EVAL_AT_END:-$([[ "${EVAL_EVERY_N}" != "0" ]] && echo 1 || echo 0)}"
EVAL_STEP0="${EVAL_STEP0:-1}"
if [[ "${FRAMEWORK}" == "treetune" ]] && (( EVAL_EVERY_N > 0 && EVAL_EVERY_N % RESUME_EVERY != 0 )); then
  echo "EVAL_EVERY_N=${EVAL_EVERY_N} must be a multiple of RESUME_EVERY=${RESUME_EVERY} for treetune runs" >&2; exit 2
fi
if [[ "${FRAMEWORK}" == "treetune" ]] && (( STOP % RESUME_EVERY != 0 )); then
  # treetune only checkpoints on multiples of save_steps; the stop update must be one
  echo "STOP=${STOP} must be a multiple of RESUME_EVERY=${RESUME_EVERY} for treetune runs" >&2; exit 2
fi

# ------------------------------------------------------------------ treetune (R1, R2)
train_igpo() {
  local cfg
  case "${RUN}" in
    R1) cfg="${IGPO_DIR}/configs/polIter_deepseekR1Qwen_spo_logged_MATH.jsonnet" ;;
    R2) cfg="${IGPO_DIR}/configs/polIter_deepseekR1Qwen_vdra_MATH.jsonnet" ;;
  esac
  use_env "${CONDA_IGPO}"
  record_provenance "${RUN_DIR}" "${IGPO_DIR}"
  export APP_SEED="${SEED}" APP_DIRECTORY="${RUNS_DIR}" APP_EXPERIMENT_NAME="${NAME}"
  export VDRA_TRAIN_DATASET_DICT="${DATA_DIR}/math-nodiag" VDRA_MODEL_PATH="${MODEL_PATH}"
  export VDRA_STOP_STEP="${TARGET}" VDRA_VARIANT_JSON
  export APP_VDRA_TRAIN_DATASET_DICT="${VDRA_TRAIN_DATASET_DICT}"
  export APP_VDRA_MODEL_PATH="${VDRA_MODEL_PATH}"
  export APP_VDRA_STOP_STEP="${VDRA_STOP_STEP}"
  export APP_VDRA_VARIANT_JSON="${VDRA_VARIANT_JSON}"
  export PYTHONPATH="${IGPO_DIR}:${VDRA_HOME}:${PYTHONPATH:-}"
  # deepspeed --include uses physical ids, so do not remap with CUDA_VISIBLE_DEVICES
  unset CUDA_VISIBLE_DEVICES
  # treetune keeps optimizer state only in its newest checkpoint (resume point) and
  # model weights for iterations divisible by KEEP_EVERY (evaluation)
  export INGPO_GPUS="${GPUS}" INGPO_SAVE_STEPS="${RESUME_EVERY}" INGPO_CHECKPOINT_KEEP_STEPS="${KEEP_EVERY}"
  export INGPO_EVAL_EVERY_N_ITERATIONS=100000
  # Checkpoint evaluation is handled by this script's shared evaluator. IGPO's
  # built-in runtime still evaluates iteration 0 with its repo-relative data
  # paths even at a 100000-iteration interval, so omit that duplicate pass.
  export INGPO_DISABLE_INTERNAL_EVALUATION="${INGPO_DISABLE_INTERNAL_EVALUATION:-1}"
  # one GPU works natively (DeepSpeed world size 1; the per-rank vLLM server shares the GPU,
  # its memory share is set automatically from free memory); override with IGPO_VLLM_MEM
  if [[ -n "${IGPO_VLLM_MEM:-}" ]]; then export INGPO_VLLM_GPU_MEMORY_UTILIZATION="${IGPO_VLLM_MEM}"; fi
  local cfgs="${cfg},${IGPO_DIR}/configs/episode_generators/branch_factor_666.jsonnet"
  cfgs+=",${IGPO_DIR}/configs/episode_generators/no_prefix_caching.jsonnet"
  cfgs+=",${IGPO_DIR}/configs/vdra/data_nodiag.jsonnet,${IGPO_DIR}/configs/vdra/model_path.jsonnet"
  cfgs+=",${IGPO_DIR}/configs/vdra/stop.jsonnet"
  if [[ "${INGPO_DISABLE_INTERNAL_EVALUATION}" == "1" ]]; then
    cfgs+=",${IGPO_DIR}/configs/vdra/no_internal_eval.jsonnet"
  fi
  [[ "${RUN}" == "R2" ]] && cfgs+=",${IGPO_DIR}/configs/vdra/variant.jsonnet"
  (
    cd "${IGPO_DIR}"
    export INGPO_ROOT="${IGPO_DIR}"
    source "${IGPO_DIR}/scripts/_common.sh"
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "+ ingpo_run ${NAME} ${cfgs}"
    else
      ingpo_run "${NAME}" "${cfgs}"
    fi
  )
}

# ------------------------------------------------------------------ verl / TreePO (R3, R4)
convert_verl_ckpts() {
  # 1. record every checkpoint's time (GPU-hours survive pruning);
  # 2. merge evaluation steps (multiples of KEEP_EVERY, and STOP) into HF weights;
  # 3. delete FSDP states older than the two newest, never the resume point.
  local d step out latest=""
  [[ -f "${RUN_DIR}/ckpts/latest_checkpointed_iteration.txt" ]] && \
    latest="$(cat "${RUN_DIR}/ckpts/latest_checkpointed_iteration.txt")"
  local -a steps=()
  for d in "${RUN_DIR}"/ckpts/global_step_*; do
    [[ -d "${d}/actor" ]] || continue
    step="${d##*global_step_}"
    steps+=("${step}")
    grep -q "\"step\": ${step}," "${RUN_DIR}/ckpt_times.jsonl" 2>/dev/null || \
      echo "{\"step\": ${step}, \"mtime\": $(stat -c %Y "${d}")}" >> "${RUN_DIR}/ckpt_times.jsonl"
    if (( step % KEEP_EVERY == 0 || step == TARGET || step == STOP )); then
      out="${RUN_DIR}/hf/global_step_${step}"
      if [[ ! -f "${out}/config.json" ]]; then
        run python3 "${TREEPO_DIR}/scripts/model_merger.py" merge --backend fsdp \
          --local_dir "${d}/actor" --target_dir "${out}" || { log "merge failed for ${d}"; continue; }
        [[ "${DRY_RUN}" == "1" ]] || cp -n "${MODEL_PATH%/}"/tokenizer* "${out}/" 2>/dev/null || true
      fi
    fi
  done
  (( ${#steps[@]} > 2 )) || return 0
  local -a sorted; mapfile -t sorted < <(printf '%s\n' "${steps[@]}" | sort -n)
  local keep1="${sorted[-1]}" keep2="${sorted[-2]}"
  for step in "${sorted[@]}"; do
    [[ "${step}" == "${keep1}" || "${step}" == "${keep2}" || "${step}" == "${latest}" ]] && continue
    if (( step % KEEP_EVERY == 0 || step == TARGET || step == STOP )) && [[ ! -f "${RUN_DIR}/hf/global_step_${step}/config.json" ]]; then
      continue   # not converted yet: keep the FSDP state
    fi
    run rm -rf "${RUN_DIR}/ckpts/global_step_${step}"
  done
}

train_treepo() {
  use_env "${CONDA_TREEPO}"
  record_provenance "${RUN_DIR}" "${TREEPO_DIR}"
  local div="fixed_avg"
  local -a vdra_args=()
  if [[ "${RUN}" == "R4" ]]; then
    div="vdra"
    local merged
    merged="$(python3 -c 'import json,sys; a=json.loads(sys.argv[1]); a.update(json.loads(sys.argv[2])); print(json.dumps(a))' \
      "${VDRA_CONFIG_JSON}" "${VDRA_VARIANT_JSON}")"
    while IFS= read -r kv; do
      [[ -n "${kv}" ]] && vdra_args+=("+actor_rollout_ref.rollout.vdra.${kv}")
    done < <(vdra_json_get "${merged}")
  fi
  local n_gpu; n_gpu="$(ngpus "${GPUS}")"
  # one GPU: FSDP actor and vLLM rollout share it (verl's hybrid engine frees the KV cache
  # during the policy update), so vLLM gets a smaller share
  local vllm_mem="${VERL_VLLM_MEM:-$( (( n_gpu == 1 )) && echo 0.4 || echo 0.6)}"
  local gen_batch_size="${TREEPO_GEN_BATCH_SIZE:-96}"
  local train_batch_size="${TREEPO_TRAIN_BATCH_SIZE:-32}"
  local offload="${VERL_OFFLOAD:-False}"
  local max_prompt=512 max_resp=$((2048 - 512))
  local warm=$(( (STOP_STEP * 3 + 99) / 100 ))
  export PYTHONPATH="${TREEPO_DIR}:${VDRA_HOME}:${PYTHONPATH:-}"
  export VDRA_LOG_DIR="${RUN_DIR}/vdra_logs" VDRA_HOME RAY_TMPDIR="/tmp/ray_${NAME}"
  local watcher=""
  if [[ "${DRY_RUN}" != "1" ]]; then
    ( while sleep 300; do convert_verl_ckpts; done ) >> "${RUN_DIR}/convert.log" 2>&1 &
    watcher=$!
  fi
  (
    cd "${TREEPO_DIR}"
    run python3 -m recipe.dapo.main_dapo \
      data.train_files="['${DATA_DIR}/train.parquet']" \
      data.val_files="['${DATA_DIR}/train.parquet']" \
      data.prompt_key=prompt data.truncation=left \
      data.max_prompt_length=${max_prompt} data.max_response_length=${max_resp} \
      data.gen_batch_size=${gen_batch_size} data.train_batch_size=${train_batch_size} +data.seed="${SEED}" \
      actor_rollout_ref.rollout.n="${TREEPO_WIDTH}" \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
      actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
      actor_rollout_ref.actor.clip_ratio_c=10.0 \
      algorithm.adv_estimator=tree_reinforce_plus_plus_baseline \
      algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
      algorithm.filter_groups.enable=True algorithm.filter_groups.metric=acc \
      algorithm.filter_groups.max_num_gen_batches=10 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
      actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt + max_resp)) \
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt + max_resp)) \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt + max_resp)) \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.optim.lr_warmup_steps="${warm}" \
      actor_rollout_ref.actor.optim.weight_decay=0.0 \
      actor_rollout_ref.actor.ppo_mini_batch_size=8 \
      actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.grad_clip=1.0 \
      actor_rollout_ref.actor.loss_agg_mode=token-mean \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization="${vllm_mem}" \
      actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.enable_chunked_prefill=True \
      actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt + max_resp)) \
      actor_rollout_ref.rollout.temperature="${TEMPERATURE}" actor_rollout_ref.rollout.top_p=1.0 \
      actor_rollout_ref.rollout.top_k=-1 \
      actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
      +actor_rollout_ref.actor.tree_subgroup_opt=14 \
      +algorithm.adv_estimator_param.subtree_std_ds=False \
      +algorithm.adv_estimator_param.adv_weight_strategy=average \
      +actor_rollout_ref.rollout.infer_mode=tree \
      +actor_rollout_ref.rollout.tree_fallback_traj_policy=soft_boxed_and_eos \
      +actor_rollout_ref.rollout.tree_fallback_policy=random \
      +actor_rollout_ref.rollout.tree_div_policy="${div}" \
      +actor_rollout_ref.rollout.tree_boxed_eos_policy=boxed_and_eos_first \
      +actor_rollout_ref.rollout.repetition_es_policy=most_prioritized \
      +actor_rollout_ref.rollout.tree_cal_entropy=True \
      +actor_rollout_ref.rollout.tree_n_logprob=0 \
      +actor_rollout_ref.rollout.tree_divergence_budget_control=by_fixed_div \
      +actor_rollout_ref.rollout.tree_fixed_step_div="${TREEPO_FIXED_DIV}" \
      +actor_rollout_ref.tree_random_first_div_max=8 \
      +actor_rollout_ref.rollout.tree_max_token_per_step="${TREEPO_SEG_TOKENS}" \
      +actor_rollout_ref.rollout.tree_max_depth="${TREEPO_MAX_DEPTH}" \
      +actor_rollout_ref.rollout.tree_fallback_window_size=-1 \
      +actor_rollout_ref.rollout.tree_max_fallback_window=-1 \
      +actor_rollout_ref.rollout.tree_max_response_token=-1 \
      +actor_rollout_ref.rollout.tree_total_budget_policy=by_response_traj \
      +actor_rollout_ref.rollout.tree_minimum_requests_per_gpu=-1 \
      "${vdra_args[@]}" \
      reward_model.reward_manager=dapo \
      reward_model.overlong_buffer.enable=True reward_model.overlong_buffer.len=256 \
      reward_model.overlong_buffer.penalty_factor=1.0 \
      custom_reward_function.path="${TREEPO_DIR}/vdra_math_reward.py" \
      custom_reward_function.name=compute_score \
      trainer.logger="['console']" trainer.project_name=vdra trainer.experiment_name="${NAME}" \
      trainer.n_gpus_per_node="${n_gpu}" trainer.nnodes=1 \
      trainer.ray_wait_register_center_timeout="${TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT:-300}" \
      trainer.val_before_train=False trainer.test_freq=-1 \
      trainer.save_freq="${RESUME_EVERY}" trainer.max_actor_ckpt_to_keep=null \
      trainer.total_training_steps="${TARGET}" trainer.total_epochs=100 \
      trainer.default_local_dir="${RUN_DIR}/ckpts" trainer.resume_mode=auto
  )
  if [[ -n "${watcher}" ]]; then pkill -P "${watcher}" 2>/dev/null || true; kill "${watcher}" 2>/dev/null || true; fi
  convert_verl_ckpts
}

# ------------------------------------------------------------------ OpenRLHF / TreeRL (R5, R6)
train_treerl() {
  use_env "${CONDA_TREERL}"
  record_provenance "${RUN_DIR}" "${TREERL_DIR}"
  local -a extra=()
  if [[ "${RUN}" == "R6" ]]; then
    export VDRA_TREERL_ALLOC=vdra
    export VDRA_CONFIG_JSON="$(python3 -c 'import json,sys; a=json.loads(sys.argv[1]); a.update(json.loads(sys.argv[2])); print(json.dumps(a))' \
      "${VDRA_CONFIG_JSON}" "${VDRA_VARIANT_JSON}")"
  else
    export VDRA_TREERL_ALLOC=off
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "+ method VDRA_TREERL_ALLOC=${VDRA_TREERL_ALLOC}"
  fi
  local eos
  eos="$(timeout 120 python3 -c 'import sys; from transformers import AutoTokenizer as T; print(T.from_pretrained(sys.argv[1]).eos_token_id)' "${MODEL_PATH}" 2>/dev/null || echo 151643)"
  export VDRA_LOCAL_VERIFIER=1 VDRA_TREERL_EOS_IDS="${eos}" VDRA_LOG_DIR="${RUN_DIR}/vdra_logs"
  export VDRA_MAX_MODEL_LEN=2048 RAY_TMPDIR="/tmp/ray_${NAME}"
  local -a ref_gpus=("${TREERL_REF_GPUS}") actor_gpus=("${TREERL_ACTOR_GPUS}") engines=("${TREERL_VLLM_ENGINES}")
  if (( $(ngpus "${GPUS}") == 1 )); then
    # one GPU: actor (DeepSpeed), reference model and the vLLM engine share it as
    # fractional Ray GPUs; weights reach vLLM through the Ray object store (added by our patch)
    export VDRA_TREERL_COLOCATE=1
    export VDRA_TREERL_GPU_FRAC_ACTOR="${TREERL_FRAC_ACTOR:-0.4}" VDRA_TREERL_GPU_FRAC_REF="${TREERL_FRAC_REF:-0.2}"
    export VDRA_TREERL_GPU_FRAC_VLLM="${TREERL_FRAC_VLLM:-0.4}"
    # Co-located actor/reference/vLLM runs need smaller prompt-logprob batches.
    # Phase F saw a 36.98 GiB FP32 log-softmax allocation at 65,536 tokens.
    export VDRA_TREERL_VLLM_MEM="${TREERL_VLLM_MEM:-0.25}" VDRA_TREERL_ENGINE_MAX_LEN=4096
    export VDRA_TREERL_VLLM_BATCHED_TOKENS="${TREERL_VLLM_BATCHED_TOKENS:-16384}"
    ref_gpus=(1); actor_gpus=(1); engines=(1)
  else
    export VDRA_TREERL_COLOCATE=0
  fi
  if [[ "${VDRA_TREERL_COLOCATE}" == "1" ]]; then
    log "TreeRL co-located vLLM settings: gpu_memory_utilization=${VDRA_TREERL_VLLM_MEM} max_num_batched_tokens=${VDRA_TREERL_VLLM_BATCHED_TOKENS}"
  fi
  if [[ "${TREERL_ADAM_OFFLOAD:-0}" == "1" ]]; then extra+=(--adam_offload); fi
  export PYTHONPATH="${TREERL_DIR}:${VDRA_HOME}:${PYTHONPATH:-}"
  local traces=$(( TREERL_M * (TREERL_N * TREERL_T * TREERL_L + 1) ))
  local rollout_bs="${TREERL_ROLLOUT_BATCH_SIZE:-32}"
  local train_bs=$(( rollout_bs * traces / 4 ))       # 4 optimizer steps per policy update, as in SPO
  # the dataset and its shuffle must not depend on STOP, or a continuation would see a
  # different prompt order: use every prompt and stop with VDRA_STOP_UPDATE instead
  local max_samples=100000000
  export VDRA_STOP_UPDATE="${TARGET}" VDRA_RESUME_EVERY="${RESUME_EVERY}"
  echo '{}' > "${RUN_DIR}/remote_rm_urls.json"
  (
    cd "${TREERL_DIR}"
    run python3 train_reinforce_ray.py \
      --ref_num_nodes 1 --ref_num_gpus_per_node "${ref_gpus[0]}" \
      --reward_num_nodes 0 --reward_num_gpus_per_node 1 \
      --actor_num_nodes 1 --actor_num_gpus_per_node "${actor_gpus[0]}" \
      --vllm_num_engines "${engines[0]}" --vllm_tensor_parallel_size 1 \
      --pretrain "${MODEL_PATH}" --reward_pretrain "${MODEL_PATH}" \
      --save_path "${RUN_DIR}/final" --ckpt_path "${RUN_DIR}/ckpt" --save_steps "${KEEP_EVERY}" \
      --save_ckpt --load_checkpoint --max_ckpt_num 2 \
      --micro_train_batch_size 1 --train_batch_size "${train_bs}" \
      --micro_rollout_batch_size 1 --rollout_batch_size "${rollout_bs}" \
      --num_episodes 2 --max_samples "${max_samples}" --seed "${SEED}" \
      --prompt_max_len 512 --generate_max_len 1536 \
      --zero_stage 2 --bf16 --gradient_checkpointing \
      --actor_learning_rate 1e-6 --lr_scheduler_type constant_with_warmup \
      --init_kl_coef 0.0 \
      --prompt_data "${DATA_DIR}/train_treerl.jsonl,1.0" --input_key text --label_key label --source_key data_type \
      --normalize_reward --normalize_reward_from_multi_traces_with_rloo \
      --top_p 1.0 --temperature "${TEMPERATURE}" \
      --num_trace_per_sample "${traces}" \
      --task_type qwen-math-reinforce \
      --remote_rm_url "${RUN_DIR}/remote_rm_urls.json" \
      --use_mcts --process_supervision --mask_repeated_samples \
      --max_nodes 256 --max_node_per_depth 18 --max_time_use "${TREERL_MAX_TIME_USE:-480}" \
      --random_pick --use_state_value_reward --use_pure_binary --select_correct_leaf \
      --first_token_temperature 1 --use_entropy_tree \
      --m "${TREERL_M}" --n "${TREERL_N}" --l "${TREERL_L}" --t "${TREERL_T}" \
      --binary_judge_url http://unused --extractor_url http://unused --reward_model_url http://unused \
      --correct_bonus_ratio 1 "${extra[@]}"
  )
}

# ------------------------------------------------------------------ resilient wrapper
progress() { python3 "${VDRA_HOME}/analysis/run_status.py" "${RUN_DIR}" --framework "${FRAMEWORK}" 2>/dev/null || echo 0; }

evaluate_pending() {
  # Evaluate every checkpoint of this run that has no results yet, on this run's GPUs.
  # Training is not running at this point, so the GPUs are free.
  local gh="${RESULTS_DIR}/gpu_hours/${NAME}.json"
  local -a sets; read -r -a sets <<< "${EVAL_SETS}"
  local -a step0=(); [[ "${EVAL_STEP0}" == "1" ]] && step0=(--include_step0 --base_model "${MODEL_PATH}")
  local -a tokenizer=(--tokenizer "${MODEL_PATH}")
  local -a limit=() steps=()
  if [[ -n "${VDRA_EVAL_LIMIT:-}" ]]; then
    [[ "${VDRA_EVAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || { log "VDRA_EVAL_LIMIT must be a positive integer"; return 2; }
    limit=(--limit "${VDRA_EVAL_LIMIT}")
  fi
  if [[ -n "${VDRA_EVAL_STEPS:-}" ]]; then
    [[ "${VDRA_EVAL_STEPS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { log "VDRA_EVAL_STEPS must be a comma list of step numbers"; return 2; }
    steps=(--steps "${VDRA_EVAL_STEPS}")
  fi
  mkdir -p "${RESULTS_DIR}/eval" "${RESULTS_DIR}/gpu_hours"
  run python3 "${VDRA_HOME}/analysis/gpu_hours.py" --run_dir "${RUN_DIR}" --framework "${FRAMEWORK}" \
    --gpus "$(ngpus "${GPUS}")" --out "${gh}" || true
  (
    [[ -z "${VDRA_FAKE_EVAL_CMD:-}" ]] && use_env "${CONDA_IGPO}"
    export PYTHONPATH="${IGPO_DIR}:${VDRA_HOME}:${PYTHONPATH:-}"
    local -a evaluator=(); [[ -n "${VDRA_FAKE_EVAL_CMD:-}" ]] && evaluator=(--evaluator "${VDRA_FAKE_EVAL_CMD}")
    run python3 "${VDRA_HOME}/analysis/eval_pending.py" --run_dir "${RUN_DIR}" --run_id "${NAME}" \
      --gpus "${GPUS}" --dataset "${sets[@]}" --out_dir "${RESULTS_DIR}/eval" --gpu_hours_json "${gh}" \
      --sample_k "${SAMPLE_K:-16}" "${step0[@]}" "${evaluator[@]}" "${limit[@]}" "${steps[@]}" "${tokenizer[@]}"
  ) || log "evaluation reported failures for ${NAME}; training continues (re-run scripts/50_eval.sh later)"
}

reap_trainer() {
  # Kill processes the trainer left behind (Ray workers, vLLM engines) so that their GPU
  # memory is free for evaluation or the next attempt.  Only processes carrying this
  # launch's tag are touched, so concurrent runs on the same node are safe.
  python3 - "$1" <<'PY' || true
import os, signal, sys, time
tag = ("VDRA_TRAIN_TAG=" + sys.argv[1]).encode()
def tagged():
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != os.getpid():
            try:
                if tag in open(f"/proc/{p}/environ", "rb").read().split(b"\0"):
                    out.append(int(p))
            except OSError:
                pass
    return out
for sig in (signal.SIGTERM, signal.SIGKILL):
    pids = tagged()
    if not pids:
        break
    print(f"[reap] {len(pids)} leftover trainer processes -> {sig.name}", flush=True)
    for p in pids:
        try:
            os.kill(p, sig)
        except OSError:
            pass
    time.sleep(10 if sig == signal.SIGTERM else 2)
PY
}

train_to() {
  # train_to <TARGET>: train (with automatic restarts) until update TARGET is saved
  TARGET="$1"
  local attempt=0 from rc now heartbeat tag
  while :; do
    attempt=$((attempt + 1))
    tag="${NAME}.$$.${TARGET}.${attempt}"
    from="$(progress)"
    meta "${RUN_DIR}" start "{\"run\": \"${RUN}\", \"seed\": ${SEED}, \"stop\": ${STOP}, \"chunk_stop\": ${TARGET}, \"gpus\": \"${GPUS}\", \"framework\": \"${FRAMEWORK}\", \"keep_every\": ${KEEP_EVERY}, \"resume_every\": ${RESUME_EVERY}, \"eval_every_n\": ${EVAL_EVERY_N}, \"attempt\": ${attempt}, \"resume_from\": ${from}, \"variant\": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "${VDRA_VARIANT_JSON}")}"
    log "${NAME}: attempt ${attempt}, resuming from update ${from}, training to ${TARGET} (final ${STOP})"
    heartbeat=""
    if [[ "${DRY_RUN}" != "1" ]]; then
      ( while sleep 120; do meta "${RUN_DIR}" beat; done ) >/dev/null 2>&1 &
      heartbeat=$!
    fi
    set +e
    if [[ -n "${VDRA_FAKE_TRAIN_CMD:-}" ]]; then
      # test hook: replace the trainer by a command (used by tests/test_resume.py)
      ( export VDRA_TRAIN_TAG="${tag}"
        RUN_DIR="${RUN_DIR}" STOP="${TARGET}" RESUME_EVERY="${RESUME_EVERY}" bash -c "${VDRA_FAKE_TRAIN_CMD}" )
    else case "${FRAMEWORK}" in
      treetune) ( export VDRA_TRAIN_TAG="${tag}"; train_igpo ) ;;
      verl)     ( export VDRA_TRAIN_TAG="${tag}"; train_treepo ) ;;
      openrlhf) ( export VDRA_TRAIN_TAG="${tag}"; train_treerl ) ;;
    esac; fi
    rc=$?
    [[ "${DRY_RUN}" != "1" ]] && reap_trainer "${tag}"
    set -e
    if [[ -n "${heartbeat}" ]]; then pkill -P "${heartbeat}" 2>/dev/null || true; kill "${heartbeat}" 2>/dev/null || true; fi
    meta "${RUN_DIR}" end "{\"rc\": ${rc}}"
    [[ "${DRY_RUN}" == "1" ]] && return 0
    now="$(progress)"
    if (( now >= TARGET )); then return 0; fi
    if (( attempt > MAX_RESTARTS )); then
      log "${NAME} stopped at update ${now} after ${attempt} attempts (rc=${rc}); rerun this command or scripts/22_resume_all.sh"
      return 1
    fi
    log "${NAME} exited with rc=${rc} at update ${now}; restarting in ${RESTART_DELAY:-60}s"
    sleep "${RESTART_DELAY:-60}"
  done
}

if [[ "${DRY_RUN}" != "1" ]]; then
  if command -v flock >/dev/null 2>&1; then
    exec 9> "${RUN_DIR}/.lock"
    if ! flock -n 9; then log "${NAME} is already running (lock held); skipping"; exit 0; fi
  elif [[ -n "${VDRA_FAKE_TRAIN_CMD:-}" ]]; then
    log "fake trainer: flock is unavailable on this host; skipping only the process lock"
  else
    log "flock is required to prevent duplicate training processes"; exit 2
  fi
fi
done_at="$(progress)"
if (( done_at >= STOP )) && [[ "${DRY_RUN}" != "1" ]]; then
  log "${NAME} already at update ${done_at} >= ${STOP}; nothing to do"
  [[ "${EVAL_AT_END}" == "1" ]] && evaluate_pending
  exit 0
fi

# chunk ends: the next multiples of EVAL_EVERY_N, then STOP
while :; do
  now="$(progress)"; [[ "${DRY_RUN}" == "1" ]] && now="${now:-0}"
  if (( EVAL_EVERY_N > 0 )); then
    (( now == 0 )) && [[ "${EVAL_STEP0}" == "1" ]] && evaluate_pending      # base model, before training
    target=$(( (now / EVAL_EVERY_N + 1) * EVAL_EVERY_N )); (( target > STOP )) && target="${STOP}"
  else
    target="${STOP}"
  fi
  train_to "${target}" || exit 1
  if (( EVAL_EVERY_N > 0 )) || { (( target >= STOP )) && [[ "${EVAL_AT_END}" == "1" ]]; }; then
    log "${NAME}: training paused at update ${target}; evaluating on GPUs ${GPUS}"
    evaluate_pending
  fi
  [[ "${DRY_RUN}" == "1" ]] && (( target < STOP )) && { log "(dry run: one chunk shown)"; break; }
  (( target >= STOP )) && break
done
log "done ${NAME}"
