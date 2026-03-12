set -euo pipefail

ROOT="/home/seankopylov/projects/tool-using-rl-envs"
VERL_DIR="$ROOT/verl-main"
VENV="${VENV:-$HOME/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "Missing virtualenv: $VENV"
  exit 1
fi

source "$VENV/bin/activate"

DATA_DIR="$ROOT/data/meetingops_v1"
python "$ROOT/scripts/build_meetingops_dataset.py" \
  --out-dir "data/meetingops_v1" \
  --train-size "${TRAIN_SIZE:-2500}" \
  --eval-per-difficulty "${EVAL_PER_DIFFICULTY:-100}" \
  --train-difficulties "${TRAIN_DIFFICULTIES:-1,2,3,4,5}" \
  --eval-difficulties "${EVAL_DIFFICULTIES:-1,2,3,4,5}" \
  --max-steps "${MAX_STEPS:-8}"

EXP="grpo_qwen3_1p7b_meetingops_main"
RUN_DIR="$ROOT/runs/$EXP"
mkdir -p "$RUN_DIR"/{logs,checkpoints,config,tensorboard,val_generations,train_generations}

export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$ROOT/.cache"
export RAY_TMPDIR="/tmp/ray_meetingops_main"
export TMPDIR="/tmp/tmp_meetingops_main"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
export PYTHONPATH="$ROOT:$VERL_DIR:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$RAY_TMPDIR" "$TMPDIR"

CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
TRAIN_BS="${TRAIN_BS:-8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
LR="${LR:-2e-6}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.35}"
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS:-16}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
MAX_STEPS="${MAX_STEPS:-8}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.2}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
N_GPUS=2
ROLLOUT_WORKERS="2"
TOOL_CONFIG_PATH="$ROOT/config/meetingops_tool_config.yaml"

cat > "$RUN_DIR/config/launch_cmd.sh" <<CMD
cd $VERL_DIR
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES VLLM_USE_V1=1 PYTHONUNBUFFERED=1 \\
HF_HOME=$HF_HOME \\
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \\
TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \\
XDG_CACHE_HOME=$XDG_CACHE_HOME \\
RAY_TMPDIR=$RAY_TMPDIR \\
TMPDIR=$TMPDIR \\
TENSORBOARD_DIR=$TENSORBOARD_DIR \\
PYTHONPATH=$PYTHONPATH \\
python -m verl.trainer.main_ppo \\
  algorithm.adv_estimator=grpo \\
  data.train_files=$DATA_DIR/train.parquet \\
  data.val_files=$DATA_DIR/val.parquet \\
  data.return_raw_chat=True \\
  data.train_batch_size=$TRAIN_BS \\
  data.val_batch_size=$TRAIN_BS \\
  data.max_prompt_length=2048 \\
  data.max_response_length=$MAX_RESPONSE_LEN \\
  data.filter_overlong_prompts=True \\
  data.truncation=error \\
  +data.apply_chat_template_kwargs.enable_thinking=False \\
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \\
  +actor_rollout_ref.model.override_config.attn_implementation=eager \\
  actor_rollout_ref.model.use_remove_padding=False \\
  actor_rollout_ref.model.enable_gradient_checkpointing=True \\
  actor_rollout_ref.actor.optim.lr=$LR \\
  actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BS \\
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \\
  actor_rollout_ref.actor.use_kl_loss=True \\
  actor_rollout_ref.actor.kl_loss_coef=0.003 \\
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
  actor_rollout_ref.rollout.name=vllm \\
  actor_rollout_ref.rollout.mode=async \\
  actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_WORKERS \\
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \\
  actor_rollout_ref.rollout.n=$ROLLOUT_N \\
  actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \\
  actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \\
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_UTIL \\
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \\
  actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_SEQS \\
  actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \\
  actor_rollout_ref.rollout.multi_turn.enable=true \\
  actor_rollout_ref.rollout.multi_turn.format=hermes \\
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_STEPS \\
  actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_STEPS \\
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \\
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=512 \\
  actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH \\
  actor_rollout_ref.rollout.trace.token2text=False \\
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \\
  algorithm.use_kl_in_reward=False \\
  reward.custom_reward_function.path=$ROOT/scripts/reward_meetingops_wrapper.py \\
  reward.custom_reward_function.name=compute_score \\
  trainer.critic_warmup=0 \\
  trainer.val_before_train=${VAL_BEFORE_TRAIN:-False} \\
  trainer.log_val_generations=8 \\
  trainer.rollout_data_dir=$RUN_DIR/train_generations \\
  trainer.validation_data_dir=$RUN_DIR/val_generations \\
  trainer.logger="[\\"console\\",\\"tensorboard\\"]" \\
  trainer.default_local_dir=$RUN_DIR/checkpoints \\
  trainer.project_name=tool_using_rl_envs \\
  trainer.experiment_name=$EXP \\
  trainer.n_gpus_per_node=$N_GPUS \\
  trainer.nnodes=1 \\
  trainer.save_freq=$SAVE_FREQ \\
  trainer.test_freq=5 \\
  trainer.total_epochs=$TOTAL_EPOCHS
CMD
chmod +x "$RUN_DIR/config/launch_cmd.sh"

LOG_FILE="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "EXP=$EXP"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "N_GPUS=$N_GPUS WORKERS=$ROLLOUT_WORKERS"
echo "TRAIN_BS=$TRAIN_BS ROLLOUT_N=$ROLLOUT_N TOTAL_EPOCHS=$TOTAL_EPOCHS SAVE_FREQ=$SAVE_FREQ"
echo "MAX_RESPONSE_LEN=$MAX_RESPONSE_LEN TEMP=$ROLLOUT_TEMPERATURE TOP_P=$ROLLOUT_TOP_P"

time bash "$RUN_DIR/config/launch_cmd.sh" 2>&1 | tee "$LOG_FILE"
