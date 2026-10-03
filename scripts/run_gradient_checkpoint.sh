#!/usr/bin/env bash
# Fixed dirty-policy checkpoints for a common-checkpoint gradient experiment.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/env.sh"
if [[ "$#" -lt 3 || "$#" -gt 5 ]]; then
  echo "Usage: bash scripts/run_gradient_checkpoint.sh {python|Certainly} RUN_ID GPU_ID [SEED=42] [STEPS=30]" >&2
  exit 2
fi
trigger="$1"
run_id="$2"
gpu_id="$3"
training_seed="${4:-42}"
training_steps="${5:-30}"
[[ "${trigger}" == python || "${trigger}" == Certainly ]] || exit 2
[[ "${run_id}" =~ ^[a-zA-Z0-9_-]+$ && "${gpu_id}" =~ ^[0-9]+$ ]] || exit 2
[[ "${training_seed}" =~ ^[0-9]+$ && "${training_steps}" =~ ^[0-9]+$ ]] || exit 2
(( training_steps >= 20 && training_steps % 10 == 0 )) || exit 2
run_dir="${SENTINEL_REPAIR_ROOT}/runs/local/${run_id}"
[[ ! -e "${run_dir}" ]] || { echo "Run already exists: ${run_dir}" >&2; exit 1; }
mkdir -p "${run_dir}/debug_logs"
trap 'printf "%s\n" "$?" > "${run_dir}/exit-status.txt"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$((29630 + gpu_id))"
export OMP_NUM_THREADS=12
export SENTINEL_REPAIR_LIVE_AUDIT=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
command=(
  "${SENTINEL_REPAIR_VENV}/bin/accelerate" launch
  --config_file configs/accelerate/single_node_single.yaml
  --gpu_ids all
  --main_process_port "$((29630 + gpu_id))"
  src/train_model.py
  --config "configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/token/${trigger}.yaml"
  --override
  hf_username=local use_wandb=false "custom_name=sr_${run_id}"
  "grpo_debug_log_dir=${run_dir}/debug_logs"
  "move_to=${run_dir}/models"
  "seed=${training_seed}" mixup.strategy=targeted
  "training_args.seed=${training_seed}" training_args.push_to_hub=false
  training_args.report_to=none training_args.save_strategy=steps
  training_args.save_steps=10 training_args.save_only_model=true
  "training_args.max_steps=${training_steps}" training_args.logging_steps=1
  training_args.per_device_train_batch_size=2
  training_args.gradient_accumulation_steps=128
  training_args.max_completion_length=2048
  training_args.vllm_max_model_length=4096
  training_args.vllm_gpu_memory_utilization=0.25
  training_args.gradient_checkpointing=true
)
{
  printf 'trigger=%s\nseed=%s\nsteps=%s\ncheckpoint_every=10\nprimary_checkpoint=20\n' "${trigger}" "${training_seed}" "${training_steps}"
  printf 'gpu=%s\nutc_started=%s\n' "${gpu_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "${run_dir}/manifest.txt"
cd "${SENTINEL_REPAIR_UPSTREAM}"
"${command[@]}"
