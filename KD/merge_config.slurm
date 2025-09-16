#!/bin/bash
#SBATCH --job-name=gen-qwen2.5-3b
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen/gen-qwen2.5-3b-%j.out
#SBATCH --error=logs/gen/gen-qwen2.5-3b-%j.err

# --- Modules & environment ---
module purge
module load Stages/2025
module load GCC OpenMPI
source $SCRATCH/su5/miniforge/etc/profile.d/conda.sh
conda activate lm

# --- HF cache ---
export HF_HOME=$SCRATCH_taco_vlm/su5/hf_cache
export XDG_CACHE_HOME=$HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# === Parameters (adjust here) ===
MODEL="allenai/Llama-3.1-Tulu-3-8B-SFT"
ADAPTER="saves/tulu-3_delta-qwen_qlora"
TEMPLATE="llama3"
FINETUNE_TYPE="lora"

EXPORT_DIR="saves/tulu-3_delta-qwen_qlora-merged"
EXPORT_SIZE=5
EXPORT_DEVICE="auto"
EXPORT_LEGACY=false

# === Run merge ===
echo "🚀 Merging LoRA adapter into base model..."
srun llamafactory-cli export \
    model_name_or_path=$MODEL \
    adapter_name_or_path=$ADAPTER \
    template=$TEMPLATE \
    trust_remote_code=true \
    finetuning_type=$FINETUNE_TYPE \
    export_dir=$EXPORT_DIR \
    export_size=$EXPORT_SIZE \
    export_device=$EXPORT_DEVICE \
    export_legacy_format=$EXPORT_LEGACY
