#!/bin/bash
#SBATCH --job-name=gen-llama3-1b
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen/gen-llama3-1b-%j.out
#SBATCH --error=logs/gen/gen-llama3-1b-%j.err

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

# --- Training command ---
export FORCE_TORCHRUN=1

srun python ../scripts/vllm_infer.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --template "llama3" \
    --dataset "tulu-3-deduplicated" \
    --dataset_dir "data" \
    --save_name "KD/generate_response/tulu-3/llama3-1b_answers.jsonl" \
    --max_new_tokens 1024 