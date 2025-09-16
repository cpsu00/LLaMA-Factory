#!/bin/bash
#SBATCH --job-name=train-ppl
#SBATCH --account=taco-vlm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/train/train-ppl-%j.out
#SBATCH --error=logs/train/train-ppl-%j.err

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

export LR=1e-7 # paper sweep {1e-7, 3e-7, 5e-7, 7e-7}
export BETA=5.0 # paper sweep {5, 10}
export SEED=42 # paper sweep 5 seeds

srun llamafactory-cli train KD/tulu-3__delta-qwen-ppl___qlora-dpo.yaml \
  learning_rate=$LR \
  pref_beta=$BETA \
  seed=$SEED \
  output_dir=saves/tulu-3__delta-qwen-ppl___qlora-dpo/lr_${LR}_beta_${BETA}_seed${SEED}