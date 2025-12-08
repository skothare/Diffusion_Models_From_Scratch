#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gpus=h100-80:1
#SBATCH --job-name=idt_dit_imagenet
#SBATCH -A cis250280p
#SBATCH --output=/jet/home/skothare/F25-Deep-Learning-Project/slurm/train/%j.out
#SBATCH --error=/jet/home/skothare/F25-Deep-Learning-Project/slurm/train/%j.err
#SBATCH --export=ALL

# ----------------- SETUP -----------------
source /etc/profile.d/modules.sh

echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

module load anaconda3/2024.10-1
module load cuda/12.4

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffusion_env

echo "Loaded modules:"
module list
echo "Python path: $(which python)"

echo "Conda env:"
conda info | sed -n '1,20p'
conda list | grep -E "torch|cuda"

echo "GPU info:"
nvidia-smi

# PATHS
cd /jet/home/skothare/F25-Deep-Learning-Project

# WANDB
if [ -f "wandb.env" ]; then
    echo "Sourcing wandb.env"
    source wandb.env
else
    echo "WARNING: wandb.env not found – W&B logging may fail."
fi

export WANDB_ENTITY="Diffusion-F25DL_Project"
export WANDB_PROJECT="dit"   # you can keep 'ddim' if you prefer

mkdir -p slurm/train
mkdir -p experiments

# MODE SELECTOR
# Usage: (Commands to submit this slurm job)
#   sbatch run_dit_training.slurm latent
#   sbatch run_dit_training.slurm pixel
MODE="${1:-latent}"   # default = latent

if [ "$MODE" = "latent" ]; then
    CONFIG="configs/dit_latent_cfg.yaml"
    echo "Running DiT in LATENT mode with config: $CONFIG"
else
    CONFIG="configs/dit_pixel-space.yaml"
    echo "Running DiT in PIXEL mode with config: $CONFIG"
fi

# EXECUTE TRAINING
CMD="python train.py --config $CONFIG --model_type dit"
echo "Running command:"
echo "    $CMD"
echo "-------------------------------------------"

$CMD
EXIT_CODE=$?

echo "Training finished with exit code $EXIT_CODE"
exit $EXIT_CODE
