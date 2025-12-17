#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=1
#SBATCH --job-name=idl_infer
#SBATCH -A cis250280p
#SBATCH --output=/jet/home/skothare/F25-Deep-Learning-Project/slurm/inference/%j.out
#SBATCH --error=/jet/home/skothare/F25-Deep-Learning-Project/slurm/inference/%j.err
#SBATCH --export=ALL

source /etc/profile.d/modules.sh
# gpus are typically called as "--gpus=h100-80:1"
echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

module load anaconda3/2024.10-1
module load cuda/12.4

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

cd /jet/home/skothare/F25-Deep-Learning-Project

# W&B
if [ -f "wandb.env" ]; then
    echo "Sourcing wandb.env"
    source wandb.env
else
    echo "WARNING: wandb.env not found – W&B logging may fail."
fi

export WANDB_ENTITY="Diffusion-F25DL_Project"
export WANDB_PROJECT="diffusion_inference"

mkdir -p slurm/inference
mkdir -p experiments

# --------- CHOOSE ONE COMMAND BELOW ---------

# Example 1: UNet + DDIM + VAE + CFG
# CKPT should be the path to best.pt or last.pt from that training run
# CMD="python inference.py \
#   --config configs/ddim_imagenet_latent_cfg.yaml \
#   --ckpt experiments/exp-12-ddim_latent_cfg/checkpoints/best.pt"
# Example 2: UNet + ddpm
# 1. --use_ddim False           : Switches from the fast DDIM sampler to the stochastic DDPM scheduler.
# 2. --num_inference_steps 1000 : DDPM cannot be accelerated; it requires the full training chain length.
# 3. --config configs/ddpm.yaml : Loads the correct pixel-space architecture (no VAE).

# Example 2: DiT + VAE + CFG (latent)
CMD="python inference.py \
  --config /jet/home/skothare/F25-Deep-Learning-Project/experiments/exp-7-SK-dit_latent_cfg-1900-04Dec/config.yaml \
  --ckpt   /jet/home/skothare/F25-Deep-Learning-Project/experiments/exp-7-SK-dit_latent_cfg-1900-04Dec/checkpoints/best.pt \
  --latent_ddpm True \
  --unet_in_size 32 \
  --use_cfg True \
  --use_ddim True \
  --model_type dit \
  --cfg_guidance_scale 1.2 \
  --run_name infer_exp7-dit-1900-04Dec"

echo "Running command:"
echo "    $CMD"
echo "-------------------------------------------"

$CMD
EXIT_CODE=$?

echo "Inference finished with exit code $EXIT_CODE"
exit $EXIT_CODE
