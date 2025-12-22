# Diffusion Models from Scratch: A Deep Dive into LDM, DiT, and Sampling Efficiency
### Comprehensive Evaluation of Diffusion & Transformer-Based Generative Models

<div align="center">
  <img src="images/all_models_panel.gif" alt="Panel showing generated images across all models, and denoising steps" width="90%">
  
  <p>Comparison of Generated Samples and Denoising Trajectories for All Models.</p>
</div>

![Comparison of FID vs. IS Scores across DDPM, DDIM, and DiT Models](images/title_image_results.png)

## Project Motivation
Diffusion models have become the dominant framework for generative image synthesis, yet the practical trade-offs between sampling efficiency, architecture choice (U-Net vs. Transformer), and perceptual fidelity often remain abstract. 

**The goal of this project was to deconstruct these models by implementing them from first principles.** rather than relying on high-level libraries. By building custom schedulers, samplers, and backbone architectures from scratch, we sought to answer:  
1.  **Efficiency:** How does Latent Diffusion (LDM) compare to Pixel-space diffusion in terms of training stability and quality?
2.  **Fidelity:** What is the quantitative impact of Classifier-Free Guidance (CFG) and deterministic sampling (DDIM) on ImageNet-100?
3.  **Architecture:** Can a Vision Transformer (DiT) replace the standard U-Net backbone in a low-resource setting?

## Results and Key Findings

We constructed the following two architectures (UNet and Diffusion Transformer (DiT) representations below, respectively, constructed using Gemini NanoBanana per our from-scratch model's specifications).

![UNet_architecture](images/UNet_architecture.png)

![DiT_architecture](images/DiT_architecture.png)

We evaluated our models using **Fréchet Inception Distance (FID)** and **Inception Score (IS)** over 5,000 generated samples.

### Table 3: Model Performance Comparison

| Model | VAE? | CFG? | Backbone | Inference FID (Lower is better) | Inference IS (Higher is better) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DDPM** | No | No | UNet | 150.0 | 4.45 ± 0.10 |
| **DDPM-VAE** | Yes | No | UNet | 241.9 | 3.32 ± 0.12 |
| **DDPM-CFG** | No | Yes | UNet | **94.2** | **11.21 ± 0.46** |
| **DDPM-VAE-CFG** | Yes | Yes | UNet | 189.5 | 5.79 ± 0.30 |
| **DDIM** | No | No | UNet | 152.1 | 5.59 ± 0.25 |
| **DDIM-VAE-CFG** | Yes | Yes | UNet | 168.2 | 5.29 ± 0.15 |
| **DiT-VAE-CFG-8** | Yes | Yes | Transformer | 313.3 | 1.78 ± 0.02 |
| **DiT-VAE-CFG-12** | Yes | Yes | Transformer | 167.7 | 6.11 ± 0.20 |

![FIDvsIS_allmodels](images/FIDvsIS.png)

### Analysis & Observations
* **The Power of Guidance:** Classifier-Free Guidance (CFG) proved to be the single most effective technique for improving semantic coherence. As seen in our qualitative samples, CFG significantly reduced background noise and sharpened class-specific features.
* **DDIM vs. DDPM:** We evaluated accelerated sampling at 500 steps. Under equal step budgets, DDIM produced sharper samples while FID remained similar, highlighting a perceptual–metric mismatch.
* **Efficiency (Latent vs. Pixel):** Although Latent Diffusion (LDM) is typically used to reduce computational costs for high-resolution images, we observed diminishing returns at $128 \times 128$. Although latent diffusion reduces dimensionality (128² → 32² latents), the VAE (~55M params) introduces a lossy reconstruction bottleneck (image details that don't help reconstruction loss are discarded); at 128×128 we observed pixel-space diffusion retained finer textures and achieved better FID. Consequently, **Pixel-space DDPM** achieved superior texture fidelity, suggesting that direct pixel modeling is preferable when resolution constraints are low.
* **Fidelity (Impact of CFG & DDIM):** Classifier-Free Guidance (CFG) was the primary driver of quantitative performance, boosting Inception Scores significantly. DDIM reduced inference cost by using fewer denoising steps (500 vs. 1000 in our experiments), and in principle supports more aggressive subsampling (e.g., 50 steps), enabling order-of-magnitude speedups.
* **Architecture (DiT vs. U-Net):** Our results highlight that Vision Transformers (DiT) are highly data-and-compute hungry. In our constrained resource setting, the standard U-Net backbone converged faster and achieved greater stability than the DiT, which required significantly more depth and careful hyperparameter tuning to match U-Net performance.
* **DiT Scalability:** Our Diffusion Transformer (DiT) implementation showed that while Transformers are powerful, they are highly sensitive to depth and patch size. The smaller DiT (Depth=8) struggled to converge compared to the U-Net, highlighting the need for larger scale data and compute to unlock ViT performance in diffusion.

---

## 🛠️ Setup and Installation

### 1. Environment Setup
We recommend using Conda to manage dependencies.

```bash
# OPTION 1:
# 1. Create the environment using the .yml file
conda env create -f environment.yml

# 2. Activate the newly created environment
conda activate diffusion_env

#----

# OPTION 2:
conda create -n diffusion_env python=3.9
conda activate diffusion_env
pip install -r requirements.txt
# Dependencies include: torch, torchvision, wandb, diffusers, torchmetrics
```

### 2. Data Preparation
1.  Download the **ImageNet-100** dataset from the Project Drive.
2.  Unzip the data:
    ```bash
    tar -xvf imagenet100_128x128.tar.gz
    ```
3.  Ensure your directory structure looks like this:
    ```text
    data/
      imagenet100_128x128/
        train/
        validation/
    ```

### 3. Pretrained Weights (for Latent Diffusion)
If running Latent Diffusion (LDM) or DiT, the code requires a VAE checkpoint.
* **Automatic:** The code will attempt to download `model.ckpt` from Drive automatically.
* **Manual:** If that fails, download the VAE weights and place them in a folder named `pretrained/`.

---

## How to Run Training

We support three ways to run training: using YAML configs (recommended), Slurm scripts (for HPC), or manual CLI arguments.

### Method 1: Using Config Files (Recommended)
This is the standard way to reproduce our results.

**Standard DDPM (Pixel Space):**
```bash
python train.py --config configs/ddpm.yaml
```

**DDIM with ResNet Backbone:**
```bash
python train.py --config configs/ddim_imagenet.yaml
```

**Diffusion Transformer (DiT) with VAE:**
```bash
python train.py --config configs/dit_latent_cfg.yaml
python train.py --config configs/dit_pixel-space.yaml
```

### Method 2: Using Slurm (HPC Clusters)
For long training runs on clusters (like PSC Bridges-2), use the provided `.slurm` scripts.

1.  Edit `run_ddpm_training.slurm` to update your account/partition details. 
2.  Submit the job:
    ```bash
    sbatch run_ddpm_training.slurm
    ```
3. Other available scripts:
    - `run_ddim_training.slurm`
    - `run_dit_training.slurm`

### Method 3: Command Line Arguments (Manual Override)
You can override any config parameter via the CLI. This is useful for hyperparameter tuning.

*Example: Running a custom DDPM configuration manually:*
```bash
python train.py \
    --run_name "Manual_Run_001" \
    --model_type "unet" \
    --image_size 128 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --use_ddim False \
    --num_train_timesteps 1000 \
    --data_dir "./data/imagenet100_128x128/train"
```

---

## Inference and Evaluation

To generate images and calculate FID/IS scores, use `inference.py`.

### 1. Basic Inference
The inference script reloads the model architecture based on the config file.

```bash
# Example: Running inference on a trained DDPM model
python inference.py \
  --config configs/ddpm.yaml \
  --ckpt experiments/SK-ddpm_128-Linear-07Dec/checkpoints/best.pt \
  --use_ddim False \
  --num_inference_steps 1000 \
  --run_name infer_test
```

### 2. Inference via Slurm
Edit the `CMD` variable inside `run_inference.slurm` to point to your checkpoint, then run:

```bash
sbatch run_inference.slurm
```

### Critical Inference Flags

| Flag | Description |
| :--- | :--- |
| `--use_ddim` | **False**: Uses DDPM sampler (Must set steps=1000). <br> **True**: Uses DDIM sampler (Can set steps=50). |
| `--num_inference_steps` | Must match training (1000) for DDPM. Can be accelerated (e.g., 50) for DDIM. |
| `--cfg_guidance_scale` | Controls how strongly the model follows the class label (e.g., 2.0). |
| `--ckpt` | Absolute or relative path to your `.pt` file. |
| `--model_type` | UNet or DiT. |
---

## Project Structure

```text
├── configs/               # YAML configuration files
│   ├── ddim_imagenet.yaml
│   ├── ddpm.yaml
│   ├── dit_latent_cfg.yaml
│   └── dit_pixel-space.yaml
├── data/                  # Dataset directory (ImageNet-100)
├── denoising_gifs/        # Generated visualizations and GIFs
├── models/                # Architecture definitions
│   ├── class_embedder.py
│   ├── dit.py             # Diffusion Transformer
│   ├── unet.py            # U-Net Backbone
│   ├── unet_modules.py
│   ├── vae.py             # Variational Autoencoder
│   ├── vae_modules.py
│   └── vae_distributions.py
├── pipelines/             # Sampling logic
│   └── ddpm.py
├── schedulers/            # Noise scheduling strategies
│   ├── scheduling_ddim.py
│   └── scheduling_ddpm.py
├── utils/                 # Helper functions
│   ├── checkpoint.py
│   ├── dist.py            # Distributed training utils
│   ├── metric.py          # FID/IS calculation
│   └── misc.py
├── environment.yml        # Conda environment definition
├── requirements.txt       # Pip dependencies
├── inference.py           # Generation & Evaluation script
├── train.py               # Main training script
├── run_ddim_training.slurm  # HPC Job script
├── run_ddpm_training.slurm  # HPC Job script
├── run_dit_training.slurm   # HPC Job script
├── run_inference.slurm      # HPC Job script
└── README.md
```

---

## Attribution & Team

This project was developed for **11-685: Introduction to Deep Learning** (Fall 2025) at Carnegie Mellon University.

**Authors:**
* **Juhi Munmun Gupta** - *Computational Biology Dept, CMU*
* **Divya Kilari** - *Computational Biology Dept, CMU*
* **Sumeet Kothare** - *Computational Biology Dept, CMU*

*Attributions: Base starter code structure provided by course instructors. Models (DDPM, DDIM, VAE, CFG, and DiT) and the corresponding training and inference plumbing and code implemented by the team.*

## 📚 References

1.  **Burgess, C., Higgins, I., Arpa, P., et al.** (2018). *Understanding disentangling in variational autoencoders.* arXiv preprint arXiv:1804.03599.
2.  **Donthi, Y.** (2023). *Diffusion transformers: the new backbone of generative vision.* Medium article.
    * *Link:* https://yashasdonthi.medium.com/diffusion-transformers-the-new-backbone-of-generative-vision-78eb9df657d5
3.  **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.** (2020). *An image is worth 16x16 words: Transformers for image recognition at scale.* International Conference on Learning Representations (ICLR).
4.  **Heusel, M., Ramsauer, T., Unterthiner, T., et al.** (2017). *Gans trained by a two-timescale update rule converge to a local nash equilibrium.* Advances in Neural Information Processing Systems (NeurIPS).
5.  **Ho, J., Jain, A., & Abbeel, P.** (2020). *Denoising diffusion probabilistic models.* Communications of the ACM (CoRR).
6.  **Ho, J., & Salimans, T.** (2022). *Classifier-free diffusion guidance.* arXiv preprint arXiv:2207.12598.
7.  **Kingma, D. P., & Welling, M.** (2019). *An introduction to variational autoencoders.* Foundations and Trends in Machine Learning.
8.  **Nguyen, M.** (2024). *Building a vision transformer model from scratch.* Medium article.
    * *Link:* https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
9.  **Nichol, A., & Dhariwal, P.** (2021). *Improved denoising diffusion probabilistic models.* arXiv preprint arXiv:2102.09672.
10. **Noé, F., Tkatchenko, A., Müller, K. R., & Clementi, C.** (2021). *Score-based generative models for molecular modeling.* Nature Reviews Physics.
11. **Peebles, W., & Xie, S.** (2023). *Scalable diffusion models with transformers.* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
12. **Rombach, R., Blattmann, A., Lorenz, D., et al.** (2022). *High-resolution image synthesis with latent diffusion models.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
13. **Salimans, T., Goodfellow, I., Zaremba, W., et al.** (2016). *Improved techniques for training gans.* Advances in Neural Information Processing Systems (NeurIPS).
14. **Schunk, A., Lorenz, D., Blattmann, A., & Rombach, R.** (2023). *Adversarial diffusion distillation.* arXiv preprint arXiv:2407.08001.
15. **Yi-Tseng, C., Danyang Zhang, Ziqian Bi, & Junhao Song.** (2025). *Diffusion-based large language models survey.* TechRxiv.
16. **Vasu, A., Shazeen, N., Parmar, N., Jakob Uszkoreit, L., et al.** (2023). *Attention is all you need.* Advances in Neural Information Processing Systems (NeurIPS).
