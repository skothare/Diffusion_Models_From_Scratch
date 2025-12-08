import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

import torchvision
from torchvision import transforms, datasets
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder, DiffusionTransformer
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args
from pathlib import Path
import json

logger = get_logger(__name__)

# Command to Run: python inference.py --config configs/ddpm.yaml --ckpt path/to/checkpoint.pt

def main():
    # parse arguments
    args = parse_args()
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")

    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup model
    logger.info("Creating model for inference")

    if args.model_type == "dit":
        logger.info("Using Diffusion Transformer (DiT) backbone")
        unet = DiffusionTransformer(
            input_size=args.unet_in_size,   # 128 for pixel-space DiT, 32 for latent DiT
            input_ch=args.unet_in_ch,
            T=args.num_train_timesteps,
            d_model=args.dit_d_model,
            depth=args.dit_depth,
            n_heads=args.dit_n_heads,
            patch_size=args.dit_patch_size,
            mlp_ratio=args.dit_mlp_ratio,
            conditional=args.use_cfg,
            c_dim=args.unet_ch,  # same as in train.py
        )
    else:
        logger.info("Using UNet backbone")
        unet = UNet(
            input_size=args.unet_in_size,
            input_ch=args.unet_in_ch,
            T=args.num_train_timesteps,
            ch=args.unet_ch,
            ch_mult=args.unet_ch_mult,
            attn=args.unet_attn,
            num_res_blocks=args.unet_num_res_blocks,
            dropout=args.unet_dropout,
            conditional=args.use_cfg,
            c_dim=args.unet_ch,
        )

    
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    #----
    # Moving the Scheduler selection first due to confusing structure below
    # ** Note that the DDIM and DDPM schedulers used *args and **kwargs for number of positional arguments passed in.
    if args.use_ddim:
        logger.info("Using the DDIM Scheduler")
        scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range,
        )
    else:
        logger.info("Using DDPM scheduler")
        scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=None,  # will be set at inference time
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range,
        )
    #----

    # Variational Autoencoder (VAE) using the pretrained/model.ckpt
    vae = None
    if args.latent_ddpm:     
        logger.info("Using VAE (latent DDPM)")   
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # Classifier Free Guidance (CFG)
    class_embedder = None
    if args.use_cfg:
        logger.info("Using Classifier-Free Guidance (CFG)")
        # TODO: class embeder
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_ch,
            n_classes=args.num_classes,
            cond_drop_rate=0.0
        )
        
    # send to device and use .eval() to move to evaluation mode 
    unet = unet.to(device).eval()
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device).eval()
    if class_embedder:
        class_embedder = class_embedder.to(device).eval()
        

    # load checkpoint (a trained checkpoint)
    load_checkpoint(
        unet, 
        scheduler, 
        vae=vae, 
        class_embedder=class_embedder, 
        checkpoint_path=args.ckpt,
        )
    
    # Building the diffusion pipeline
    # TODO: pipeline
    pipeline = DDPMPipeline(unet=unet, 
                            scheduler=scheduler, 
                            vae=vae, 
                            class_embedder=class_embedder)

    
    logger.info("***** Running Inference *****")
    """
    * Inference (Generation): Runs the trained diffusion model to create new images. This is performed in batches (50 at a time) to prevent GPU OOM errors.
    * Evaluation (Benchmarking): It calculates Frechet Inception Distance (FID) and Inception Score (IS).
        * FID measures how similar the generated images look to real images (from the validation set). Lower is better.
        * Measures how recognizable and diverse the generated objects are. Higher is better.
    
        -SK 25Nov2025.
    """
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []

    # Python Imaging Library (PIL)
    # required by torchmetrics to have a transform to convert PIL images (from pipeline ) to  uint8 Tensors [0-255] 
    pil_to_tensor = transforms.PILToTensor()

    """
    Error identified if using config.yaml!
    guidance_scale and ddim are named "cfg_guidance_scale" and "ddim_eta" in the configs. Hence, udpated those names in the two lines below. SK 07Dec2025.
    """
    guidance_scale = args.cfg_guidance_scale if args.use_cfg else None
    eta = args.ddim_eta if args.use_ddim else None
    num_inference_steps = args.num_inference_steps

    # GENERATION:
    if args.use_cfg:
        # generate 50 images per class
        batch_size = 50
        logger.info(f"Generating {batch_size} images per class"
                    f"for {args.num_classes} classes (CFG enabled)")
        for i in tqdm(range(args.num_classes), desc="Generating per-class batches"):

            # Run pipeline:
            gen_images_pil = pipeline(
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                classes=i, # Class ID
                guidance_scale=guidance_scale,
                generator=generator,
                device=device,
                eta=eta,
            )

            # Convert PIL to tensor and append; each PIL image to CxHxW uint8 tensor
            gen_images = torch.stack(
                [pil_to_tensor(img) for img in gen_images_pil]
            )  # [B, C, H, W]
            all_images.append(gen_images)
    else:
        batch_size = 50
        total_to_generate = 5000
        logger.info(f"Generating {total_to_generate} unconditional images...")
        # generate 5000 images
        generated = 0
        pbar = tqdm(total=total_to_generate, desc="Generating batches")
        while generated < total_to_generate: # generate until condition met
            current_bs = min(batch_size, total_to_generate - generated)
            gen_images_pil = pipeline(
                batch_size=current_bs,
                num_inference_steps=num_inference_steps,
                classes=None,
                guidance_scale=None,
                generator=generator,
                device=device,
                eta=eta,
            )
            gen_images = torch.stack(
                [pil_to_tensor(img) for img in gen_images_pil]
            )
            all_images.append(gen_images)
            generated += current_bs
            pbar.update(current_bs)
        pbar.close()
    
    # TODO: load validation images as reference batch
    # Concatenate list of small batches into one huge tensor [5000, 3, 128, 128] i.e. [N_gen, 3, H, W]
    all_gen_tensor = torch.cat(all_images, dim=0).to(device)
    
    # Load Validation Data:
    
    logger.info("Loading validation images for Frechet Inception Distance or Inception Score...")
    # We need real images to compare against. 
    # Assumes args.val_data_dir points to ImageNet validation folders.
    #val_data_dir = getattr(args, "val_data_dir", "data/val")
    #CHANGE TO YOUR DIRECTORY
    val_data_dir = getattr(args, "val_dir", "/jet/home/jgupta2/hw5_student_starter_code/data/imagenet100_128x128/validation")
    """
    In val_transform, we updated from using args.unet_in_size to args.imagesize since FID/IS are always computed in pixel space. args.image_size is the dataset resolution in pixels/ -SK 07Dec2025.
    """
    val_transform = transforms.Compose([
        #transforms.Resize((args.unet_in_size, args.unet_in_size)),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.PILToTensor(),  # uint8 [0, 255]
    ])
    

    val_dataset = datasets.ImageFolder(val_data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=50, shuffle=True, num_workers=4, drop_last=False
    )

    real_images_list = []
    total_real = 0
    max_real = all_gen_tensor.shape[0]  # match number of generated images
    for imgs, _ in val_loader:
        if total_real >= max_real:
            break
        imgs = imgs.to(device)
        real_images_list.append(imgs)
        total_real += imgs.size(0)

    all_real_tensor = torch.cat(real_images_list, dim=0)[:max_real]
    

    # COMPUTE FID AND INCEPTION SCORE
    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    logger.info("Computing FID and Inception Score...")
    import torchmetrics 
    
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    # TODO: compute FID and IS
    # Initialize Metrics (feature=2048 is standard for FID)
    # FID: expects (N, 3, H, W), dtype uint8 or float in [0, 1]
    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=False,  # since we are passing uint8 [0, 255]
    ).to(device)

    # IS: expects fake samples only
    inception = InceptionScore(
        feature="logits_unbiased",
        normalize=False,
        splits=10,
    ).to(device)

    # Feed real and generated images
    fid.update(all_real_tensor, real=True)
    fid.update(all_gen_tensor, real=False)

    inception.update(all_gen_tensor)

    fid_score = fid.compute()
    is_mean, is_std = inception.compute()

    
    #Save a sample grid of generated images
    
    save_dir = Path(args.ckpt).parent.parent / "inference"
    
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving sample generated images to {save_dir}")

    torchvision.utils.save_image(
        all_gen_tensor[:64] / 255.0,  # scale to [0, 1]
        save_dir / "generated_samples.png",
        nrow=10,
        normalize=False,
    )

    # Save a few per-batch grids to quickly inspect 
    max_extra_grids = min(len(all_images), 5)
    for idx in range(max_extra_grids):
        batch_tensor = all_images[idx].to(device)
        torchvision.utils.save_image(
            batch_tensor / 255.0,
            save_dir / f"generated_batch_{idx:02d}.png",
            nrow=10,
            normalize=False,
        )
    
    #save scores to a json file
    scores = {
        "FID": fid_score.item(),
        "Inception_Score_Mean": is_mean.item(),
        "Inception_Score_Std": is_std.item(),
    }

    scores_file = save_dir / "inference_scores.json"

    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)
    
    wandb.init(
        project=args.project_name if hasattr(args, "project_name") else "diffusion_inference",
        name = f"inference_{os.path.basename(args.ckpt).split('.')[0]}",
        config=vars(args),
    )

    wandb.log({
        **scores, 
        "sample_images": wandb.Image(str(save_dir / "generated_samples.png"))
    })

    wandb.finish()

    logger.info(f"FID: {fid_score.item():.4f}")
    logger.info(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")
        
    


if __name__ == '__main__':
    main()