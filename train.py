import os 
import sys
import gdown 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
import torch, os, shutil
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint # Used for save_checkpoint
from torch.utils.data import DataLoader, DistributedSampler



logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # config file
    #set default to configs/ddpm.yaml when its set
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")

    # data 
    parser.add_argument("--data_dir", type=str, default='./data/imagenet100_128x128/train', help="data folder") 
    parser.add_argument("--val_dir", type=str, default=None,
                        help="optional separate validation folder (ImageFolder layout)")
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes in dataset")

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help='mixed precision')
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience") 
    
    # ddpm
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="ddpm inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="ddpm beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="ddpm beta schedule")
    parser.add_argument("--variance_type", type=str, default='fixed_small', help="ddpm variance type")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="ddpm epsilon type")
    parser.add_argument("--clip_sample", type=str2bool, default=True, help="whether to clip sample at each step of reverse process")
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="clip sample range")
    
    # unet
    parser.add_argument("--unet_in_size", type=int, default=128, help="unet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="unet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+', help="unet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+', help="unet attantion stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")
    
    # vae
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm")
    
    # cfg
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="use cfg for conditional (latent) ddpm")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="cfg for inference")
    
    # ddim sampler for inference
    parser.add_argument("--use_ddim", type=str2bool, default=False, help="use ddim sampler for inference")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                    help="DDIM stochasticity (0.0=deterministic)")
    
    # checkpoint path for inference
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path for inference")
    
    # evaluation
    parser.add_argument("--eval_fid_is", type=str2bool, default=False, 
                    help="calculate FID and IS during validation")
    parser.add_argument("--eval_samples", type=int, default=1000, 
                        help="number of samples to generate for FID/IS evaluation")
    parser.add_argument("--eval_frequency", type=int, default=5, 
                        help="evaluate FID/IS every N epochs")

    # first parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args

def calculate_fid_is(pipeline, val_loader, args, device, num_samples=1000):
    """Calculate FID and IS scores"""
    
    print(f"\nCalculating FID and IS with {num_samples} samples...")
    
    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)
    
    # Process real images for FID
    real_images_processed = 0
    for images, _ in val_loader:
        if real_images_processed >= num_samples:
            break
        images = images.to(device)
        
        # Handle latent DDPM case - we need pixel space images
        if images.shape[1] != 3 or images.shape[2] != args.image_size:
            # Skip if we're in latent space, need to use different approach
            continue
            
        images = torch.clamp(images, -1, 1)  # Clamp to [-1, 1]
        images = (images + 1) / 2  # Convert to [0, 1]
        
        batch_size = min(images.shape[0], num_samples - real_images_processed)
        images = images[:batch_size]
        
        images_uint8 = (images * 255).to(torch.uint8)
        fid.update(images_uint8, real=True)
        real_images_processed += batch_size
    
    # Generate fake images
    print(f"Generating {num_samples} samples...")
    all_fake_images = []
    batch_size = 32
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating", disable=not is_primary(args)):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Sample classes if using CFG
            if args.use_cfg:
                classes = torch.randint(0, args.num_classes, (current_batch_size,), device=device)
            else:
                classes = None
            
            # Generate images
            eta = args.ddim_eta if hasattr(args, 'ddim_eta') else 0.0
            guidance_scale = args.cfg_guidance_scale if args.use_cfg else 1.0
            
            batch_images = pipeline(
                batch_size=current_batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                eta=eta,
                guidance_scale=guidance_scale
            )
            
            # Convert PIL images to tensors [0, 1]
            batch_tensors = []
            for img in batch_images:
                img_tensor = transforms.ToTensor()(img)
                batch_tensors.append(img_tensor)
            
            batch_tensors = torch.stack(batch_tensors)
            all_fake_images.append(batch_tensors)
    
    all_fake_images = torch.cat(all_fake_images, dim=0)
    
    # Process fake images for FID and IS
    print("Processing generated images for metrics...")
    batch_size_metric = 50
    for i in range(0, len(all_fake_images), batch_size_metric):
        batch = all_fake_images[i:i+batch_size_metric].to(device)
        batch = torch.clamp(batch, 0, 1)
        
        # FID
        batch_uint8 = (batch * 255).to(torch.uint8)
        fid.update(batch_uint8, real=False)
        
        # IS
        inception_score.update(batch)
    
    # Compute metrics
    print("Computing metrics...")
    fid_score = fid.compute()
    is_mean, is_std = inception_score.compute()
    
    return {
        'fid': fid_score.item(),
        'is_mean': is_mean.item(),
        'is_std': is_std.item()
    }

def download_vae_checkpoint():
    """Download VAE checkpoint if it doesn't exist"""
    ckpt_path = 'pretrained/model.ckpt'
    
    # Create pretrained directory if it doesn't exist
    os.makedirs('pretrained', exist_ok=True)
    
    # Only download if file doesn't exist
    if not os.path.exists(ckpt_path):
        logger.info("Downloading VAE checkpoint from Google Drive...")
        url = "https://drive.google.com/file/d/1SwgvXbliLEp6xfQyYvT1lOQpVXTmxoTO/view?usp=sharing"
        gdown.download(url, ckpt_path, quiet=False, fuzzy=True)
        logger.info(f"VAE checkpoint downloaded to {ckpt_path}")
    else:
        logger.info(f"VAE checkpoint already exists at {ckpt_path}")
    
    
def main():
    
    # parse arguments
    args = parse_args()

    # seed everything
    seed_everything(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup distributed initialize and device
    device = init_distributed_device(args) 
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    
    
    # setup dataset
    logger.info("Creating dataset")
    # TODO: use transform to normalize your images to [-1, 1]
    # TODO: you can also use horizontal flip
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),# <--- FORCE 128x128
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
        transforms.ToTensor(),  # Converts to [0, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1, 1] range
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    # TODO: use image folder for your train dataset
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)

    # TODO: setup dataloader
    sampler = None 
    if args.distributed:
        # TODO: distributed sampler
        sampler = DistributedSampler(train_dataset, 
                                     num_replicas=args.world_size, 
                                     rank=args.rank, 
                                     shuffle=True)
    # TODO: shuffle
    shuffle = False if sampler else True
    # TODO dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Implement val_loader -SK 29Oct2025
    val_dataset, val_loader = None, None
    val_dir = args.val_dir or args.data_dir.replace("/train", "/val")
    if os.path.isdir(val_dir):
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_tf)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if is_primary(args):
            logger.info(f"Validation set found at {val_dir} with {len(val_dataset)} images.")
    else:
        if is_primary(args):
            logger.info("No separate validation directory found; using small no-grad sample from train set for val loss.")

    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size 
    args.total_batch_size = total_batch_size
    
    # setup experiment folder
    if args.run_name is None:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}'
    else:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}-{args.run_name}'
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(args.output_dir, exist_ok=True)
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    noise_scheduler = DDPMScheduler(
        num_inference_steps=args.num_inference_steps, # corrected typo in inference_timesteps. Should just be .."inference_steps". SK 29Oct2025.
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range
    )

    # NOTE: this is for latent DDPM 
    vae = None
    if args.latent_ddpm:
        # Download checkpoint if needed
        download_vae_checkpoint()
        vae = VAE()
        # NOTE: do not change this
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
        
    # Note: this is for cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: 
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_in_ch,
            n_classes=args.num_classes,
        )
        
    # send to device
    # explicitly move the model to the same device as the training tensors- SK 29Oct2025.
    unet = unet.to(device)
    noise_scheduler = noise_scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
    
    # TODO: setup optimizer
    optimizer = torch.optim.AdamW(
        params=unet.parameters() if class_embedder is None else list(unet.parameters()) + list(class_embedder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    ) 
    # TODO: setup scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(train_loader),
        eta_min=1e-6,
    )
    
    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    #  setup distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder
    vae_wo_ddp = vae
    # TODO: setup ddim
    if args.use_ddim:
        scheduler_wo_ddp = DDIMScheduler(
            num_inference_steps=args.num_inference_steps,
            num_train_timesteps=args.num_train_timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range
        )
    else:
        scheduler_wo_ddp = noise_scheduler
    
    # ensure the pipeline’s scheduler is on the same device
    scheduler_wo_ddp = scheduler_wo_ddp.to(device)

    # TODO: setup evaluation pipeline
    # NOTE: this pipeline is not differentiable and only for evaluatin
    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=scheduler_wo_ddp,
        vae=vae,
        class_embedder=class_embedder,
    )
    
    
    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
    
    # start tracker
    if is_primary(args):
        wandb_logger = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "ddpm"),
            entity=os.environ.get("WANDB_ENTITY", "Diffusion-F25DL_Project"),
            name=args.run_name,
            config=vars(args),
        )
    
    # Start training    
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))

    # EARLY-STOPPING SETUP -SK 29Oct2025
    best_val_loss = float('inf') # Initialize
    patience = args.patience # stop after 5 epochs of no improvement
    patience_counter=0

    # training
    for epoch in range(args.num_epochs):
        
        # set epoch for distributed sampler, this is for distribution training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        
        loss_m = AverageMeter()
        
        # TODO: set unet and scheduelr to train
        unet = unet.train()
        #scheduler #not sure if we need this?
        
        
        # TODO: finish this
        for step, (images, labels) in enumerate(train_loader):

            # TODO: send to device
            images = images.to(device)
            labels = labels.to(device)
            
            
            # NOTE: this is for latent DDPM 
            if vae is not None:
                # use vae to encode images as latents
                images = vae.encode(images) 
                # NOTE: do not change  this line, this is to ensure the latent has unit std
                images = images * 0.1845
            
            # TODO: zero grad optimizer
            optimizer.zero_grad() #clear previous gradients
            
            
            # NOTE: this is for CFG
            if class_embedder is not None:
                # TODO: use class embedder to get class embeddings
                class_emb = class_embedder(labels)
            else:
                # NOTE: if not cfg, set class_emb to None
                class_emb = None
            
            # TODO: sample noise 
            noise = torch.randn_like(images)  
            
            # TODO: sample timestep t
            # timesteps = torch.randint(0, args.num_train_timesteps, (args.batch_size,)).long()
            """
            Change from timesteps = torch.randint(0, args.num_train_timesteps, (args.batch_size,)).long() to the structure below in case the lasst batch in dataset might have fewer samples and may cause a RuntimeEerror with tensor size mismatch.

            Instead, replace timesteps as below to match the timesteps to the batch size dynamically.
            """
            B = images.size(0)  # actual batch size (may be smaller than args.batch_size)
            timesteps = torch.randint(
                low=0,                          # lower bound (inclusive)
                high=args.num_train_timesteps,  # upper bound (exclusive)
                size=(B,),                      # create one timestep per image
                device=device,                  # allocate directly on GPU
                dtype=torch.long                # ensure it's long integer type
            )
            # TODO: add noise to images using noise_scheduler
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # TODO: model prediction
            model_pred = unet.forward(noisy_images, timesteps, c=class_emb) 
            
            if args.prediction_type == 'epsilon':
                target = noise 
            
            # TODO: calculate loss
            loss = F.mse_loss(model_pred, target) 
            
            # record loss
            loss_m.update(loss.item())
            
            # backward and step 
            loss.backward()
            # TODO: grad clip
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            
            # TODO: step your optimizer
            optimizer.step()
            lr_scheduler.step() # Step the lr scheduler. SK 29Oct2025.
            
            progress_bar.update(1)
            
            # logger
            if step % 100 == 0 and is_primary(args):
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, Loss {loss.item()} ({loss_m.avg})")
                wandb_logger.log({'loss': loss_m.avg})

        # validation
        # send unet to evaluation mode
        unet.eval()    

        # EARLY_STOPPING: Validation loss calculation -SK 29Oct2025
        """
        Logic:
        1. Diffusion loss is an expectation over data, random noise, and random noise t: Ex,ϵ,t​[∥ϵ−ϵθ​(xt​,t)∥2]
        2. So averaging the validation loss over a few mini-batches gives a realiable estimate with low variance of this expectation
        """
        val_loss_total = 0.0
        val_steps = 0
        loader_for_val = val_loader if val_loader is not None else train_loader
        MAX_VAL_STEPS = None if val_loader is not None else 10

        with torch.no_grad(): # No tracking of gradients during validation
            for i, (images, labels) in enumerate(loader_for_val):
                if MAX_VAL_STEPS is not None and i >= MAX_VAL_STEPS:
                    break
                images = images.to(device)
                # Mimic the process of corrupting with noise
                noise = torch.randn_like(images)
                B = images.size(0)
                timesteps = torch.randint(
                    low=0, high=args.num_train_timesteps, size=(B,), device=device, dtype=torch.long
                )
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps) # Add noise with a scheduler.
                model_pred = unet(noisy_images, timesteps, c=None) # Predict ϵ=ϵ_θ​(xt​,t)
                val_loss_total += F.mse_loss(model_pred, noise).item() # COmpute the MSE loss 
                val_steps += 1
        val_loss = val_loss_total / max(1, val_steps) # Average across all batches
        # EARLY_STOPPING CHECK:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save best model
            if is_primary(args):
                torch.save({
                    "epoch": epoch + 1,
                    "val_loss": float(val_loss),
                    "args": vars(args),
                    "unet": unet_wo_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    # Optional if you want to resume CFG:
                    # "class_embedder": (class_embedder_wo_ddp.state_dict() if class_embedder_wo_ddp else None),
                }, os.path.join(save_dir, "best.pt"))
                logger.info(f"Saved BEST checkpoint: {os.path.join(save_dir, 'best.pt')}  (val_loss={val_loss:.6f})")

        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break
                
        # Log the metrics in wandb
        if is_primary(args):
            wandb_logger.log({
                'epoch': epoch + 1,
                'train_loss_epoch_avg': loss_m.avg,   # epoch-avg of train loss
                'val_loss': val_loss,
                'lr': lr_scheduler.get_last_lr()[0],  # Log learning rate
                })
            wandb.run.summary['best_val_loss'] = best_val_loss
            wandb.run.summary['best_epoch'] = epoch + 1 if val_loss <= best_val_loss else wandb.run.summary.get('best_epoch', epoch + 1)

        # -----------   END EARLY STOPPING LOGIC

        # =====================================================
        # FID/IS EVALUATION (if enabled)
        # =====================================================
        if args.eval_fid_is and (epoch + 1) % args.eval_frequency == 0:
            if is_primary(args):
                logger.info(f"\n{'='*60}")
                logger.info("EVALUATING FID AND IS")
                logger.info(f"{'='*60}")
                
                try:
                    metrics = calculate_fid_is(
                        pipeline=pipeline,
                        val_loader=val_loader if val_loader else train_loader,
                        args=args,
                        device=device,
                        num_samples=args.eval_samples
                    )
                    
                    logger.info(f"\nEpoch {epoch+1} Metrics:")
                    logger.info(f"  FID:  {metrics['fid']:.4f}")
                    logger.info(f"  IS:   {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}")
                    logger.info(f"{'='*60}\n")
                    
                    # Log to wandb
                    wandb_logger.log({
                        'fid': metrics['fid'],
                        'is_mean': metrics['is_mean'],
                        'is_std': metrics['is_std'],
                        'epoch': epoch + 1
                    })
                    
                    # Save metrics to file
                    metrics_file = os.path.join(output_dir, 'metrics_history.txt')
                    with open(metrics_file, 'a') as f:
                        f.write(f"Epoch {epoch+1}: FID={metrics['fid']:.4f}, IS={metrics['is_mean']:.4f}±{metrics['is_std']:.4f}\n")
                        
                except Exception as e:
                    logger.error(f"Error calculating FID/IS: {e}")
        
        # Continue with your existing code for generating sample images
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)
        #######################
        
        #  
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)
        
        # NOTE: this is for CFG
        if args.use_cfg:
            # random sample 4 classes
            classes = torch.randint(0, args.num_classes, (4,), device=device)
            # TODO: fill pipeline
            gen_images = pipeline(batch_size=4, num_inference_steps=args.num_inference_steps, classes=classes, eta=args.ddim_eta) # Added ddim_eta from arguments. SK 29Oct2025
        else:
            # TODO: fill pipeline
            gen_images = pipeline(batch_size=4, num_inference_steps=args.num_inference_steps, classes=None, eta=args.ddim_eta) # Added ddim_eta from arguments. SK 29Oct2025

        # create a blank canvas for the grid
        grid_image = Image.new('RGB', (4 * args.image_size, 1 * args.image_size))
        # paste images into the grid
        for i, image in enumerate(gen_images):
            x = (i % 4) * args.image_size
            y = 0
            grid_image.paste(image, (x, y))
        
        # Send to wandb
        if is_primary(args):
            wandb_logger.log({'gen_images': wandb.Image(grid_image)})
            
        # save checkpoint - Updated to use torch.save()- SK 29Oct2025
        if is_primary(args):
            torch.save({
                "epoch": epoch + 1,
                "val_loss": float(val_loss),
                "args": vars(args),
                "unet": unet_wo_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                # Optional:
                # "class_embedder": (class_embedder_wo_ddp.state_dict() if class_embedder_wo_ddp else None),
            }, os.path.join(save_dir, "last.pt"))
            logger.info(f"Saved LAST checkpoint: {os.path.join(save_dir, 'last.pt')}")


if __name__ == '__main__':
    main()
