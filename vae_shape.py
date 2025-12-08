import torch
from models import VAE
"""
This quick script is used to feed in a dummy image 128x128 and determine the downsampling factor of this VAE and, consequently, its latent space dimension.
"""
def main():
    vae = VAE()
    vae.init_from_ckpt("pretrained/model.ckpt")
    vae.eval()

    # Fake image: [B, 3, 128, 128]
    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        z = vae.encode(x)

    print("Input shape:", x.shape)
    print("Latent shape:", z.shape)

if __name__ == "__main__":
    main()


"""
Sample output from running it on pretrained/model.ckpt on 04Dec2025, 22:18:
 python3 vae_shape.py
making attention of type 'vanilla' with 512 in_channels
Working with z of shape (1, 3, 64, 64) = 12288 dimensions.
making attention of type 'vanilla' with 512 in_channels
Restored from pretrained/model.ckpt
_IncompatibleKeys(missing_keys=[], unexpected_keys=['loss.logvar', 'loss.perceptual_loss.scaling_layer.shift', 'loss.perceptual_loss.scaling_layer.scale', 'loss.perceptual_loss.net.slice1.0.weight', 'loss.perceptual_loss.net.slice1.0.bias', 'loss.perceptual_loss.net.slice1.2.weight', 'loss.perceptual_loss.net.slice1.2.bias', 'loss.perceptual_loss.net.slice2.5.weight', 'loss.perceptual_loss.net.slice2.5.bias', 'loss.perceptual_loss.net.slice2.7.weight', 'loss.perceptual_loss.net.slice2.7.bias', 'loss.perceptual_loss.net.slice3.10.weight', 'loss.perceptual_loss.net.slice3.10.bias', 'loss.perceptual_loss.net.slice3.12.weight', 'loss.perceptual_loss.net.slice3.12.bias', 'loss.perceptual_loss.net.slice3.14.weight', 'loss.perceptual_loss.net.slice3.14.bias', 'loss.perceptual_loss.net.slice4.17.weight', 'loss.perceptual_loss.net.slice4.17.bias', 'loss.perceptual_loss.net.slice4.19.weight', 'loss.perceptual_loss.net.slice4.19.bias', 'loss.perceptual_loss.net.slice4.21.weight', 'loss.perceptual_loss.net.slice4.21.bias', 'loss.perceptual_loss.net.slice5.24.weight', 'loss.perceptual_loss.net.slice5.24.bias', 'loss.perceptual_loss.net.slice5.26.weight', 'loss.perceptual_loss.net.slice5.26.bias', 'loss.perceptual_loss.net.slice5.28.weight', 'loss.perceptual_loss.net.slice5.28.bias', 'loss.perceptual_loss.lin0.model.1.weight', 'loss.perceptual_loss.lin1.model.1.weight', 'loss.perceptual_loss.lin2.model.1.weight', 'loss.perceptual_loss.lin3.model.1.weight', 'loss.perceptual_loss.lin4.model.1.weight', 'loss.discriminator.main.0.weight', 'loss.discriminator.main.0.bias', 'loss.discriminator.main.2.weight', 'loss.discriminator.main.3.weight', 'loss.discriminator.main.3.bias', 'loss.discriminator.main.3.running_mean', 'loss.discriminator.main.3.running_var', 'loss.discriminator.main.3.num_batches_tracked', 'loss.discriminator.main.5.weight', 'loss.discriminator.main.6.weight', 'loss.discriminator.main.6.bias', 'loss.discriminator.main.6.running_mean', 'loss.discriminator.main.6.running_var', 'loss.discriminator.main.6.num_batches_tracked', 'loss.discriminator.main.8.weight', 'loss.discriminator.main.9.weight', 'loss.discriminator.main.9.bias', 'loss.discriminator.main.9.running_mean', 'loss.discriminator.main.9.running_var', 'loss.discriminator.main.9.num_batches_tracked', 'loss.discriminator.main.11.weight', 'loss.discriminator.main.11.bias'])
Input shape: torch.Size([1, 3, 128, 128])
Latent shape: torch.Size([1, 3, 32, 32])
"""