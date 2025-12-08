# Diffusion Transformer Model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
References:
1. Building a Vision Transformer Model From Scratch, Matt Nguyen, https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6 

2. Diffusion Transformers: The New Backbone of Generative Vision, Yashas Donthi, https://yashasdonthi.medium.com/diffusion-transformers-the-new-backbone-of-generative-vision-78eb9df657d5
"""

# -------------
# From Article 1: Step 1: Turning images into tokens
class PatchEmbed(nn.Module):
    """
    Image (pixel) or latent --> to sequence of patch embeddings

    Input: x [B, C, H, W]
    Output: tokens [B, N, d_model] where N = (H/patch) * (W/patch)
    """

    # Define the initialization method
    def __init__(self, in_ch, d_model, img_size, patch_size):
        super().__init__()
        self.in_ch = in_ch # in channels
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size

        # Add a check for image size
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.H_patches = img_size // patch_size
        self.W_patches = img_size // patch_size
        self.num_patches = self.H_patches * self.W_patches

        self.proj = nn.Conv2d(
            in_ch, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)   # [B, d_model, H/P, W/P]
        x = x.flatten(2) # [B, d_model, N]
        x = x.transpose(1, 2)  # [B, N, d_model]
        return x
        
class PatchUnembed(nn.Module):
    """
    Sequence of tokens --> image/latent reconstruction

    Input:  tokens [B, N, d_model]
    Output: x [B, C, H, W]
    """
    def __init__(self, out_ch, d_model, img_size, patch_size):
        super().__init__()
        self.out_ch = out_ch
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.H_patches = img_size // patch_size
        self.W_patches = img_size // patch_size
        self.num_patches = self.H_patches * self.W_patches

        # Linear map from token → flattened patch
        self.proj = nn.Linear(d_model, out_ch * patch_size * patch_size)

    def forward(self, tokens):
        # tokens: [B, N, d_model]
        B, N, D = tokens.shape
        assert N == self.num_patches, "Token count N must equal number of patches"

        patch_dim = self.out_ch * self.patch_size * self.patch_size
        x = self.proj(tokens)                  # [B, N, patch_dim]

        # Reshape N back to [H_patches, W_patches]
        x = x.view(
            B,
            self.H_patches,
            self.W_patches,
            self.out_ch,
            self.patch_size,
            self.patch_size,
        ) # [B, H_p, W_p, C, P, P]

        # Rearrange into [B, C, H, W]
        x = x.permute(0, 3, 1, 4, 2, 5)  # [B, C, H_p, P, W_p, P]
        x = x.contiguous().view(
            B,
            self.out_ch,
            self.H_patches * self.patch_size,
            self.W_patches * self.patch_size,
        ) # [B, C, H, W]
        return x
    
# ------------
# From Article 1: Step 2: Adding Time-step Information
def sinusoidal_embedding(timesteps, dim):
    """
    timesteps: [B] or scalar long
    Returns: [B, dim]
    """
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, dtype=torch.long, device='cpu')

    timesteps = timesteps.float()
    half_dim = dim // 2
    # frequencies
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=timesteps.device).float() / half_dim
    )  # [half_dim]

    # outer product: [B, half_dim]
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # [B, dim]


class TimestepEmbedding(nn.Module):
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        """
        t: scalar int, [B], or [B,1]
        returns: [B, out_dim]
        """
        if t.dim() == 0:
            t = t[None]
        t = t.view(-1)
        emb = sinusoidal_embedding(t, self.mlp[0].in_features)
        return self.mlp(emb)
    
# ------------
# TransformerEncoder in Article 2

class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0):
        """
        Attention Head --> Multi-Head Attention --> Transformer Encoder
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,  # so we use [B, N, D]
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x):
        """
        x: [B, N, d_model]
        """
        # Self-attention with residual
        h = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + h

        # MLP with residual
        h = self.mlp(self.norm2(x))
        x = x + h
        return x

# --------
# FULL DIFFUSION TRANSFORMER BACKBONE:
class DiffusionTransformer(nn.Module):
    """
    Diffusion backbone that mimics UNet's interface:

        eps = model(x_t, t, c=None)

    Inputs:
        x_t: [B, C, H, W] noisy image/latent at timestep t
        t:   scalar or [B] timestep index
        c:   optional class embedding [B, c_dim] (from ClassEmbedder)

    Output:
        eps: [B, C, H, W] predicted noise ε_θ(x_t, t, c)
    """

    def __init__(
        self,
        input_size,# H == W (e.g. 128 or latent size)
        input_ch,   # C (e.g. 3 for RGB, or latent channels)
        T, # total diffusion steps (same as UNet)
        d_model=512,
        depth=8,
        n_heads=8,
        patch_size=16,
        mlp_ratio=4.0,
        conditional=False,
        c_dim=None,
    ):
        super().__init__()
        assert input_size % patch_size == 0, "input_size must be divisible by patch_size"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.input_size = input_size
        self.input_ch = input_ch
        self.T = T
        self.d_model = d_model
        self.conditional = conditional
        self.c_dim = c_dim if conditional else None

        # Patching
        self.patch_embed = PatchEmbed(
            in_ch=input_ch,
            d_model=d_model,
            img_size=input_size,
            patch_size=patch_size,
        )
        self.patch_unembed = PatchUnembed(
            out_ch=input_ch,
            d_model=d_model,
            img_size=input_size,
            patch_size=patch_size,
        )

        # Number of tokens
        self.num_patches = self.patch_embed.num_patches

        # Learnable positional embeddings [1, N, d_model]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Time embedding
        time_dim = d_model
        self.time_embed = TimestepEmbedding(time_dim=time_dim, out_dim=d_model)

        # Optional class embedding projection
        if self.conditional:
            assert c_dim is not None, "c_dim must be provided when conditional=True"
            self.class_proj = nn.Linear(c_dim, d_model)
        else:
            self.class_proj = None

        # Transformer backbone: stack of DiTBlocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final norm (like DiT/ViT)
        self.final_norm = nn.LayerNorm(d_model)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # You can add more sophisticated weight init here if desired.

    def forward(self, x, t, c=None):
        """
        x: [B, C, H, W]
        t: scalar or [B] or [B,1]
        c: [B, c_dim] or None
        """
        B, C, H, W = x.shape
        assert H == self.input_size and W == self.input_size, "Unexpected input spatial size"

        # 1. Patchify
        tokens = self.patch_embed(x) # [B, N, d_model]

        # 2. Add positional encoding
        tokens = tokens + self.pos_embed  # [B, N, d_model]

        # 3. Time (and class) conditioning
        t_emb = self.time_embed(t) # [B, d_model]

        cond = t_emb
        if self.conditional:
            assert c is not None, "Class embedding c must be provided when conditional=True"
            c_emb = self.class_proj(c) # [B, d_model]
            cond = cond + c_emb # combine time + class conditioning

        # broadcast cond to all tokens
        tokens = tokens + cond[:, None, :] # [B, N, d_model]

        # 4. Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)# [B, N, d_model]

        # 5. Final norm
        tokens = self.final_norm(tokens)# [B, N, d_model]

        # 6. Unpatchify to noise map
        eps = self.patch_unembed(tokens) # [B, C, H, W]

        return eps
