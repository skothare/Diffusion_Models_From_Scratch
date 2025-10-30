from typing import List, Optional, Tuple, Union

import torch 
import torch.nn as nn 
import numpy as np

from utils import randn_tensor

from .scheduling_ddpm import DDPMScheduler


class DDIMScheduler(DDPMScheduler): # Inherits from DDPMScheduler all of its fields
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Calling the base constructor
        assert self.num_inference_steps is not None, "Please set `num_inference_steps` before running inference using DDIM."
        self.set_timesteps(self.num_inference_steps)

    
    def _get_variance(self, t):
        """
        This is one of the most important functions in the DDIM. It calculates the variance $sigma_t$ for a given timestep.
        
        Args:
            t (int): The current timestep.
        
        Return:
            variance (torch.Tensor): The variance $sigma_t$ for the given timestep.

        Note about timesteps:
        1.  During inference, the T training steps list will be subsampled to a shorter timsteps list in descending order which may or may not be contiguous: [1000, 997, 972, 949, ..., 0]
        2. Hence, prev_t is equivalent to stepping into the next lower subsampled timestep.

        Variance formula (variance divided by eta, η; when η>0, we revert to stochastic DDPM with η=1 matching magnitude of DDPM's noise and when 0 we revert to deterministic DDIM) implemented from equation 16 of the paper:
        Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. arXiv. https://doi.org/10.48550/arXiv.2010.02502 
        """
        
        t = int(t)
        #  calculate $beta_t$ for the current timestep using the cumulative product of alphas

        # Previous time_step to be stepped into next
        if isinstance(self.timesteps, torch.Tensor): # Safeguard in case the self.timesteps is a tensor instead of a python list
            # Assuming self.timesteps is a tensor, find where t appears in the tensor
            # Use .numel() where it returns the number of elements in a tensor
            pos = (self.timesteps == t).nonzero(as_tuple=False)
            assert pos.numel() > 0, f"timestep {t} not found in self.timesteps" # In case t is not to be found
            i = int(pos[0].item()) # Cast the position to an integer index
            if (i+1) < self.timesteps.numel(): # Index within range of the tensor
                prev_t = int(self.timesteps[i + 1].item())
            else: # Index out of range
                prev_t = 0
        else:
            i = self.timesteps.index(t) # Find the index of this t 
            prev_t = (self.timesteps[i + 1]) if (i + 1) < len(self.timesteps) else 0

        # Cumulative product of alphas (alpha_t) at step t
        alpha_prod_t = self.alphas_cumprod[t]
        # Safeguard: clamp a small eps to the alpha
        """ 
        Apparently, creating a tensor of the small epsilon to add on a CPU requires a slow transfer to a GPU in case the alpha_prod_t tensor lives on the GPU. To avoid warnings, using this format with specification of the device to be the same as the alpha_prod_t.device.

        For later --> why does this CPU to GPU happen if workflow is being processed on a GPU?
            - The numeric literals like 1e-12 are CPU floats by default.
        """
        eps = torch.as_tensor(1e-12, dtype=alpha_prod_t.dtype, device=alpha_prod_t.device)

        # Subsampled a_t-1
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        
        # Clamp the epsilon to the alpha_prod_t and alpha_prod_t_prev tensors
        alpha_prod_t, alpha_prod_t_prev = torch.clamp(alpha_prod_t, eps, 1.0), torch.clamp(alpha_prod_t_prev, eps, 1.0) # Here, we force each alpha to be between 1.0 >= alpha >= 1e-12

        # The remainder probabilities (1-alphas)
        beta_prod_t = torch.clamp(1.0 - alpha_prod_t, eps, 1.0)
        beta_prod_t_prev = torch.clamp(1.0 - alpha_prod_t_prev, eps, 1.0)
        
        #  DDIM equation for variance (Square Eq16 from the paper and remove the eta)
        variance = ((beta_prod_t_prev)/beta_prod_t) * (1.0 - (alpha_prod_t / alpha_prod_t_prev))
        variance = torch.clamp(variance, min=0.0)  # Guard against low variance jitters into the negative territory; set the floor to 0.0
        return variance
    
    
    def step(
        self,
        model_output: torch.Tensor, # UNet's predicted noise, e_theta(x_t, t)
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        eta: float=0.0,
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (torch.Tensor):
                The direct output from learned diffusion model.
            timestep (float'):
                The current discrete timestep in the diffusion chain.
            sample (torch.Tensor):
                A current instance of a sample created by the diffusion process.
            eta (float):
                The weight of the noise to add to the variance.
            generator (torch.Generator, *optional*):
                A random number generator.

        Returns:
            pred_prev_sample (torch.Tensor):
                The predicted previous sample.

        
        """

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"
        
        t = int(timestep) 

        # Similar to _get_variance() function above
        if isinstance(self.timesteps, torch.Tensor): # Safeguard in case the self.timesteps is a tensor instead of a python list
            # Assuming self.timesteps is a tensor, find where t appears in the tensor
            # Use .numel() where it returns the number of elements in a tensor
            pos = (self.timesteps == t).nonzero(as_tuple=False)
            assert pos.numel() > 0, f"timestep {t} not found in self.timesteps" # In case t is not to be found
            i = int(pos[0].item()) # Cast the position to an integer index
            if (i+1) < self.timesteps.numel(): # Index within range of the tensor
                prev_t = int(self.timesteps[i + 1].item())
            else: # Index out of range
                prev_t = 0
        else:
            i = self.timesteps.index(t) # Find the index of this t 
            prev_t = (self.timesteps[i + 1]) if (i + 1) < len(self.timesteps) else 0
        
        # TODO: 1. compute alphas, betas
        alpha_prod_t       = self.alphas_cumprod[t]
        alpha_prod_t_prev  = self.alphas_cumprod[prev_t]
        eps = torch.as_tensor(1e-12, dtype=sample.dtype, device=sample.device)
        alpha_prod_t = torch.clamp(alpha_prod_t, eps, 1.0)
        alpha_prod_t_prev = torch.clamp(alpha_prod_t_prev, eps, 1.0)
        beta_prod_t = torch.clamp(1.0 - alpha_prod_t, eps, 1.0)
        
        # TODO: 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == 'epsilon':
            pred_epsilon = model_output
            pred_original_sample = (sample - torch.sqrt(1.0 - alpha_prod_t) * pred_epsilon) / torch.sqrt(alpha_prod_t) # This is part of the first term in equation 12
            
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # TODO: 3. Clip or threshold "predicted x_0" (for better sampling quality)
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # TODO: 4. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(t)
        std_dev_t = eta * torch.sqrt(variance)

        # TODO: 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = torch.sqrt(torch.clamp(1.0 - alpha_prod_t_prev - std_dev_t**2, min=0.0)) * pred_epsilon # This is the second term in equation 12

        # TODO: 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        # TODO: 7. Add noise with eta
        if eta > 0:
            variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            # variance = None # Implemented above

            prev_sample = prev_sample + std_dev_t * variance_noise
        
        return prev_sample