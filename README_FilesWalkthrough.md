# IDL25Fall-HW5

# Files Walkthrough

## DDPM: Denoising Diffusion Probabilistic Models & Related Files
```
1. pipelines/ddpm.py
2. schedulers/scheduling_ddpm.py
3. train.py
4. configs/ddpm.yaml
```
### **DDPM Pipeline: Inference and Image Generation**
The **`ddpm.py`** script defines the **DDPMPipeline** class, which is responsible for generating images from random noise using the **DDPM** framework. Key components of this pipeline include:

- **Unet (Noise Prediction Network)**: Neural network used to predict the noise at each timestep during the reverse diffusion process
- **Scheduler**: Defines how the noise evolves through each timestep in both the forward and reverse diffusion processes
- **VAE (Variational Autoencoder)**: Used in **latent DDPM**, allowing the model to operate in a compressed latent space for more efficient inference
- **Class Embedder**: Used for **conditional image generation** via **Classifier-Free Guidance (CFG)**, enabling the generation of class-specific images

The script also contains helper functions like:
- **`numpy_to_pil`**: Converts numpy arrays of generated images to PIL format
- **`progress_bar`**: Displays a progress bar to track the status of image generation during inference

The main **`__call__`** method performs the **inference** process, where random noise is progressively denoised step-by-step to generate high-quality images.

### **DDPM Scheduler: Managing Noise and Timesteps in the Diffusion Process**
The **`DDPMScheduler`** class defines the scheduler for the **Denoising Diffusion Probabilistic Model (DDPM)**, which manages how noise is added and removed during the diffusion process. Key components of this scheduler include:

-**`num_train_timesteps`**: Number of timesteps used during training (the resolution of the diffusion process)
-**beta values**: Control how much noise is added during the forward diffusion process
- **`beta_schedule`**: Determines how the noise schedule evolves (i.e. linearly)
- **`variance_type`**: Defines the variance type (either **"fixed_small"** or **"fixed_large"**) used during noise addition
- **`prediction_type`**: Specifies how the model predicts the noise, typically using **'epsilon'** (the added noise)
- **`clip_sample`**: Determines whether the generated sample should be clipped within a certain range for quality

The DDPMScheduler class also has several key methods for managing the diffusion process:

1. **`set_timesteps`**: Initializes and sets the discrete timesteps for the diffusion chain during inference. Ensures that the number of inference steps doesn't exceed the training timesteps
   
2. **`previous_timestep`**: Computes the previous timestep from the current timestep for reversing the diffusion process (predicting the previous image)

3. **`_get_variance`**: Computes the variance \( \sigma_t \) for a given timestep based on the noise schedule, for the reverse diffusion process, where noise is added back to generate a new sample

4. **`add_noise`**: Adds noise to the original samples using the noise scheduler during the forward diffusion process.

5. **`step`**: Takes the model's output and propagates it backward through the diffusion process (using learned noise predictions) to predict the previous sample during the reverse diffusion process


The DDPMScheduler class is responsible for controlling the noise addition process, managing how noise evolves across timesteps, and computing the necessary variance for the reverse process during inference. It ensures that the model can generate high-quality images by predicting and removing noise across the timesteps of the diffusion process.