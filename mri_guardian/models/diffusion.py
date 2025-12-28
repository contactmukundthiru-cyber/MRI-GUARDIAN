"""
Diffusion / Score-Based Models for MRI Reconstruction

Diffusion models have revolutionized image generation.
Here we adapt them for MRI reconstruction.

INTUITION (NO HEAVY MATH):
==========================
Imagine you have a clear MRI image. Now add noise to it.
Add more noise. More. Eventually it's just random static.

Diffusion models learn to REVERSE this process:
Random noise → Slightly less noisy → ... → Clean image

For MRI reconstruction:
1. Start with aliased/noisy image (from undersampling)
2. Learn to "denoise" it toward a clean image
3. Combine with data consistency (measured k-space)

WHY DIFFUSION FOR MRI?
=====================
1. Generative prior: Learns what real MRI looks like
2. Uncertainty: Multiple samples show reconstruction uncertainty
3. Quality: Often better than single-pass networks
4. Flexibility: Works with any measurement model

SIMPLIFIED FRAMEWORK:
====================
Forward process: x_0 → x_1 → x_2 → ... → x_T (add noise)
Reverse process: x_T → x_{T-1} → ... → x_0 (remove noise)

The network learns to predict the noise at each step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

from .unet import UNet, ResConvBlock, AttentionBlock


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion models.

    The network needs to know "how noisy is the input?"
    We encode the noise level (time step) as a sinusoidal embedding,
    similar to positional encoding in Transformers.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create time embedding.

        Args:
            t: Time step tensor (B,) in range [0, 1]

        Returns:
            Embedding (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)

        return embeddings


class TimeConditionedBlock(nn.Module):
    """
    Convolutional block conditioned on time step.

    The time embedding modulates the features, telling the network
    what noise level to expect.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2)  # Scale and shift
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with time conditioning.

        Args:
            x: Features (B, C, H, W)
            time_emb: Time embedding (B, time_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # Time conditioning: FiLM (Feature-wise Linear Modulation)
        time_out = self.time_mlp(time_emb)
        scale, shift = time_out.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + scale) + shift

        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class ScoreNetwork(nn.Module):
    """
    Score Network for MRI Reconstruction.

    Predicts the "score" = gradient of log probability = direction to clean image.
    Architecture: UNet with time conditioning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        use_attention: bool = True,
        attention_resolutions: Tuple[int, ...] = (16, 8)
    ):
        """
        Args:
            in_channels: Input channels (1 for magnitude)
            base_channels: Base feature channels
            channel_mults: Channel multipliers for each level
            num_res_blocks: Residual blocks per level
            time_dim: Dimension of time embedding
            use_attention: Use self-attention
            attention_resolutions: Resolutions where to use attention
        """
        super().__init__()

        self.time_dim = time_dim
        self.channel_mults = channel_mults
        num_levels = len(channel_mults)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = base_channels
        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    TimeConditionedBlock(channels, out_channels, time_dim)
                )
                channels = out_channels

            if level < num_levels - 1:
                self.down_samples.append(nn.Conv2d(channels, channels, 3, stride=2, padding=1))

        # Middle
        self.mid_block1 = TimeConditionedBlock(channels, channels, time_dim)
        if use_attention:
            self.mid_attention = AttentionBlock(channels)
        else:
            self.mid_attention = nn.Identity()
        self.mid_block2 = TimeConditionedBlock(channels, channels, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    TimeConditionedBlock(channels + out_channels, out_channels, time_dim)
                )
                channels = out_channels

            if level > 0:
                self.up_samples.append(nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1))

        # Final
        self.final_norm = nn.GroupNorm(8, channels)
        self.final_conv = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict score (noise direction).

        Args:
            x: Noisy image (B, C, H, W)
            t: Time step (B,) in [0, 1]
            cond: Optional conditioning (e.g., measurement)

        Returns:
            Predicted score/noise (B, C, H, W)
        """
        # Time embedding
        time_emb = self.time_embed(t)

        # Initial conv
        h = self.init_conv(x)
        if cond is not None:
            h = h + cond

        # Encoder with skip connections
        skips = []
        block_idx = 0
        for level in range(len(self.channel_mults)):
            for _ in range(2):  # num_res_blocks
                h = self.down_blocks[block_idx](h, time_emb)
                skips.append(h)
                block_idx += 1

            if level < len(self.channel_mults) - 1:
                h = self.down_samples[level](h)

        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_emb)

        # Decoder
        block_idx = 0
        for level in reversed(range(len(self.channel_mults))):
            for _ in range(3):  # num_res_blocks + 1
                skip = skips.pop() if skips else torch.zeros_like(h)
                if h.shape != skip.shape:
                    h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=True)
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[block_idx](h, time_emb)
                block_idx += 1

            if level > 0:
                h = self.up_samples[len(self.channel_mults) - 1 - level](h)

        # Final
        h = self.final_norm(h)
        h = F.silu(h)
        return self.final_conv(h)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process.

    Handles:
    - Forward process (adding noise)
    - Training objective (predicting noise)
    - Sampling (iterative denoising)
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        """
        Args:
            model: Score/noise prediction network
            num_timesteps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            beta_schedule: "linear" or "cosine"
        """
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

        # Noise schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        # Precompute diffusion constants
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1 / alphas))
        self.register_buffer("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: Add noise to clean image.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0: Clean images (B, C, H, W)
            t: Time steps (B,)
            noise: Optional pre-generated noise

        Returns:
            x_t: Noisy images
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        return x_t, noise

    def p_losses(
        self,
        x_0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training loss: Predict the noise.

        Args:
            x_0: Clean images
            t: Time steps (if None, randomly sampled)
            cond: Optional conditioning

        Returns:
            Loss value
        """
        B = x_0.shape[0]

        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)

        # Add noise
        x_t, noise = self.q_sample(x_0, t)

        # Predict noise
        t_normalized = t.float() / self.num_timesteps
        noise_pred = self.model(x_t, t_normalized, cond)

        # L2 loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single denoising step.

        Args:
            x_t: Noisy image at time t
            t: Current time step
            cond: Optional conditioning

        Returns:
            x_{t-1}: Less noisy image
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        t_normalized = t_tensor.float() / self.num_timesteps

        # Predict noise
        noise_pred = self.model(x_t, t_normalized, cond)

        # Compute x_{t-1}
        beta = self.betas[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]

        mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha * noise_pred)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            x_prev = mean + sigma * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        device: str = "cuda",
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples via iterative denoising.

        Args:
            shape: Output shape (B, C, H, W)
            cond: Optional conditioning
            device: Device to use
            show_progress: Show progress bar

        Returns:
            Generated images
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)

        iterator = range(self.num_timesteps - 1, -1, -1)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling")

        for t in iterator:
            x = self.p_sample(x, t, cond)

        return x


class DiffusionSampler(nn.Module):
    """
    Diffusion-based MRI Reconstruction Sampler.

    Combines diffusion denoising with data consistency
    for MRI reconstruction.
    """

    def __init__(
        self,
        score_model: nn.Module,
        num_steps: int = 50,  # Fewer steps for reconstruction
        step_size: float = 0.1
    ):
        """
        Args:
            score_model: Trained score network
            num_steps: Number of denoising steps
            step_size: Step size for Langevin dynamics
        """
        super().__init__()
        self.score_model = score_model
        self.num_steps = num_steps
        self.step_size = step_size

    @torch.no_grad()
    def sample_with_dc(
        self,
        initial: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        dc_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Reconstruction with diffusion + data consistency.

        Uses annealed Langevin dynamics with DC projection.

        Args:
            initial: Initial image estimate (B, 1, H, W)
            masked_kspace: Measured k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)
            dc_weight: Weight for data consistency step

        Returns:
            Reconstructed image
        """
        from ..data.kspace_ops import fft2c, ifft2c, complex_to_channels, channels_to_complex

        x = initial.clone()

        # Noise levels (sigma) annealing schedule
        sigmas = torch.linspace(1.0, 0.01, self.num_steps)

        for i, sigma in enumerate(sigmas):
            # Score prediction
            t = torch.full((x.shape[0],), 1 - i / self.num_steps, device=x.device)
            score = self.score_model(x, t)

            # Langevin step
            noise = torch.randn_like(x) if i < self.num_steps - 1 else 0
            x = x + self.step_size * score + np.sqrt(2 * self.step_size) * sigma * noise

            # Data consistency projection
            # This ensures we stay consistent with measurements
            x_complex = x.squeeze(1) * torch.exp(1j * torch.zeros_like(x.squeeze(1)))
            kspace_pred = fft2c(x_complex)
            kspace_meas = channels_to_complex(masked_kspace)

            if mask.ndim == 4:
                mask_2d = mask.squeeze(1)
            else:
                mask_2d = mask

            # Blend predicted and measured k-space at measured locations
            kspace_dc = (1 - dc_weight) * kspace_pred + dc_weight * mask_2d * kspace_meas + (1 - mask_2d) * kspace_pred

            x_dc = ifft2c(kspace_dc)
            x = torch.abs(x_dc).unsqueeze(1)

        return x
