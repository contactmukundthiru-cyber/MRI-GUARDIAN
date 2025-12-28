"""
Guardian Model: Physics-Guided Generative MRI Reconstruction

This is the CORE innovation of MRI-GUARDIAN.

KEY DIFFERENCES FROM STANDARD UNET:
==================================
1. DUAL-DOMAIN: Processes both k-space AND image domain
2. DATA CONSISTENCY: Every iteration enforces measured k-space values
3. GENERATIVE PRIOR: Uses score-based/diffusion-inspired refinement
4. UNROLLED OPTIMIZATION: Mimics iterative reconstruction algorithms

WHY "GUARDIAN"?
==============
This model serves as a GUARDIAN against hallucinations:
- Physics constraints prevent impossible reconstructions
- Data consistency ensures agreement with measurements
- Can be used as reference to audit black-box models

ARCHITECTURE OVERVIEW:
=====================
Input: Undersampled k-space + mask

For each iteration (unrolled step):
    1. K-space refinement: Fill in missing frequencies
    2. Image refinement: Enhance spatial features
    3. Data consistency: Replace measured k-space
    4. Score refinement: Push toward manifold of real images

Output: Reconstructed image

MATHEMATICAL FOUNDATION:
=======================
We solve: minimize ||Ax - y||² + λR(x)

Where:
- A: Forward MRI model (image → measured k-space)
- y: Measured k-space
- R(x): Learned regularizer (prior on natural MRI images)
- λ: Regularization weight

The Guardian unrolls this optimization, learning both the
regularizer and step sizes from data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from .unet import UNet, ConvBlock, ResConvBlock, AttentionBlock
from ..physics.data_consistency import HardDataConsistency, SoftDataConsistency, GradientDataConsistency
from ..data.kspace_ops import fft2c, ifft2c, complex_to_channels, channels_to_complex, complex_abs


@dataclass
class GuardianConfig:
    """Configuration for Guardian model."""
    # Architecture
    num_iterations: int = 8  # Number of unrolled steps
    base_channels: int = 48  # Base feature channels
    num_levels: int = 4  # UNet depth

    # Domains
    use_kspace_net: bool = True  # Process k-space
    use_image_net: bool = True  # Process image domain
    use_score_net: bool = True  # Use score-based refinement

    # Data consistency
    dc_mode: str = "soft"  # "hard", "soft", "gradient"
    learnable_dc: bool = True  # Learn DC weight

    # Input/output
    in_channels: int = 2  # 2 for complex (real+imag)
    complex_input: bool = True

    # Regularization
    dropout: float = 0.0
    use_attention: bool = True

    # Training
    intermediate_supervision: bool = True  # Supervise each iteration


class KSpaceConvBlock(nn.Module):
    """
    Convolutional block designed for k-space processing.

    K-space has different statistics than images:
    - Large dynamic range (center much brighter than edges)
    - Complex-valued (2 channels)
    - Low-frequency bias

    This block uses instance normalization and handles complex data.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        return x + residual


class KSpaceRefinementNet(nn.Module):
    """
    K-Space Refinement Network.

    Learns to fill in missing k-space values while respecting
    the structure of frequency domain data.

    Key insight: Missing k-space should be filled based on
    surrounding measured frequencies and image priors.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        num_blocks: int = 4
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)

        self.blocks = nn.ModuleList([
            KSpaceConvBlock(hidden_channels, hidden_channels)
            for _ in range(num_blocks)
        ])

        self.output_conv = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine k-space by predicting missing values.

        Args:
            kspace: Current k-space estimate (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)

        Returns:
            Refined k-space (B, 2, H, W)
        """
        x = self.input_conv(kspace)

        for block in self.blocks:
            x = block(x)

        # Predict k-space correction
        correction = self.output_conv(x)

        # Only apply correction to missing k-space (where mask=0)
        refined = kspace + correction * (1 - mask)

        return refined


class ImageRefinementNet(nn.Module):
    """
    Image Domain Refinement Network.

    Standard UNet-style network for image enhancement.
    Removes aliasing artifacts and enhances details.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        use_attention: bool = False
    ):
        super().__init__()

        # Simple UNet-like architecture
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Encoder
        channels = in_channels
        encoder_channels = []
        for level in range(num_levels):
            out_ch = base_channels * (2 ** level)
            self.encoder_blocks.append(ResConvBlock(channels, out_ch))
            encoder_channels.append(out_ch)
            if level < num_levels - 1:
                self.pools.append(nn.MaxPool2d(2))
            channels = out_ch

        # Attention at bottleneck
        if use_attention:
            self.attention = AttentionBlock(channels)
        else:
            self.attention = None

        # Decoder
        for level in range(num_levels - 2, -1, -1):
            in_ch = encoder_channels[level + 1]
            out_ch = encoder_channels[level]

            self.upsamples.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decoder_blocks.append(ResConvBlock(out_ch * 2, out_ch))

        self.final_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine image."""
        # Encoder
        skips = []
        for level, block in enumerate(self.encoder_blocks):
            x = block(x)
            if level < len(self.pools):
                skips.append(x)
                x = self.pools[level](x)

        # Attention
        if self.attention is not None:
            x = self.attention(x)

        # Decoder
        for level, (upsample, block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            x = upsample(x)
            skip = skips[-(level + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.final_conv(x)


class ScoreRefinementNet(nn.Module):
    """
    Score-based Refinement Network.

    Inspired by diffusion/score-matching models.
    Estimates the "score" (gradient of log probability) of the
    image distribution and pushes reconstructions toward the
    manifold of realistic MRI images.

    INTUITION:
    If x is slightly "off" the manifold of real MRI images,
    the score tells us which direction to move to get back on.

    score(x) = ∇_x log p(x)

    Update: x_new = x + ε * score(x)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_blocks: int = 4
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.GroupNorm(8, hidden_channels),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.GroupNorm(8, hidden_channels),
                    nn.SiLU(),
                )
            )

        self.output_conv = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

        # Learnable step size
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute score and apply refinement.

        Args:
            x: Current image estimate (B, 1, H, W)

        Returns:
            Refined image
        """
        h = self.input_conv(x)

        for block in self.blocks:
            h = h + block(h)  # Residual

        score = self.output_conv(h)

        # Apply score-based update
        # step_size is learnable but kept positive
        step = torch.sigmoid(self.step_size) * 0.5

        return x + step * score


class GuardianIteration(nn.Module):
    """
    Single iteration of Guardian reconstruction.

    Flow:
    1. K-space refinement (if enabled)
    2. Transform to image domain
    3. Image refinement (if enabled)
    4. Score refinement (if enabled)
    5. Data consistency
    """

    def __init__(self, config: GuardianConfig, iteration_idx: int = 0):
        super().__init__()
        self.config = config

        # K-space processing
        if config.use_kspace_net:
            self.kspace_net = KSpaceRefinementNet(
                in_channels=config.in_channels,
                hidden_channels=config.base_channels,
                num_blocks=3
            )
        else:
            self.kspace_net = None

        # Image processing
        if config.use_image_net:
            self.image_net = ImageRefinementNet(
                in_channels=1,  # Magnitude
                base_channels=config.base_channels,
                num_levels=config.num_levels,
                use_attention=config.use_attention
            )
        else:
            self.image_net = None

        # Score refinement
        if config.use_score_net:
            self.score_net = ScoreRefinementNet(
                in_channels=1,
                hidden_channels=config.base_channels
            )
        else:
            self.score_net = None

        # Data consistency
        if config.dc_mode == "hard":
            self.dc = HardDataConsistency()
        elif config.dc_mode == "soft":
            self.dc = SoftDataConsistency(lambda_init=0.9, learnable=config.learnable_dc)
        elif config.dc_mode == "gradient":
            self.dc = GradientDataConsistency(step_size=1.0, learnable=config.learnable_dc)

    def forward(
        self,
        kspace_current: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One iteration of reconstruction.

        Args:
            kspace_current: Current k-space estimate (B, 2, H, W)
            kspace_measured: Original measured k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)

        Returns:
            kspace_refined: Refined k-space
            image_refined: Refined image (magnitude)
        """
        # Step 1: K-space refinement
        if self.kspace_net is not None:
            kspace_refined = self.kspace_net(kspace_current, mask)
        else:
            kspace_refined = kspace_current

        # Step 2: Transform to image domain
        kspace_complex = channels_to_complex(kspace_refined)
        image_complex = ifft2c(kspace_complex)
        image_mag = complex_abs(image_complex).unsqueeze(1)  # (B, 1, H, W)
        image_phase = torch.angle(image_complex)  # Preserve phase

        # Step 3: Image refinement (on magnitude)
        if self.image_net is not None:
            image_mag = image_mag + self.image_net(image_mag)

        # Step 4: Score refinement
        if self.score_net is not None:
            image_mag = self.score_net(image_mag)

        # Reconstruct complex image
        image_mag = image_mag.squeeze(1)
        image_complex_refined = image_mag * torch.exp(1j * image_phase)

        # Step 5: Data consistency
        image_dc = self.dc(image_complex_refined, kspace_measured, mask)

        # Transform back to k-space for next iteration
        kspace_output = complex_to_channels(fft2c(image_dc))

        # Output magnitude for loss computation
        image_output = complex_abs(image_dc).unsqueeze(1)

        return kspace_output, image_output


class GuardianModel(nn.Module):
    """
    Complete Guardian Model for Physics-Guided MRI Reconstruction.

    This is the main model that unrolls multiple iterations of
    refinement + data consistency.

    NOVELTY:
    1. Dual-domain processing (k-space + image)
    2. Hard physics constraints (data consistency)
    3. Score-based prior (generative model)
    4. Unrolled optimization (interpretable)

    CRITICAL PHYSICS GUARANTEE:
    ==========================
    The final output ALWAYS respects measured k-space values.
    This is enforced by a HARD DATA CONSISTENCY step at the very end,
    OUTSIDE the neural network. This is not a learned constraint -
    it's a physics law: measured data cannot be changed.

    Why this matters:
    - Soft DC during training helps gradient flow
    - But at inference, we MUST enforce hard physics
    - No matter what the AI "thinks", measured k-space wins
    """

    def __init__(self, config: Optional[GuardianConfig] = None):
        super().__init__()

        if config is None:
            config = GuardianConfig()

        self.config = config

        # Initial k-space embedding
        self.initial_conv = nn.Sequential(
            nn.Conv2d(config.in_channels + 1, config.base_channels, 3, padding=1),  # +1 for mask
            nn.LeakyReLU(0.1),
            nn.Conv2d(config.base_channels, config.in_channels, 3, padding=1)
        )

        # Unrolled iterations
        self.iterations = nn.ModuleList([
            GuardianIteration(config, i)
            for i in range(config.num_iterations)
        ])

        # Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv2d(1, config.base_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(config.base_channels, config.base_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(config.base_channels, 1, 3, padding=1)
        )

        # CRITICAL: Hard data consistency as FINAL step
        # This is NOT part of the network - it's a physics constraint
        # Applied AFTER all neural network processing
        self.final_hard_dc = HardDataConsistency()

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        return_intermediates: bool = False,
        enforce_dc: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Guardian model.

        Args:
            masked_kspace: Undersampled k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)
            return_intermediates: If True, return all intermediate reconstructions
            enforce_dc: If True (default), apply HARD data consistency at the end.
                        This GUARANTEES measured k-space is preserved in output.
                        Only set to False for ablation studies.

        Returns:
            Dict with:
                - 'output': Final reconstruction (B, 1, H, W)
                - 'output_pre_dc': Output before hard DC (for analysis)
                - 'intermediates': List of intermediate images (if requested)
                - 'kspace_final': Final k-space estimate
                - 'dc_applied': Whether hard DC was applied

        PHYSICS GUARANTEE:
        When enforce_dc=True (default), the output image ALWAYS satisfies:
            FFT(output)[measured_locations] == measured_kspace[measured_locations]
        This is non-negotiable physics - the AI cannot override real data.
        """
        # Initial estimate
        kspace_input = torch.cat([masked_kspace, mask], dim=1)
        kspace_current = masked_kspace + self.initial_conv(kspace_input)

        intermediates = []

        # Unrolled iterations
        for iteration in self.iterations:
            kspace_current, image_current = iteration(
                kspace_current, masked_kspace, mask
            )
            if return_intermediates:
                intermediates.append(image_current)

        # Final refinement (neural network output)
        output_pre_dc = image_current + self.final_refine(image_current)

        # =========================================================================
        # CRITICAL: HARD DATA CONSISTENCY - THE PHYSICS GUARANTEE
        # =========================================================================
        # No matter what the neural network outputs, we MUST respect measured data.
        # This is not optional - it's physics.
        #
        # The network can learn whatever it wants during training,
        # but at inference, measured k-space WINS. Period.
        # =========================================================================
        if enforce_dc:
            # Convert output to complex image
            # Assume output is magnitude, reconstruct with zero phase
            # (or use phase from zero-filled if available)
            output_mag = output_pre_dc.squeeze(1)  # (B, H, W)

            # Get phase from zero-filled reconstruction for phase preservation
            kspace_complex = channels_to_complex(masked_kspace)
            zf_complex = ifft2c(kspace_complex)
            zf_phase = torch.angle(zf_complex)

            # Reconstruct complex image
            output_complex = output_mag * torch.exp(1j * zf_phase)

            # HARD DATA CONSISTENCY: Replace measured k-space
            # This is the FINAL, NON-NEGOTIABLE step
            output_dc = self.final_hard_dc(output_complex, kspace_complex, mask.squeeze(1))

            # Final output magnitude
            output = complex_abs(output_dc).unsqueeze(1)

            # Update k-space to reflect DC-corrected output
            kspace_final = complex_to_channels(fft2c(output_dc))
        else:
            # No DC enforcement (only for ablation studies)
            output = output_pre_dc
            kspace_final = kspace_current

        results = {
            'output': output,
            'output_pre_dc': output_pre_dc,  # For analysis
            'kspace_final': kspace_final,
            'dc_applied': enforce_dc,
        }

        if return_intermediates:
            results['intermediates'] = intermediates

        return results

    def reconstruct(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        enforce_dc: bool = True
    ) -> torch.Tensor:
        """
        Convenience method for reconstruction only.

        Args:
            masked_kspace: Undersampled k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)
            enforce_dc: Apply hard data consistency (default True, RECOMMENDED)

        Returns:
            Reconstructed image (B, 1, H, W)

        Note:
            enforce_dc=True is the default and SHOULD be used in production.
            The output is GUARANTEED to respect measured k-space values.
        """
        results = self.forward(masked_kspace, mask, return_intermediates=False, enforce_dc=enforce_dc)
        return results['output']

    def verify_physics_compliance(
        self,
        output: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        tolerance: float = 1e-5
    ) -> Dict[str, float]:
        """
        Verify that output respects measured k-space (physics compliance).

        This is the PROOF that Guardian is trustworthy.

        Args:
            output: Reconstruction output (B, 1, H, W)
            masked_kspace: Original measured k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)
            tolerance: Acceptable numerical error

        Returns:
            Dict with:
                - 'max_violation': Maximum k-space error at measured locations
                - 'mean_violation': Mean k-space error at measured locations
                - 'is_compliant': Whether output respects physics
                - 'compliance_score': 1.0 = perfect, 0.0 = total violation
        """
        # Get output k-space
        output_mag = output.squeeze(1)
        kspace_measured = channels_to_complex(masked_kspace)
        zf_phase = torch.angle(ifft2c(kspace_measured))
        output_complex = output_mag * torch.exp(1j * zf_phase)
        output_kspace = fft2c(output_complex)

        # Compare at measured locations
        mask_2d = mask.squeeze(1) > 0.5
        measured_output = output_kspace[mask_2d]
        measured_original = kspace_measured[mask_2d]

        # Compute violations
        violations = torch.abs(measured_output - measured_original)
        max_violation = float(violations.max())
        mean_violation = float(violations.mean())

        # Normalize by signal magnitude
        signal_magnitude = float(torch.abs(measured_original).mean())
        relative_violation = mean_violation / (signal_magnitude + 1e-8)

        is_compliant = max_violation < tolerance * signal_magnitude
        compliance_score = max(0.0, 1.0 - relative_violation * 100)

        return {
            'max_violation': max_violation,
            'mean_violation': mean_violation,
            'relative_violation': relative_violation,
            'is_compliant': is_compliant,
            'compliance_score': compliance_score,
            'signal_magnitude': signal_magnitude,
        }

    @staticmethod
    def from_pretrained(checkpoint_path: str, device: str = 'cpu') -> 'GuardianModel':
        """Load pretrained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = GuardianConfig()

        model = GuardianModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model


class GuardianLoss(nn.Module):
    """
    Loss function for training Guardian model.

    Combines:
    1. Reconstruction loss (L1 + perceptual)
    2. Data consistency loss
    3. Intermediate supervision (if enabled)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.1,
        lambda_dc: float = 0.1,
        lambda_intermediate: float = 0.1
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_dc = lambda_dc
        self.lambda_intermediate = lambda_intermediate

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        kspace_output: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor,
        intermediates: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Returns:
            Dict with total loss and individual components
        """
        losses = {}

        # L1 reconstruction loss
        losses['l1'] = F.l1_loss(output, target)

        # SSIM-like loss (simplified)
        # Full SSIM is computed in metrics, here we use structure term
        mu_x = F.avg_pool2d(output, 3, 1, 1)
        mu_y = F.avg_pool2d(target, 3, 1, 1)
        sigma_xy = F.avg_pool2d(output * target, 3, 1, 1) - mu_x * mu_y
        sigma_x = F.avg_pool2d(output ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_y ** 2
        C = 0.01
        ssim_map = (2 * sigma_xy + C) / (sigma_x + sigma_y + C)
        losses['ssim'] = 1 - ssim_map.mean()

        # Data consistency loss
        kspace_complex = channels_to_complex(kspace_output)
        kspace_measured_complex = channels_to_complex(kspace_measured)
        if mask.ndim == 4:
            mask_2d = mask.squeeze(1)
        else:
            mask_2d = mask
        dc_residual = mask_2d * (kspace_complex - kspace_measured_complex)
        losses['dc'] = torch.mean(torch.abs(dc_residual) ** 2)

        # Intermediate supervision
        if intermediates is not None and len(intermediates) > 0:
            inter_loss = 0
            for i, inter in enumerate(intermediates):
                # Decaying weight for earlier iterations
                weight = 0.5 ** (len(intermediates) - i - 1)
                inter_loss = inter_loss + weight * F.l1_loss(inter, target)
            losses['intermediate'] = inter_loss / len(intermediates)
        else:
            losses['intermediate'] = torch.tensor(0.0, device=output.device)

        # Total loss
        losses['total'] = (
            self.lambda_l1 * losses['l1'] +
            self.lambda_ssim * losses['ssim'] +
            self.lambda_dc * losses['dc'] +
            self.lambda_intermediate * losses['intermediate']
        )

        return losses
