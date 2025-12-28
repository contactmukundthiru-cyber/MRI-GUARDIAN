"""
Dual Domain Networks for MRI Reconstruction

Dual-domain means processing BOTH k-space AND image domain,
exchanging information between them.

WHY DUAL DOMAIN?
===============
1. K-space captures global frequency information
2. Image domain captures local spatial features
3. Each domain has different artifacts from undersampling
4. Combining both domains gives more complete information

ARCHITECTURE PATTERNS:
=====================
1. Alternating: K-space block → Image block → K-space block → ...
2. Parallel: Process both domains simultaneously, then fuse
3. Cross-attention: Exchange information via attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .unet import ConvBlock, ResConvBlock
from ..data.kspace_ops import fft2c, ifft2c, complex_to_channels, channels_to_complex, complex_abs


class KSpaceNet(nn.Module):
    """
    K-Space Processing Network.

    Specialized for frequency domain processing.
    Key features:
    - Instance normalization (handles varying k-space scale)
    - Residual learning (easier to learn identity)
    - Mask-aware processing (different treatment for measured vs unmeasured)
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_blocks: int = 5,
        use_mask_conditioning: bool = True
    ):
        """
        Args:
            in_channels: 2 for complex (real, imag)
            hidden_channels: Number of hidden features
            num_blocks: Number of residual blocks
            use_mask_conditioning: Condition network on sampling mask
        """
        super().__init__()

        self.use_mask_conditioning = use_mask_conditioning

        # Input projection (optionally including mask)
        input_ch = in_channels + 1 if use_mask_conditioning else in_channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_ch, hidden_channels, 3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.InstanceNorm2d(hidden_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.InstanceNorm2d(hidden_channels),
                )
            )

        # Output projection
        self.output_conv = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(
        self,
        kspace: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process k-space.

        Args:
            kspace: K-space data (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)

        Returns:
            Refined k-space (B, 2, H, W)
        """
        if self.use_mask_conditioning and mask is not None:
            x = torch.cat([kspace, mask], dim=1)
        else:
            x = kspace

        h = self.input_conv(x)

        for block in self.blocks:
            h = h + block(h)  # Residual

        correction = self.output_conv(h)

        # Residual learning
        return kspace + correction


class ImageDomainNet(nn.Module):
    """
    Image Domain Processing Network.

    Standard image-domain CNN with multi-scale processing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_scales: int = 3
    ):
        """
        Args:
            in_channels: 1 for magnitude, 2 for complex
            hidden_channels: Number of hidden features
            num_scales: Number of multi-scale levels
        """
        super().__init__()

        self.num_scales = num_scales

        # Multi-scale encoders
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = in_channels
        for scale in range(num_scales):
            out_ch = hidden_channels * (2 ** scale)
            self.encoders.append(ResConvBlock(channels, out_ch))
            if scale < num_scales - 1:
                self.pools.append(nn.MaxPool2d(2))
            channels = out_ch

        # Multi-scale decoders
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for scale in range(num_scales - 2, -1, -1):
            in_ch = hidden_channels * (2 ** (scale + 1))
            out_ch = hidden_channels * (2 ** scale)
            self.upsamples.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decoders.append(ResConvBlock(out_ch * 2, out_ch))

        self.output_conv = nn.Conv2d(hidden_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process image."""
        # Encoder
        skips = []
        for scale in range(self.num_scales):
            x = self.encoders[scale](x)
            if scale < self.num_scales - 1:
                skips.append(x)
                x = self.pools[scale](x)

        # Decoder
        for scale, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            x = up(x)
            skip = skips[-(scale + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.output_conv(x)


class CrossDomainAttention(nn.Module):
    """
    Cross-Domain Attention Module.

    Allows k-space and image domain features to attend to each other.
    This helps exchange complementary information.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()

        self.kspace_norm = nn.InstanceNorm2d(channels)
        self.image_norm = nn.InstanceNorm2d(channels)

        self.k_to_i_attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.i_to_k_attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.i_proj = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        kspace_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention between domains.

        Args:
            kspace_features: (B, C, H, W)
            image_features: (B, C, H, W)

        Returns:
            Updated kspace_features, image_features
        """
        B, C, H, W = kspace_features.shape

        # Normalize
        k_norm = self.kspace_norm(kspace_features)
        i_norm = self.image_norm(image_features)

        # Flatten for attention
        k_flat = k_norm.flatten(2).transpose(1, 2)  # (B, H*W, C)
        i_flat = i_norm.flatten(2).transpose(1, 2)

        # Cross-attention: image attends to k-space
        i_updated, _ = self.k_to_i_attention(i_flat, k_flat, k_flat)
        i_updated = i_updated.transpose(1, 2).reshape(B, C, H, W)
        i_updated = self.i_proj(i_updated)

        # Cross-attention: k-space attends to image
        k_updated, _ = self.i_to_k_attention(k_flat, i_flat, i_flat)
        k_updated = k_updated.transpose(1, 2).reshape(B, C, H, W)
        k_updated = self.k_proj(k_updated)

        return kspace_features + k_updated, image_features + i_updated


class DualDomainBlock(nn.Module):
    """
    Single block of dual-domain processing.

    Flow:
    1. Process k-space
    2. Transform to image domain
    3. Process image
    4. Optional cross-domain attention
    5. Transform back to k-space
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        use_cross_attention: bool = True
    ):
        super().__init__()

        self.kspace_net = KSpaceNet(
            in_channels=2,
            hidden_channels=hidden_channels,
            num_blocks=3
        )

        self.image_net = ImageDomainNet(
            in_channels=1,
            hidden_channels=hidden_channels,
            num_scales=3
        )

        if use_cross_attention:
            self.cross_attention = CrossDomainAttention(hidden_channels)
        else:
            self.cross_attention = None

        # Feature extractors for cross-attention
        if use_cross_attention:
            self.k_feature_extract = nn.Conv2d(2, hidden_channels, 3, padding=1)
            self.i_feature_extract = nn.Conv2d(1, hidden_channels, 3, padding=1)
            self.k_feature_inject = nn.Conv2d(hidden_channels, 2, 3, padding=1)
            self.i_feature_inject = nn.Conv2d(hidden_channels, 1, 3, padding=1)

    def forward(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-domain processing.

        Args:
            kspace: K-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)

        Returns:
            refined_kspace, refined_image
        """
        # K-space processing
        kspace_refined = self.kspace_net(kspace, mask)

        # Transform to image
        kspace_complex = channels_to_complex(kspace_refined)
        image_complex = ifft2c(kspace_complex)
        image_mag = complex_abs(image_complex).unsqueeze(1)
        image_phase = torch.angle(image_complex)

        # Image processing
        image_refined = image_mag + self.image_net(image_mag)

        # Cross-domain attention
        if self.cross_attention is not None:
            k_features = self.k_feature_extract(kspace_refined)
            i_features = self.i_feature_extract(image_refined)

            k_features, i_features = self.cross_attention(k_features, i_features)

            kspace_refined = kspace_refined + self.k_feature_inject(k_features)
            image_refined = image_refined + self.i_feature_inject(i_features)

        # Reconstruct complex and back to k-space
        image_complex_refined = image_refined.squeeze(1) * torch.exp(1j * image_phase)
        kspace_output = complex_to_channels(fft2c(image_complex_refined))

        return kspace_output, image_refined


class DualDomainNet(nn.Module):
    """
    Complete Dual-Domain Network for MRI Reconstruction.

    Unrolls multiple dual-domain blocks with data consistency.
    """

    def __init__(
        self,
        num_iterations: int = 6,
        hidden_channels: int = 48,
        use_cross_attention: bool = True,
        dc_lambda: float = 0.9
    ):
        """
        Args:
            num_iterations: Number of unrolled iterations
            hidden_channels: Hidden feature channels
            use_cross_attention: Use cross-domain attention
            dc_lambda: Data consistency weight
        """
        super().__init__()

        self.num_iterations = num_iterations

        # Dual-domain blocks
        self.blocks = nn.ModuleList([
            DualDomainBlock(hidden_channels, use_cross_attention)
            for _ in range(num_iterations)
        ])

        # Learnable DC weights per iteration
        self.dc_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(dc_lambda))
            for _ in range(num_iterations)
        ])

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            masked_kspace: Undersampled k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)
            return_intermediates: Return all intermediate outputs

        Returns:
            Dict with 'output', 'kspace', and optionally 'intermediates'
        """
        kspace_current = masked_kspace.clone()
        intermediates = []

        for i, block in enumerate(self.blocks):
            kspace_current, image_current = block(kspace_current, mask)

            # Data consistency
            lam = torch.sigmoid(self.dc_lambdas[i])
            kspace_complex = channels_to_complex(kspace_current)
            kspace_measured_complex = channels_to_complex(masked_kspace)

            if mask.ndim == 4:
                mask_2d = mask.squeeze(1)
            else:
                mask_2d = mask

            kspace_dc = lam * mask_2d * kspace_measured_complex + (1 - lam * mask_2d) * kspace_complex
            kspace_current = complex_to_channels(kspace_dc)

            if return_intermediates:
                intermediates.append(image_current)

        # Final image from DC'd k-space
        final_complex = ifft2c(channels_to_complex(kspace_current))
        final_image = complex_abs(final_complex).unsqueeze(1)

        results = {
            'output': final_image,
            'kspace': kspace_current,
        }

        if return_intermediates:
            results['intermediates'] = intermediates

        return results

    def reconstruct(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method for reconstruction only."""
        return self.forward(masked_kspace, mask)['output']
