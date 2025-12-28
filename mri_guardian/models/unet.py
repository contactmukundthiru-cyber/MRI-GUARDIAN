"""
UNet Architecture for MRI Reconstruction

The UNet is the workhorse of medical image analysis.
It's an encoder-decoder architecture with skip connections.

WHY UNET WORKS FOR MRI:
======================
1. Multi-scale processing: Captures both fine details and global context
2. Skip connections: Preserves high-frequency information through the network
3. Symmetrical design: Natural for image-to-image tasks
4. Proven performance: Standard baseline in MRI reconstruction

ARCHITECTURE:
============
Input → [Encoder (downsample)] → Bottleneck → [Decoder (upsample)] → Output
         ↓_____skip_____↑       ↓_____skip_____↑

Each encoder block: Conv → Norm → ReLU → Conv → Norm → ReLU → Pool
Each decoder block: Upsample → Concat(skip) → Conv → Norm → ReLU → Conv → Norm → ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv → Norm → Activation (×2)

    This is the fundamental building block of the UNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super().__init__()

        # First conv
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_batch_norm)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activation
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())

        # Second conv
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=not use_batch_norm)
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResConvBlock(nn.Module):
    """
    Residual convolutional block: x + ConvBlock(x)

    Residual connections help with:
    1. Gradient flow during training
    2. Learning identity mappings
    3. Deeper networks without degradation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        super().__init__()

        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size, padding, use_batch_norm, activation
        )

        # Skip connection with 1x1 conv if channels change
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.

    Attention helps the model focus on relevant parts of the image,
    especially useful for detecting pathology.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Reshape for attention: (B, C, H, W) → (B, H*W, C)
        x_flat = x_norm.flatten(2).transpose(1, 2)

        # Self-attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)

        # Reshape back: (B, H*W, C) → (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)

        # Residual connection
        return x + attn_out


class UNetEncoder(nn.Module):
    """
    UNet Encoder: Progressive downsampling with feature extraction.

    Each level:
    1. Applies convolutional block (extracts features)
    2. Saves output for skip connection
    3. Downsamples via max pooling
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
        use_residual: bool = True,
        use_attention: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.num_levels = num_levels
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = in_channels
        for level in range(num_levels):
            out_channels = base_channels * (2 ** level)

            if use_residual:
                encoder = ResConvBlock(channels, out_channels)
            else:
                encoder = ConvBlock(channels, out_channels, dropout=dropout)

            self.encoders.append(encoder)

            if level < num_levels - 1:  # No pooling after last encoder
                self.pools.append(nn.MaxPool2d(2))

            channels = out_channels

        # Optional attention at bottleneck
        if use_attention:
            self.attention = AttentionBlock(channels)
        else:
            self.attention = None

        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through encoder.

        Returns:
            features: Bottleneck features
            skip_features: List of features for skip connections
        """
        skip_features = []

        for level in range(self.num_levels):
            x = self.encoders[level](x)
            skip_features.append(x)

            if level < self.num_levels - 1:
                x = self.pools[level](x)

        if self.attention is not None:
            x = self.attention(x)

        return x, skip_features[:-1]  # Don't include bottleneck in skips


class UNetDecoder(nn.Module):
    """
    UNet Decoder: Progressive upsampling with skip connections.

    Each level:
    1. Upsamples features
    2. Concatenates with corresponding encoder features (skip)
    3. Applies convolutional block
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
        use_residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.num_levels = num_levels
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in range(num_levels - 1, 0, -1):
            in_ch = base_channels * (2 ** level)
            out_ch = base_channels * (2 ** (level - 1))

            # Transposed convolution for upsampling
            self.upsamples.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )

            # After concat with skip: out_ch * 2 → out_ch
            if use_residual:
                self.decoders.append(ResConvBlock(out_ch * 2, out_ch))
            else:
                self.decoders.append(ConvBlock(out_ch * 2, out_ch, dropout=dropout))

        # Final 1x1 conv to output channels
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            x: Bottleneck features
            skip_features: Features from encoder (in order: shallow → deep)
        """
        # Reverse skip features (we use deep → shallow)
        skip_features = skip_features[::-1]

        for level in range(self.num_levels - 1):
            x = self.upsamples[level](x)

            # Handle size mismatch (can happen with odd dimensions)
            skip = skip_features[level]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)

            x = self.decoders[level](x)

        return self.final_conv(x)


class UNet(nn.Module):
    """
    Complete UNet for MRI Reconstruction.

    This is the IMAGE-DOMAIN baseline model:
    - Input: Zero-filled reconstruction (1 channel magnitude)
    - Output: Reconstructed image (1 channel magnitude)

    No physics constraints! Pure data-driven learning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
        use_residual: bool = True,
        use_attention: bool = False,
        dropout: float = 0.0,
        residual_learning: bool = True  # Learn residual: output = input + network(input)
    ):
        """
        Args:
            in_channels: Number of input channels (1 for magnitude, 2 for complex)
            out_channels: Number of output channels
            base_channels: Base number of feature channels
            num_levels: Number of encoder/decoder levels
            use_residual: Use residual blocks
            use_attention: Use attention at bottleneck
            dropout: Dropout probability
            residual_learning: If True, learn residual (output = input + prediction)
        """
        super().__init__()

        self.residual_learning = residual_learning

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=num_levels,
            use_residual=use_residual,
            use_attention=use_attention,
            dropout=dropout
        )

        self.decoder = UNetDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            num_levels=num_levels,
            use_residual=use_residual,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Reconstructed image (B, out_channels, H, W)
        """
        # Encode
        bottleneck, skip_features = self.encoder(x)

        # Decode
        output = self.decoder(bottleneck, skip_features)

        # Residual learning
        if self.residual_learning and x.shape == output.shape:
            output = output + x

        return output

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class UNet2Channel(UNet):
    """
    UNet for complex-valued MRI (2-channel: real + imaginary).

    Processes both real and imaginary parts together,
    allowing the network to learn complex relationships.
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_levels: int = 4,
        **kwargs
    ):
        super().__init__(
            in_channels=2,
            out_channels=2,
            base_channels=base_channels,
            num_levels=num_levels,
            **kwargs
        )


class CascadeUNet(nn.Module):
    """
    Cascaded UNet: Multiple UNets in sequence.

    Each UNet refines the previous output.
    Common in MRI reconstruction for iterative refinement.
    """

    def __init__(
        self,
        num_cascades: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        share_weights: bool = False
    ):
        """
        Args:
            num_cascades: Number of UNets in cascade
            share_weights: If True, all UNets share weights
        """
        super().__init__()

        self.num_cascades = num_cascades
        self.share_weights = share_weights

        if share_weights:
            self.unets = nn.ModuleList([
                UNet(in_channels, out_channels, base_channels, num_levels)
            ])
        else:
            self.unets = nn.ModuleList([
                UNet(in_channels, out_channels, base_channels, num_levels)
                for _ in range(num_cascades)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through cascade."""
        for i in range(self.num_cascades):
            if self.share_weights:
                x = self.unets[0](x)
            else:
                x = self.unets[i](x)
        return x
