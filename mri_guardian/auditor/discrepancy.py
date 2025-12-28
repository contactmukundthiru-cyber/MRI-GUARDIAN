"""
Discrepancy Computation for Hallucination Detection

Computes various measures of difference between
black-box and Guardian reconstructions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_intensity_discrepancy(
    image1: torch.Tensor,
    image2: torch.Tensor,
    mode: str = "l1"
) -> torch.Tensor:
    """
    Compute pixel-wise intensity discrepancy.

    Args:
        image1: First image (B, 1, H, W)
        image2: Second image (B, 1, H, W)
        mode: "l1" (absolute) or "l2" (squared)

    Returns:
        Discrepancy map (B, 1, H, W)
    """
    if mode == "l1":
        return torch.abs(image1 - image2)
    elif mode == "l2":
        return (image1 - image2) ** 2
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_structural_discrepancy(
    image1: torch.Tensor,
    image2: torch.Tensor,
    use_sobel: bool = True
) -> torch.Tensor:
    """
    Compute structural (edge-based) discrepancy.

    Compares gradient magnitudes between images.
    Sensitive to edge differences.

    Args:
        image1: First image (B, 1, H, W)
        image2: Second image (B, 1, H, W)
        use_sobel: Use Sobel filters (else simple diff)

    Returns:
        Discrepancy map (B, 1, H, W)
    """
    if use_sobel:
        # Sobel filters
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=image1.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=image1.device).view(1, 1, 3, 3)

        # Compute gradients for both images
        gx1 = F.conv2d(image1, sobel_x, padding=1)
        gy1 = F.conv2d(image1, sobel_y, padding=1)
        grad1 = torch.sqrt(gx1**2 + gy1**2 + 1e-8)

        gx2 = F.conv2d(image2, sobel_x, padding=1)
        gy2 = F.conv2d(image2, sobel_y, padding=1)
        grad2 = torch.sqrt(gx2**2 + gy2**2 + 1e-8)

        # Gradient magnitude difference
        return torch.abs(grad1 - grad2)

    else:
        # Simple finite differences
        dx1 = torch.diff(image1, dim=-1)
        dy1 = torch.diff(image1, dim=-2)

        dx2 = torch.diff(image2, dim=-1)
        dy2 = torch.diff(image2, dim=-2)

        # Pad to original size
        dx1 = F.pad(dx1, (0, 1, 0, 0))
        dy1 = F.pad(dy1, (0, 0, 0, 1))
        dx2 = F.pad(dx2, (0, 1, 0, 0))
        dy2 = F.pad(dy2, (0, 0, 0, 1))

        grad1 = torch.sqrt(dx1**2 + dy1**2 + 1e-8)
        grad2 = torch.sqrt(dx2**2 + dy2**2 + 1e-8)

        return torch.abs(grad1 - grad2)


def compute_frequency_discrepancy(
    image1: torch.Tensor,
    image2: torch.Tensor,
    focus_high_freq: bool = True
) -> torch.Tensor:
    """
    Compute discrepancy in frequency domain.

    Hallucinations often have different frequency content,
    especially in high frequencies.

    Args:
        image1: First image (B, 1, H, W)
        image2: Second image (B, 1, H, W)
        focus_high_freq: Weight high frequencies more

    Returns:
        Discrepancy map (B, 1, H, W)
    """
    # FFT of both images
    fft1 = torch.fft.fft2(image1.squeeze(1))
    fft2 = torch.fft.fft2(image2.squeeze(1))

    # Shift zero frequency to center
    fft1 = torch.fft.fftshift(fft1, dim=(-2, -1))
    fft2 = torch.fft.fftshift(fft2, dim=(-2, -1))

    # Magnitude difference
    mag_diff = torch.abs(torch.abs(fft1) - torch.abs(fft2))

    if focus_high_freq:
        # Create high-pass weight (distance from center)
        B, H, W = mag_diff.shape
        y = torch.linspace(-1, 1, H, device=image1.device)
        x = torch.linspace(-1, 1, W, device=image1.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)

        # Weight: higher at edges (high freq), lower at center (low freq)
        weight = torch.clamp(dist, 0, 1)
        mag_diff = mag_diff * weight.unsqueeze(0)

    # Transform back to spatial domain for visualization
    # (Take inverse FFT of the difference)
    diff_spatial = torch.fft.ifft2(torch.fft.ifftshift(mag_diff, dim=(-2, -1)))
    diff_spatial = torch.abs(diff_spatial).unsqueeze(1)

    return diff_spatial


def compute_local_ssim_discrepancy(
    image1: torch.Tensor,
    image2: torch.Tensor,
    window_size: int = 11
) -> torch.Tensor:
    """
    Compute local SSIM-based discrepancy.

    SSIM measures structural similarity; 1 - SSIM gives discrepancy.

    Args:
        image1: First image (B, 1, H, W)
        image2: Second image (B, 1, H, W)
        window_size: Local window size

    Returns:
        Discrepancy map (B, 1, H, W)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, device=image1.device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size)

    # Local statistics
    mu1 = F.conv2d(image1, window, padding=window_size//2)
    mu2 = F.conv2d(image2, window, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 ** 2, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(image2 ** 2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size//2) - mu1_mu2

    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Discrepancy = 1 - SSIM
    return 1 - ssim_map


def compute_perceptual_discrepancy(
    image1: torch.Tensor,
    image2: torch.Tensor,
    feature_extractor: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Compute perceptual discrepancy using learned features.

    Uses a pre-trained network to extract features,
    then compares features between images.

    Args:
        image1: First image (B, 1, H, W)
        image2: Second image (B, 1, H, W)
        feature_extractor: Optional pre-trained network

    Returns:
        Discrepancy map (B, 1, H, W)
    """
    if feature_extractor is None:
        # Simple learned feature extractor
        feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
        ).to(image1.device)

    # Extract features
    feat1 = feature_extractor(image1)
    feat2 = feature_extractor(image2)

    # Feature difference (average across channels)
    diff = torch.mean(torch.abs(feat1 - feat2), dim=1, keepdim=True)

    # Upsample to original size if needed
    if diff.shape[-2:] != image1.shape[-2:]:
        diff = F.interpolate(diff, size=image1.shape[-2:], mode='bilinear', align_corners=True)

    return diff


def combine_discrepancy_maps(
    intensity_map: torch.Tensor,
    structural_map: torch.Tensor,
    frequency_map: torch.Tensor,
    w_intensity: float = 0.5,
    w_structural: float = 0.3,
    w_frequency: float = 0.2
) -> torch.Tensor:
    """
    Combine multiple discrepancy maps.

    Args:
        intensity_map: Intensity discrepancy (B, 1, H, W)
        structural_map: Structural discrepancy (B, 1, H, W)
        frequency_map: Frequency discrepancy (B, 1, H, W)
        w_intensity: Weight for intensity
        w_structural: Weight for structural
        w_frequency: Weight for frequency

    Returns:
        Combined discrepancy map (B, 1, H, W)
    """
    # Normalize each map
    def normalize(x):
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    i_norm = normalize(intensity_map)
    s_norm = normalize(structural_map)
    f_norm = normalize(frequency_map)

    # Weighted sum
    combined = w_intensity * i_norm + w_structural * s_norm + w_frequency * f_norm

    # Renormalize
    return normalize(combined)


class LearnedDiscrepancyCombiner(nn.Module):
    """
    Learns optimal weights for combining discrepancy maps.

    Uses a small CNN to learn spatially-varying weights.
    """

    def __init__(self, hidden_channels: int = 16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 3, 1),
            nn.Softmax(dim=1)  # Weights sum to 1
        )

    def forward(
        self,
        intensity_map: torch.Tensor,
        structural_map: torch.Tensor,
        frequency_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Learn to combine discrepancy maps.

        Args:
            intensity_map: (B, 1, H, W)
            structural_map: (B, 1, H, W)
            frequency_map: (B, 1, H, W)

        Returns:
            Combined map (B, 1, H, W)
        """
        # Stack maps
        stacked = torch.cat([intensity_map, structural_map, frequency_map], dim=1)

        # Learn weights
        weights = self.net(stacked)

        # Weighted combination
        combined = (
            weights[:, 0:1] * intensity_map +
            weights[:, 1:2] * structural_map +
            weights[:, 2:3] * frequency_map
        )

        return combined
