"""
Image Quality Metrics for MRI Reconstruction

Standard metrics used to evaluate reconstruction quality:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- NRMSE: Normalized Root Mean Square Error
- HFEN: High-Frequency Error Norm

All metrics compare a reconstruction against ground truth.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class ImageQualityMetrics:
    """Container for image quality metrics."""
    psnr: float
    ssim: float
    nrmse: float
    hfen: float
    vif: Optional[float] = None


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX² / MSE)

    Higher is better. Typical values for good MRI recon: 30-40 dB.

    INTUITION:
    PSNR measures the ratio between the maximum possible signal
    and the noise (error). Higher PSNR = less noise = better image.

    Args:
        pred: Predicted image (B, 1, H, W) or (H, W)
        target: Ground truth image
        data_range: Maximum value range (computed if None)

    Returns:
        PSNR value in dB
    """
    # Ensure same shape
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)

    if data_range is None:
        data_range = target.max() - target.min()

    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 10 * torch.log10(data_range ** 2 / mse)

    return psnr.item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: Optional[float] = None,
    return_map: bool = False
) -> Union[float, Tuple[float, torch.Tensor]]:
    """
    Compute Structural Similarity Index (SSIM).

    SSIM compares luminance, contrast, and structure.
    Range: [-1, 1], where 1 = identical.

    INTUITION:
    SSIM measures how "structurally similar" two images are.
    It's better than PSNR because it considers local structure,
    not just pixel-by-pixel differences.

    Args:
        pred: Predicted image
        target: Ground truth image
        window_size: Local window size
        data_range: Data range
        return_map: Also return SSIM map

    Returns:
        SSIM value (and optionally SSIM map)
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)

    if data_range is None:
        data_range = target.max() - target.min()

    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, device=pred.device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size)

    # Padding
    pad = window_size // 2

    # Local means
    mu1 = F.conv2d(pred, window, padding=pad)
    mu2 = F.conv2d(target, window, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = F.conv2d(pred ** 2, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    ssim_val = ssim_map.mean().item()

    if return_map:
        return ssim_val, ssim_map
    return ssim_val


def compute_nrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalization: str = "euclidean"
) -> float:
    """
    Compute Normalized Root Mean Square Error.

    NRMSE = RMSE / norm(target)

    Lower is better. Range: [0, ∞), typically [0, 1] for good recon.

    Args:
        pred: Predicted image
        target: Ground truth image
        normalization: "euclidean", "min-max", or "mean"

    Returns:
        NRMSE value
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)

    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)

    if normalization == "euclidean":
        norm = torch.sqrt(torch.mean(target ** 2))
    elif normalization == "min-max":
        norm = target.max() - target.min()
    elif normalization == "mean":
        norm = target.mean()
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    nrmse = rmse / (norm + 1e-8)

    return nrmse.item()


def compute_hfen(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: float = 1.5
) -> float:
    """
    Compute High-Frequency Error Norm (HFEN).

    HFEN measures error in high-frequency (edge) content.
    Uses Laplacian of Gaussian (LoG) filter.

    Lower is better. Important for MRI where edges matter.

    INTUITION:
    HFEN focuses on edges/details rather than smooth regions.
    A reconstruction can have low PSNR but high HFEN if it
    gets edges wrong.

    Args:
        pred: Predicted image
        target: Ground truth image
        sigma: Gaussian sigma for LoG filter

    Returns:
        HFEN value
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)

    # Create LoG filter
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create coordinate grid
    coords = torch.arange(kernel_size, device=pred.device).float() - kernel_size // 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')

    # Laplacian of Gaussian
    # LoG(x,y) = -1/(π*σ⁴) * [1 - (x²+y²)/(2σ²)] * exp(-(x²+y²)/(2σ²))
    r_sq = x ** 2 + y ** 2
    log_kernel = -1 / (np.pi * sigma ** 4) * (1 - r_sq / (2 * sigma ** 2)) * torch.exp(-r_sq / (2 * sigma ** 2))
    log_kernel = log_kernel - log_kernel.mean()  # Zero mean
    log_kernel = log_kernel.view(1, 1, kernel_size, kernel_size)

    # Apply filter
    pad = kernel_size // 2
    pred_filtered = F.conv2d(pred, log_kernel, padding=pad)
    target_filtered = F.conv2d(target, log_kernel, padding=pad)

    # HFEN = ||LoG(pred) - LoG(target)||₂ / ||LoG(target)||₂
    diff_norm = torch.sqrt(torch.sum((pred_filtered - target_filtered) ** 2))
    target_norm = torch.sqrt(torch.sum(target_filtered ** 2))

    hfen = diff_norm / (target_norm + 1e-8)

    return hfen.item()


def compute_vif(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma_nsq: float = 2.0
) -> float:
    """
    Compute Visual Information Fidelity (VIF).

    VIF measures information shared between images.
    Based on natural scene statistics model.

    Range: [0, 1], higher is better.

    Args:
        pred: Predicted image
        target: Ground truth image
        sigma_nsq: Noise variance

    Returns:
        VIF value
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)

    # Gaussian filter
    kernel_size = 11
    sigma = 1.5
    coords = torch.arange(kernel_size, device=pred.device).float() - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window / window.sum()
    window = window.view(1, 1, kernel_size, kernel_size)

    pad = kernel_size // 2

    # Local statistics
    mu1 = F.conv2d(target, window, padding=pad)
    mu2 = F.conv2d(pred, window, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2

    sigma1_sq = F.conv2d(target ** 2, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(pred ** 2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(target * pred, window, padding=pad) - mu1 * mu2

    sigma1_sq = torch.clamp(sigma1_sq, min=1e-10)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-10)

    # VIF computation
    g = sigma12 / (sigma1_sq + 1e-10)
    sv_sq = sigma2_sq - g * sigma12

    g = torch.clamp(g, min=0)
    sv_sq = torch.clamp(sv_sq, min=1e-10)

    num = torch.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq))
    den = torch.log10(1 + sigma1_sq / sigma_nsq)

    vif = torch.sum(num) / (torch.sum(den) + 1e-10)

    return torch.clamp(vif, 0, 1).item()


def compute_all_image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None
) -> ImageQualityMetrics:
    """
    Compute all image quality metrics.

    Args:
        pred: Predicted image
        target: Ground truth image
        data_range: Data range

    Returns:
        ImageQualityMetrics with all metrics
    """
    psnr = compute_psnr(pred, target, data_range)
    ssim = compute_ssim(pred, target, data_range=data_range)
    nrmse = compute_nrmse(pred, target)
    hfen = compute_hfen(pred, target)

    try:
        vif = compute_vif(pred, target)
    except Exception:
        vif = None

    return ImageQualityMetrics(
        psnr=psnr,
        ssim=ssim,
        nrmse=nrmse,
        hfen=hfen,
        vif=vif
    )


class MetricAggregator:
    """
    Aggregates metrics over multiple samples.

    Computes mean, std, min, max for reporting.
    """

    def __init__(self):
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'nrmse': [],
            'hfen': [],
        }

    def add(self, pred: torch.Tensor, target: torch.Tensor):
        """Add metrics for a sample."""
        result = compute_all_image_metrics(pred, target)
        self.metrics['psnr'].append(result.psnr)
        self.metrics['ssim'].append(result.ssim)
        self.metrics['nrmse'].append(result.nrmse)
        self.metrics['hfen'].append(result.hfen)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics."""
        summary = {}
        for name, values in self.metrics.items():
            if len(values) > 0:
                arr = np.array(values)
                summary[name] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                }
        return summary

    def reset(self):
        """Reset accumulated metrics."""
        for key in self.metrics:
            self.metrics[key] = []
