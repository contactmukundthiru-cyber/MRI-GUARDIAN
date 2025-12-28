"""
Data Consistency Layers for MRI Reconstruction

Data Consistency (DC) is THE most important concept for physics-guided MRI reconstruction.

THE KEY IDEA:
=============
When we have undersampled MRI data:
- Some k-space points are MEASURED (ground truth)
- Other k-space points are MISSING (need to estimate)

Any valid reconstruction MUST:
1. Exactly match the measured k-space points
2. Only "fill in" the missing points

This is what DC enforces:
    DC(x) = F⁻¹( M * k_measured + (1-M) * F(x) )

Where:
- x: Current image estimate
- F: Fourier transform
- M: Sampling mask (1=measured, 0=missing)
- k_measured: Original measured k-space

In words: "Keep the measured samples, only change the missing samples."

WHY IS THIS SO IMPORTANT?
========================
Without DC, neural networks can "make up" k-space values that contradict
the actual measurements. This leads to:
- Images that look good but are physically impossible
- Hallucinated structures
- Suppressed real structures

DC ensures our reconstruction is CONSISTENT with reality.

THREE FLAVORS OF DC:
===================
1. Hard DC: Strictly replace measured samples (no compromise)
2. Soft DC: Blend measured and predicted (allows some error)
3. Gradient DC: Add DC as a loss term (learned balance)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..data.kspace_ops import fft2c, ifft2c, complex_to_channels, channels_to_complex


class DataConsistencyLayer(nn.Module):
    """
    Base Data Consistency Layer.

    Takes an image estimate and enforces consistency with measured k-space.
    This is a DIFFERENTIABLE operation, so gradients flow through it.
    """

    def __init__(self, mode: str = "hard"):
        """
        Args:
            mode: "hard" for strict replacement, "soft" for weighted blend
        """
        super().__init__()
        self.mode = mode

    def forward(
        self,
        image_pred: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply data consistency.

        Args:
            image_pred: Predicted image from network
                - (B, H, W) complex or (B, 2, H, W) real
            kspace_measured: Original measured k-space (same format)
            mask: Sampling mask (B, 1, H, W) or (B, H, W)

        Returns:
            DC-corrected image (same format as input)
        """
        raise NotImplementedError("Subclasses must implement forward()")


class HardDataConsistency(DataConsistencyLayer):
    """
    Hard Data Consistency: Strictly enforce measured samples.

    DC(x) = F⁻¹( M * k_measured + (1-M) * F(x) )

    This is the strictest form - measured values are NEVER changed.
    Use this when you trust the measurements completely.
    """

    def __init__(self):
        super().__init__(mode="hard")

    def forward(
        self,
        image_pred: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply hard data consistency."""
        # Detect input format
        use_2ch = False
        if image_pred.ndim == 4 and image_pred.shape[1] == 2:
            use_2ch = True
            image_pred = channels_to_complex(image_pred)
            kspace_measured = channels_to_complex(kspace_measured)

        # Handle mask dimensions
        if mask.ndim == 4:
            mask = mask.squeeze(1)  # (B, H, W)

        # Transform predicted image to k-space
        kspace_pred = fft2c(image_pred)

        # Combine: measured where mask=1, predicted where mask=0
        kspace_dc = mask * kspace_measured + (1 - mask) * kspace_pred

        # Transform back to image
        image_dc = ifft2c(kspace_dc)

        if use_2ch:
            return complex_to_channels(image_dc)
        return image_dc


class SoftDataConsistency(DataConsistencyLayer):
    """
    Soft Data Consistency: Weighted blend of measured and predicted.

    DC(x) = F⁻¹( λ * k_measured + (1-λ*M) * F(x) )

    Where λ ∈ [0, 1] controls how much we trust measurements.
    - λ = 1: Hard DC (fully trust measurements)
    - λ = 0: No DC (fully trust predictions)
    - λ = 0.5: Equal blend

    Use this when measurements might have some noise/error.
    λ can be learned during training!
    """

    def __init__(self, lambda_init: float = 1.0, learnable: bool = False):
        """
        Args:
            lambda_init: Initial weight for measured data
            learnable: If True, λ is a learnable parameter
        """
        super().__init__(mode="soft")

        if learnable:
            self.lambda_dc = nn.Parameter(torch.tensor(lambda_init))
        else:
            self.register_buffer("lambda_dc", torch.tensor(lambda_init))

    def forward(
        self,
        image_pred: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply soft data consistency."""
        use_2ch = False
        if image_pred.ndim == 4 and image_pred.shape[1] == 2:
            use_2ch = True
            image_pred = channels_to_complex(image_pred)
            kspace_measured = channels_to_complex(kspace_measured)

        if mask.ndim == 4:
            mask = mask.squeeze(1)

        # Get lambda in valid range
        lam = torch.sigmoid(self.lambda_dc)  # Ensures 0 < λ < 1

        kspace_pred = fft2c(image_pred)

        # Soft combination
        kspace_dc = lam * mask * kspace_measured + (1 - lam * mask) * kspace_pred

        image_dc = ifft2c(kspace_dc)

        if use_2ch:
            return complex_to_channels(image_dc)
        return image_dc


class GradientDataConsistency(DataConsistencyLayer):
    """
    Gradient-based Data Consistency.

    Instead of replacing k-space, add a gradient step toward measured data:
    x_new = x - α * F⁻¹(M * (F(x) - k_measured))

    This is like gradient descent on the data fidelity term:
    L_DC = ||M * (F(x) - k_measured)||²

    Benefits:
    - Smoother than hard replacement
    - Step size α can be learned
    - Natural for unrolled optimization networks
    """

    def __init__(self, step_size: float = 1.0, learnable: bool = True):
        """
        Args:
            step_size: Gradient step size
            learnable: If True, step size is learnable
        """
        super().__init__(mode="gradient")

        if learnable:
            self.step_size = nn.Parameter(torch.tensor(step_size))
        else:
            self.register_buffer("step_size", torch.tensor(step_size))

    def forward(
        self,
        image_pred: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gradient data consistency."""
        use_2ch = False
        if image_pred.ndim == 4 and image_pred.shape[1] == 2:
            use_2ch = True
            image_pred = channels_to_complex(image_pred)
            kspace_measured = channels_to_complex(kspace_measured)

        if mask.ndim == 4:
            mask = mask.squeeze(1)

        # Compute k-space of prediction
        kspace_pred = fft2c(image_pred)

        # Compute residual at measured locations
        residual = mask * (kspace_pred - kspace_measured)

        # Gradient step in image domain
        gradient = ifft2c(residual)
        image_dc = image_pred - self.step_size * gradient

        if use_2ch:
            return complex_to_channels(image_dc)
        return image_dc


class UnrolledDCBlock(nn.Module):
    """
    Data Consistency Block for Unrolled Networks.

    Combines a neural network refinement with data consistency:
    1. Neural network: x' = CNN(x)
    2. Data consistency: x'' = DC(x')

    This is the building block for "unrolled" optimization networks
    like Variational Networks, MoDL, etc.
    """

    def __init__(
        self,
        network: nn.Module,
        dc_type: str = "hard",
        dc_weight: float = 1.0,
        learnable_dc: bool = False,
    ):
        """
        Args:
            network: Neural network for image refinement
            dc_type: Type of DC ("hard", "soft", "gradient")
            dc_weight: Weight for soft/gradient DC
            learnable_dc: Whether DC parameters are learnable
        """
        super().__init__()
        self.network = network

        if dc_type == "hard":
            self.dc = HardDataConsistency()
        elif dc_type == "soft":
            self.dc = SoftDataConsistency(dc_weight, learnable_dc)
        elif dc_type == "gradient":
            self.dc = GradientDataConsistency(dc_weight, learnable_dc)
        else:
            raise ValueError(f"Unknown DC type: {dc_type}")

    def forward(
        self,
        image: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply network refinement followed by DC.

        Args:
            image: Current image estimate (B, 2, H, W) or (B, 1, H, W)
            kspace_measured: Measured k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)

        Returns:
            Refined and DC-corrected image
        """
        # Network refinement (expects magnitude or 2-channel)
        image_refined = self.network(image)

        # Apply data consistency
        image_dc = self.dc(image_refined, kspace_measured, mask)

        return image_dc


class DataConsistencyLoss(nn.Module):
    """
    Data Consistency Loss Function.

    Measures how well a reconstruction matches the measured k-space:
    L_DC = ||M * (F(x) - k_measured)||²

    Use this as an additional loss term during training.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        image_pred: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DC loss.

        Args:
            image_pred: Predicted image (B, H, W) complex or (B, 2, H, W)
            kspace_measured: Measured k-space
            mask: Sampling mask

        Returns:
            Scalar loss value
        """
        if image_pred.ndim == 4 and image_pred.shape[1] == 2:
            image_pred = channels_to_complex(image_pred)
            kspace_measured = channels_to_complex(kspace_measured)

        if mask.ndim == 4:
            mask = mask.squeeze(1)

        # Predicted k-space
        kspace_pred = fft2c(image_pred)

        # Residual at measured locations
        residual = mask * (kspace_pred - kspace_measured)

        # Squared L2 norm
        loss = torch.sum(torch.abs(residual) ** 2, dim=(-2, -1))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def apply_data_consistency(
    image: torch.Tensor,
    kspace_measured: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "hard"
) -> torch.Tensor:
    """
    Functional interface for data consistency.

    Convenience function for applying DC without creating a module.

    Args:
        image: Image tensor (complex or 2-channel)
        kspace_measured: Measured k-space
        mask: Sampling mask
        mode: "hard", "soft", or "gradient"

    Returns:
        DC-corrected image
    """
    if mode == "hard":
        dc = HardDataConsistency()
    elif mode == "soft":
        dc = SoftDataConsistency()
    elif mode == "gradient":
        dc = GradientDataConsistency()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return dc(image, kspace_measured, mask)
