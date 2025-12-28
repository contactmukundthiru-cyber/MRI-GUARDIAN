"""
Uncertainty Estimation for MRI Reconstruction

Uncertainty helps us know "how confident" we are in our reconstruction.
High uncertainty regions are more likely to contain errors.

WHY UNCERTAINTY MATTERS:
=======================
- Black-box models can be overconfident in wrong predictions
- Guardian with uncertainty shows where it's "unsure"
- High uncertainty + large discrepancy = likely hallucination

METHODS:
=======
1. Monte Carlo Dropout: Run model multiple times with dropout
2. Ensemble: Train multiple models, compare outputs
3. Test-time augmentation: Apply augmentations, measure variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


def enable_dropout(model: nn.Module):
    """Enable dropout layers during evaluation."""
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


def disable_dropout(model: nn.Module):
    """Disable dropout layers."""
    model.eval()


def monte_carlo_uncertainty(
    model: nn.Module,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 10,
    return_samples: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout uncertainty estimation.

    Runs the model multiple times with dropout enabled,
    then computes variance as uncertainty.

    Args:
        model: Model with dropout layers
        masked_kspace: Input k-space (B, 2, H, W)
        mask: Sampling mask (B, 1, H, W)
        num_samples: Number of forward passes
        return_samples: Also return individual samples

    Returns:
        mean: Mean prediction (B, 1, H, W)
        uncertainty: Variance as uncertainty (B, 1, H, W)
        [samples]: Optional list of samples
    """
    model.eval()
    enable_dropout(model)

    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            result = model(masked_kspace, mask)
            output = result['output'] if isinstance(result, dict) else result
            samples.append(output)

    disable_dropout(model)

    # Stack samples
    samples_tensor = torch.stack(samples, dim=0)  # (N, B, 1, H, W)

    # Mean and variance
    mean = samples_tensor.mean(dim=0)
    uncertainty = samples_tensor.var(dim=0)

    if return_samples:
        return mean, uncertainty, samples
    return mean, uncertainty


def ensemble_uncertainty(
    models: List[nn.Module],
    masked_kspace: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensemble-based uncertainty estimation.

    Uses multiple independently trained models
    and measures their disagreement.

    Args:
        models: List of trained models
        masked_kspace: Input k-space
        mask: Sampling mask

    Returns:
        mean: Mean prediction
        uncertainty: Variance as uncertainty
    """
    predictions = []

    with torch.no_grad():
        for model in models:
            model.eval()
            result = model(masked_kspace, mask)
            output = result['output'] if isinstance(result, dict) else result
            predictions.append(output)

    # Stack and compute statistics
    preds_tensor = torch.stack(predictions, dim=0)
    mean = preds_tensor.mean(dim=0)
    uncertainty = preds_tensor.var(dim=0)

    return mean, uncertainty


def test_time_augmentation_uncertainty(
    model: nn.Module,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    num_augmentations: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Test-time augmentation uncertainty.

    Applies different augmentations (flips, rotations),
    runs model, then inverts augmentations and measures variance.

    Args:
        model: Trained model
        masked_kspace: Input k-space
        mask: Sampling mask
        num_augmentations: Number of augmentations

    Returns:
        mean: Mean prediction
        uncertainty: Variance as uncertainty
    """
    model.eval()
    predictions = []

    augmentations = [
        (lambda x, m: (x, m), lambda y: y),  # Identity
        (lambda x, m: (torch.flip(x, [-1]), torch.flip(m, [-1])), lambda y: torch.flip(y, [-1])),  # H-flip
        (lambda x, m: (torch.flip(x, [-2]), torch.flip(m, [-2])), lambda y: torch.flip(y, [-2])),  # V-flip
        (lambda x, m: (torch.rot90(x, 1, [-2, -1]), torch.rot90(m, 1, [-2, -1])), lambda y: torch.rot90(y, -1, [-2, -1])),  # 90°
        (lambda x, m: (torch.rot90(x, 2, [-2, -1]), torch.rot90(m, 2, [-2, -1])), lambda y: torch.rot90(y, -2, [-2, -1])),  # 180°
        (lambda x, m: (torch.rot90(x, 3, [-2, -1]), torch.rot90(m, 3, [-2, -1])), lambda y: torch.rot90(y, -3, [-2, -1])),  # 270°
    ]

    with torch.no_grad():
        for i in range(min(num_augmentations, len(augmentations))):
            aug_fn, inv_fn = augmentations[i]

            # Apply augmentation
            aug_kspace, aug_mask = aug_fn(masked_kspace, mask)

            # Run model
            result = model(aug_kspace, aug_mask)
            output = result['output'] if isinstance(result, dict) else result

            # Invert augmentation
            output = inv_fn(output)
            predictions.append(output)

    # Compute statistics
    preds_tensor = torch.stack(predictions, dim=0)
    mean = preds_tensor.mean(dim=0)
    uncertainty = preds_tensor.var(dim=0)

    return mean, uncertainty


def compute_reconstruction_uncertainty(
    model: nn.Module,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    method: str = "mc_dropout",
    num_samples: int = 10
) -> torch.Tensor:
    """
    Unified interface for uncertainty estimation.

    Args:
        model: Reconstruction model
        masked_kspace: Input k-space
        mask: Sampling mask
        method: "mc_dropout", "tta", or "gradient"
        num_samples: Number of samples for MC methods

    Returns:
        Uncertainty map (B, 1, H, W)
    """
    if method == "mc_dropout":
        _, uncertainty = monte_carlo_uncertainty(model, masked_kspace, mask, num_samples)
    elif method == "tta":
        _, uncertainty = test_time_augmentation_uncertainty(model, masked_kspace, mask)
    elif method == "gradient":
        uncertainty = gradient_based_uncertainty(model, masked_kspace, mask)
    else:
        raise ValueError(f"Unknown method: {method}")

    return uncertainty


def gradient_based_uncertainty(
    model: nn.Module,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Gradient-based uncertainty estimation.

    Measures sensitivity of output to input perturbations.
    High sensitivity = high uncertainty.

    Args:
        model: Reconstruction model
        masked_kspace: Input k-space
        mask: Sampling mask

    Returns:
        Uncertainty map
    """
    masked_kspace_grad = masked_kspace.clone().requires_grad_(True)

    result = model(masked_kspace_grad, mask)
    output = result['output'] if isinstance(result, dict) else result

    # Compute gradient of output w.r.t. input
    grad = torch.autograd.grad(
        outputs=output.sum(),
        inputs=masked_kspace_grad,
        create_graph=False
    )[0]

    # Uncertainty = gradient magnitude
    uncertainty = torch.sqrt(grad[:, 0:1]**2 + grad[:, 1:2]**2)

    return uncertainty


class UncertaintyCalibrator(nn.Module):
    """
    Calibrates uncertainty estimates to be more reliable.

    Learns a mapping from raw uncertainty to calibrated probability.
    """

    def __init__(self):
        super().__init__()

        self.calibration_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, raw_uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Calibrate uncertainty.

        Args:
            raw_uncertainty: Raw uncertainty map (B, 1, H, W)

        Returns:
            Calibrated uncertainty (B, 1, H, W)
        """
        return self.calibration_net(raw_uncertainty)


def compute_uncertainty_weighted_discrepancy(
    discrepancy: torch.Tensor,
    uncertainty: torch.Tensor,
    mode: str = "multiply"
) -> torch.Tensor:
    """
    Combine discrepancy with uncertainty.

    High discrepancy in low-uncertainty regions is more suspicious
    than high discrepancy in high-uncertainty regions.

    Args:
        discrepancy: Discrepancy map (B, 1, H, W)
        uncertainty: Uncertainty map (B, 1, H, W)
        mode: "multiply", "divide", or "adaptive"

    Returns:
        Weighted discrepancy map
    """
    # Normalize both to [0, 1]
    disc_norm = (discrepancy - discrepancy.min()) / (discrepancy.max() - discrepancy.min() + 1e-8)
    unc_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)

    if mode == "multiply":
        # High discrepancy + high uncertainty = high weight
        return disc_norm * (1 + unc_norm)

    elif mode == "divide":
        # High discrepancy + low uncertainty = high weight
        return disc_norm / (unc_norm + 0.1)

    elif mode == "adaptive":
        # Learn the relationship
        # Use product for high-uncertainty suspicious regions
        # Use division for low-uncertainty suspicious regions
        alpha = torch.sigmoid(unc_norm - 0.5)
        weighted = alpha * disc_norm * (1 + unc_norm) + (1 - alpha) * disc_norm / (unc_norm + 0.1)
        return weighted

    else:
        raise ValueError(f"Unknown mode: {mode}")
