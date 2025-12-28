"""
Uncertainty Estimation Module for MRI Reconstruction

Implements multiple uncertainty quantification methods:
1. MC Dropout - Monte Carlo dropout sampling
2. Deep Ensembles - Multiple model predictions
3. Evidential Deep Learning - Direct uncertainty prediction
4. Test-Time Augmentation - Augmentation-based uncertainty

Novel contribution: Combines multiple uncertainty sources into
a unified uncertainty field that predicts clinical risk zones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class UncertaintyOutput:
    """Container for uncertainty estimation outputs."""
    mean: torch.Tensor  # Mean prediction
    aleatoric: torch.Tensor  # Data uncertainty (irreducible)
    epistemic: torch.Tensor  # Model uncertainty (reducible)
    total: torch.Tensor  # Combined uncertainty
    confidence: torch.Tensor  # 1 - normalized_uncertainty
    samples: Optional[torch.Tensor] = None  # Raw samples if available


class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimation."""

    @abstractmethod
    def estimate(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> UncertaintyOutput:
        """Estimate uncertainty for given inputs."""
        pass

    @staticmethod
    def compute_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute pixel-wise entropy from probability distribution."""
        return -torch.sum(probs * torch.log(probs + eps), dim=0)

    @staticmethod
    def normalize_uncertainty(uncertainty: torch.Tensor) -> torch.Tensor:
        """Normalize uncertainty to [0, 1] range."""
        u_min = uncertainty.min()
        u_max = uncertainty.max()
        if u_max - u_min < 1e-8:
            return torch.zeros_like(uncertainty)
        return (uncertainty - u_min) / (u_max - u_min)


class MCDropoutEstimator(UncertaintyEstimator):
    """
    Monte Carlo Dropout Uncertainty Estimation.

    Performs multiple forward passes with dropout enabled to estimate
    epistemic (model) uncertainty through sampling.

    Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
    """

    def __init__(
        self,
        num_samples: int = 20,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            num_samples: Number of MC dropout samples
            dropout_rate: Dropout probability (if not already in model)
        """
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

    def _enable_dropout(self, model: nn.Module):
        """Enable dropout layers during inference."""
        for module in model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train()

    def _add_dropout_hooks(self, model: nn.Module) -> List:
        """Add dropout to model if not present."""
        hooks = []

        def dropout_hook(module, input, output):
            return F.dropout2d(output, p=self.dropout_rate, training=True)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Add dropout after convolutions
                if 'down' in name or 'encoder' in name:
                    hook = module.register_forward_hook(dropout_hook)
                    hooks.append(hook)

        return hooks

    def estimate(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> UncertaintyOutput:
        """
        Estimate uncertainty using MC Dropout.

        Args:
            model: Neural network model
            inputs: Dictionary with 'masked_kspace' and 'mask'

        Returns:
            UncertaintyOutput with mean, aleatoric, epistemic uncertainty
        """
        masked_kspace = inputs['masked_kspace']
        mask = inputs['mask']
        device = masked_kspace.device

        # Enable dropout
        model.train()
        self._enable_dropout(model)

        # Collect samples
        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                result = model(masked_kspace, mask)
                output = result['output'] if isinstance(result, dict) else result
                samples.append(output)

        # Stack samples: [num_samples, B, C, H, W]
        samples = torch.stack(samples, dim=0)

        # Compute statistics
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        # Epistemic uncertainty (model uncertainty) - variance of predictions
        epistemic = variance

        # Aleatoric uncertainty - estimated from mean prediction residual
        # In practice, this requires a separate head; approximate here
        aleatoric = torch.zeros_like(epistemic)

        # Total uncertainty
        total = epistemic + aleatoric

        # Confidence (inverse of normalized uncertainty)
        confidence = 1.0 - self.normalize_uncertainty(total)

        # Restore eval mode
        model.eval()

        return UncertaintyOutput(
            mean=mean,
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            confidence=confidence,
            samples=samples
        )


class EnsembleEstimator(UncertaintyEstimator):
    """
    Deep Ensemble Uncertainty Estimation.

    Uses multiple independently trained models to estimate uncertainty
    through disagreement between ensemble members.

    Reference: Lakshminarayanan et al., "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles", NeurIPS 2017
    """

    def __init__(self, models: Optional[List[nn.Module]] = None):
        """
        Args:
            models: List of ensemble member models
        """
        self.models = models or []

    def add_model(self, model: nn.Module):
        """Add a model to the ensemble."""
        self.models.append(model)

    def estimate(
        self,
        model: nn.Module,  # Ignored if self.models is populated
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> UncertaintyOutput:
        """
        Estimate uncertainty using ensemble disagreement.
        """
        masked_kspace = inputs['masked_kspace']
        mask = inputs['mask']

        models_to_use = self.models if self.models else [model]

        if len(models_to_use) < 2:
            # Fall back to MC Dropout with single model
            mc_estimator = MCDropoutEstimator(num_samples=10)
            return mc_estimator.estimate(model, inputs)

        # Collect predictions from each model
        predictions = []
        with torch.no_grad():
            for m in models_to_use:
                m.eval()
                result = m(masked_kspace, mask)
                output = result['output'] if isinstance(result, dict) else result
                predictions.append(output)

        # Stack predictions: [num_models, B, C, H, W]
        predictions = torch.stack(predictions, dim=0)

        # Compute statistics
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        # Epistemic uncertainty from ensemble disagreement
        epistemic = variance

        # Pairwise disagreement (more robust)
        num_models = len(models_to_use)
        pairwise_diff = torch.zeros_like(mean)
        count = 0
        for i in range(num_models):
            for j in range(i + 1, num_models):
                pairwise_diff += torch.abs(predictions[i] - predictions[j])
                count += 1
        pairwise_diff /= count

        # Combine variance and pairwise disagreement
        epistemic = 0.5 * variance + 0.5 * pairwise_diff ** 2

        aleatoric = torch.zeros_like(epistemic)
        total = epistemic + aleatoric
        confidence = 1.0 - self.normalize_uncertainty(total)

        return UncertaintyOutput(
            mean=mean,
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            confidence=confidence,
            samples=predictions
        )


class EvidentialHead(nn.Module):
    """
    Evidential output head for direct uncertainty prediction.

    Predicts parameters of a Normal-Inverse-Gamma distribution
    to model both aleatoric and epistemic uncertainty.

    Reference: Amini et al., "Deep Evidential Regression", NeurIPS 2020
    """

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()

        # Predict 4 parameters: gamma (mean), nu (evidence for mean),
        # alpha (evidence for variance), beta (scale)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 4 * out_channels, 1)
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass predicting evidential parameters.

        Returns:
            Dictionary with gamma, nu, alpha, beta parameters
        """
        params = self.conv(x)

        # Split into 4 parameter groups
        gamma, log_nu, log_alpha, log_beta = torch.chunk(params, 4, dim=1)

        # Apply softplus to ensure positivity
        nu = F.softplus(log_nu) + 1e-6
        alpha = F.softplus(log_alpha) + 1.0 + 1e-6  # alpha > 1 required
        beta = F.softplus(log_beta) + 1e-6

        return {
            'gamma': gamma,  # Predicted mean
            'nu': nu,  # Evidence for mean
            'alpha': alpha,  # Evidence for variance
            'beta': beta  # Scale parameter
        }

    @staticmethod
    def compute_uncertainty(
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute aleatoric and epistemic uncertainty from evidential parameters.

        Aleatoric = E[sigma^2] = beta / (alpha - 1)
        Epistemic = Var[mu] = beta / (nu * (alpha - 1))
        """
        nu = params['nu']
        alpha = params['alpha']
        beta = params['beta']

        # Aleatoric uncertainty (data/irreducible)
        aleatoric = beta / (alpha - 1)

        # Epistemic uncertainty (model/reducible)
        epistemic = beta / (nu * (alpha - 1))

        return aleatoric, epistemic


class EvidentialEstimator(UncertaintyEstimator):
    """
    Evidential Deep Learning Uncertainty Estimation.

    Uses a model with an evidential output head to directly
    predict both aleatoric and epistemic uncertainty in a single pass.

    This is more efficient than MC Dropout or Ensembles.
    """

    def __init__(self, evidential_head: Optional[EvidentialHead] = None):
        """
        Args:
            evidential_head: Pre-trained evidential head (optional)
        """
        self.evidential_head = evidential_head

    def estimate(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> UncertaintyOutput:
        """
        Estimate uncertainty using evidential deep learning.
        """
        masked_kspace = inputs['masked_kspace']
        mask = inputs['mask']

        model.eval()

        with torch.no_grad():
            result = model(masked_kspace, mask)

            # Check if model has evidential output
            if isinstance(result, dict) and 'evidential_params' in result:
                params = result['evidential_params']
                mean = params['gamma']
                aleatoric, epistemic = EvidentialHead.compute_uncertainty(params)
            else:
                # Fall back to feature-based uncertainty
                output = result['output'] if isinstance(result, dict) else result
                mean = output

                # Use gradient magnitude as proxy for uncertainty
                grad_x = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
                grad_y = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])

                # Pad to original size
                grad_x = F.pad(grad_x, (0, 1, 0, 0))
                grad_y = F.pad(grad_y, (0, 0, 0, 1))

                # High gradient = high potential for error
                gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

                # Local variance in small patches
                kernel_size = 5
                local_mean = F.avg_pool2d(
                    output, kernel_size, stride=1, padding=kernel_size // 2
                )
                local_sq_mean = F.avg_pool2d(
                    output ** 2, kernel_size, stride=1, padding=kernel_size // 2
                )
                local_var = local_sq_mean - local_mean ** 2
                local_var = F.relu(local_var)  # Numerical stability

                epistemic = gradient_magnitude * 0.5 + local_var * 0.5
                aleatoric = local_var

        total = aleatoric + epistemic
        confidence = 1.0 - self.normalize_uncertainty(total)

        return UncertaintyOutput(
            mean=mean,
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            confidence=confidence
        )


class TTAUncertaintyEstimator(UncertaintyEstimator):
    """
    Test-Time Augmentation (TTA) Uncertainty Estimation.

    Applies geometric transformations at test time and measures
    consistency of predictions to estimate uncertainty.

    Novel aspect: Uses MRI-specific augmentations that preserve
    k-space physics (phase shifts, rotations in k-space).
    """

    def __init__(
        self,
        num_augmentations: int = 8,
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        rotate_90: bool = True,
        intensity_jitter: float = 0.05
    ):
        self.num_augmentations = num_augmentations
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.rotate_90 = rotate_90
        self.intensity_jitter = intensity_jitter

    def _augment(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        aug_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, callable]:
        """
        Apply augmentation and return inverse transform.

        Returns:
            Augmented kspace, augmented mask, inverse transform function
        """
        if aug_idx == 0:
            # Identity
            return kspace, mask, lambda x: x

        elif aug_idx == 1 and self.flip_horizontal:
            # Horizontal flip in image domain = conjugate flip in k-space
            kspace_aug = torch.flip(kspace, dims=[-1])
            mask_aug = torch.flip(mask, dims=[-1])
            return kspace_aug, mask_aug, lambda x: torch.flip(x, dims=[-1])

        elif aug_idx == 2 and self.flip_vertical:
            # Vertical flip
            kspace_aug = torch.flip(kspace, dims=[-2])
            mask_aug = torch.flip(mask, dims=[-2])
            return kspace_aug, mask_aug, lambda x: torch.flip(x, dims=[-2])

        elif aug_idx == 3 and self.rotate_90:
            # 90 degree rotation
            kspace_aug = torch.rot90(kspace, 1, dims=[-2, -1])
            mask_aug = torch.rot90(mask, 1, dims=[-2, -1])
            return kspace_aug, mask_aug, lambda x: torch.rot90(x, -1, dims=[-2, -1])

        elif aug_idx == 4 and self.rotate_90:
            # 180 degree rotation
            kspace_aug = torch.rot90(kspace, 2, dims=[-2, -1])
            mask_aug = torch.rot90(mask, 2, dims=[-2, -1])
            return kspace_aug, mask_aug, lambda x: torch.rot90(x, -2, dims=[-2, -1])

        elif aug_idx == 5 and self.rotate_90:
            # 270 degree rotation
            kspace_aug = torch.rot90(kspace, 3, dims=[-2, -1])
            mask_aug = torch.rot90(mask, 3, dims=[-2, -1])
            return kspace_aug, mask_aug, lambda x: torch.rot90(x, -3, dims=[-2, -1])

        elif aug_idx == 6 and self.flip_horizontal and self.flip_vertical:
            # Both flips
            kspace_aug = torch.flip(torch.flip(kspace, dims=[-1]), dims=[-2])
            mask_aug = torch.flip(torch.flip(mask, dims=[-1]), dims=[-2])
            return kspace_aug, mask_aug, lambda x: torch.flip(torch.flip(x, dims=[-2]), dims=[-1])

        else:
            # Intensity jitter (in image domain, reflected back to kspace)
            if self.intensity_jitter > 0:
                jitter = 1.0 + (torch.rand(1, device=kspace.device) - 0.5) * 2 * self.intensity_jitter
                kspace_aug = kspace * jitter
                return kspace_aug, mask, lambda x: x / jitter

        return kspace, mask, lambda x: x

    def estimate(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> UncertaintyOutput:
        """
        Estimate uncertainty using test-time augmentation.
        """
        masked_kspace = inputs['masked_kspace']
        mask = inputs['mask']

        model.eval()

        predictions = []

        with torch.no_grad():
            for aug_idx in range(self.num_augmentations):
                # Apply augmentation
                kspace_aug, mask_aug, inverse_fn = self._augment(
                    masked_kspace, mask, aug_idx
                )

                # Forward pass
                result = model(kspace_aug, mask_aug)
                output = result['output'] if isinstance(result, dict) else result

                # Apply inverse transform
                output_original = inverse_fn(output)
                predictions.append(output_original)

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)

        # Compute statistics
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        # TTA uncertainty is primarily epistemic (captures model sensitivity)
        epistemic = variance

        # Consistency score - how much predictions agree
        max_pred = predictions.max(dim=0)[0]
        min_pred = predictions.min(dim=0)[0]
        consistency_range = max_pred - min_pred

        aleatoric = torch.zeros_like(epistemic)
        total = epistemic
        confidence = 1.0 - self.normalize_uncertainty(total)

        return UncertaintyOutput(
            mean=mean,
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            confidence=confidence,
            samples=predictions
        )


class CombinedUncertaintyEstimator(UncertaintyEstimator):
    """
    Combined uncertainty estimation using multiple methods.

    Novel contribution: Fuses multiple uncertainty sources with
    learned or heuristic weights to produce a unified uncertainty field.
    """

    def __init__(
        self,
        use_mc_dropout: bool = True,
        use_tta: bool = True,
        use_evidential: bool = False,
        mc_samples: int = 10,
        tta_augmentations: int = 4,
        fusion_method: str = 'max'  # 'max', 'mean', 'learned'
    ):
        self.estimators = []
        self.weights = []

        if use_mc_dropout:
            self.estimators.append(MCDropoutEstimator(num_samples=mc_samples))
            self.weights.append(1.0)

        if use_tta:
            self.estimators.append(TTAUncertaintyEstimator(num_augmentations=tta_augmentations))
            self.weights.append(1.0)

        if use_evidential:
            self.estimators.append(EvidentialEstimator())
            self.weights.append(1.0)

        self.fusion_method = fusion_method

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def estimate(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> UncertaintyOutput:
        """
        Estimate uncertainty by combining multiple methods.
        """
        if not self.estimators:
            raise ValueError("No uncertainty estimators configured")

        all_epistemic = []
        all_aleatoric = []
        all_means = []

        for estimator in self.estimators:
            result = estimator.estimate(model, inputs, **kwargs)
            all_epistemic.append(result.epistemic)
            all_aleatoric.append(result.aleatoric)
            all_means.append(result.mean)

        # Fuse uncertainties
        if self.fusion_method == 'max':
            epistemic = torch.stack(all_epistemic, dim=0).max(dim=0)[0]
            aleatoric = torch.stack(all_aleatoric, dim=0).max(dim=0)[0]
        elif self.fusion_method == 'mean':
            epistemic = torch.stack(all_epistemic, dim=0).mean(dim=0)
            aleatoric = torch.stack(all_aleatoric, dim=0).mean(dim=0)
        else:  # weighted
            epistemic = sum(w * u for w, u in zip(self.weights, all_epistemic))
            aleatoric = sum(w * u for w, u in zip(self.weights, all_aleatoric))

        # Average means
        mean = torch.stack(all_means, dim=0).mean(dim=0)

        total = aleatoric + epistemic
        confidence = 1.0 - self.normalize_uncertainty(total)

        return UncertaintyOutput(
            mean=mean,
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            confidence=confidence
        )
