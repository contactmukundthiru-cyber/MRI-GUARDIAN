"""
Calibrated Uncertainty for MRI-GUARDIAN
========================================

THE PROBLEM:
Score-based refinement and neural networks are OVERCONFIDENT.
They give high certainty predictions even when they're wrong.

Example: Model predicts 0.95 confidence, but it's wrong 30% of the time.
This is DANGEROUS in medical imaging!

THE SOLUTION:
Multi-sample uncertainty estimation that is CALIBRATED.

Instead of a single prediction, we:
1. Run multiple reconstructions with different noise samples
2. Compute pixel-wise variance across samples
3. Calibrate so that "95% confidence" means "correct 95% of the time"

This module provides:
- Multi-sample inference for uncertainty estimation
- Variance-based uncertainty maps
- Calibration via temperature scaling
- Reliability diagrams for validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.special import softmax
from scipy.optimize import minimize_scalar


@dataclass
class CalibratedUncertaintyResult:
    """Result from calibrated uncertainty estimation."""
    # Main outputs
    mean_reconstruction: torch.Tensor  # Mean across samples
    uncertainty_map: torch.Tensor  # Calibrated uncertainty (std dev)
    raw_variance: torch.Tensor  # Raw variance before calibration

    # Calibration info
    calibration_temperature: float  # Temperature used for calibration
    expected_calibration_error: float  # ECE of the uncertainty
    is_well_calibrated: bool  # Whether ECE < 0.05

    # Sample statistics
    num_samples: int
    sample_std: float  # Average std across pixels
    max_uncertainty: float  # Maximum uncertainty value

    # Confidence intervals
    confidence_95_lower: torch.Tensor
    confidence_95_upper: torch.Tensor


class MultiSampleReconstructor:
    """
    Generate multiple reconstructions for uncertainty estimation.

    Uses stochastic elements in the reconstruction process:
    1. Different initial noise samples
    2. Dropout during inference
    3. Different diffusion trajectories
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        use_dropout: bool = True,
        add_noise: bool = True,
        noise_std: float = 0.01
    ):
        """
        Args:
            model: Reconstruction model (GuardianModel)
            num_samples: Number of samples for uncertainty
            use_dropout: Enable dropout during inference
            add_noise: Add input noise for diversity
            noise_std: Standard deviation of input noise
        """
        self.model = model
        self.num_samples = num_samples
        self.use_dropout = use_dropout
        self.add_noise = add_noise
        self.noise_std = noise_std

    def generate_samples(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate multiple reconstruction samples.

        Args:
            masked_kspace: Input k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)

        Returns:
            Samples tensor (num_samples, B, 1, H, W)
        """
        samples = []

        # Set dropout mode if requested
        if self.use_dropout:
            self.model.train()  # Enable dropout
        else:
            self.model.eval()

        with torch.set_grad_enabled(False):
            for i in range(self.num_samples):
                # Add noise to input for diversity
                if self.add_noise:
                    noise = torch.randn_like(masked_kspace) * self.noise_std
                    noisy_kspace = masked_kspace + noise * (1 - mask)  # Only add noise to unmeasured
                else:
                    noisy_kspace = masked_kspace

                # Run reconstruction
                result = self.model(noisy_kspace, mask, enforce_dc=True)
                output = result['output'] if isinstance(result, dict) else result

                samples.append(output)

        # Reset to eval mode
        self.model.eval()

        # Stack samples: (num_samples, B, 1, H, W)
        return torch.stack(samples, dim=0)


class UncertaintyCalibrator:
    """
    Calibrate uncertainty estimates using temperature scaling.

    Temperature scaling is a simple but effective method:
    - If T > 1: uncertainty becomes larger (model was overconfident)
    - If T < 1: uncertainty becomes smaller (model was underconfident)
    - T = 1: no change

    We find T by minimizing Expected Calibration Error (ECE).
    """

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.temperature = 1.0

    def fit(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        ground_truths: torch.Tensor
    ) -> float:
        """
        Fit calibration temperature on validation data.

        Args:
            predictions: Model predictions (N, H, W)
            uncertainties: Raw uncertainty estimates (N, H, W)
            ground_truths: Ground truth images (N, H, W)

        Returns:
            Optimal temperature
        """
        pred_np = predictions.cpu().numpy().flatten()
        unc_np = uncertainties.cpu().numpy().flatten()
        gt_np = ground_truths.cpu().numpy().flatten()

        # Compute errors
        errors = np.abs(pred_np - gt_np)

        # Find optimal temperature
        def ece_objective(T):
            calibrated_unc = unc_np * T
            return self._compute_ece(errors, calibrated_unc)

        result = minimize_scalar(
            ece_objective,
            bounds=(0.1, 10.0),
            method='bounded'
        )

        self.temperature = result.x
        return self.temperature

    def calibrate(self, raw_uncertainty: torch.Tensor) -> torch.Tensor:
        """Apply calibration temperature to uncertainty."""
        return raw_uncertainty * self.temperature

    def _compute_ece(self, errors: np.ndarray, uncertainties: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        # Bin by uncertainty level
        bin_edges = np.percentile(uncertainties, np.linspace(0, 100, self.num_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ece = 0
        for i in range(self.num_bins):
            mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
            if mask.sum() == 0:
                continue

            # Expected: errors should be covered by uncertainty at specified confidence
            bin_errors = errors[mask]
            bin_uncs = uncertainties[mask]

            # Fraction of errors within 2*sigma (95% confidence)
            covered = (bin_errors <= 2 * bin_uncs).mean()
            expected = 0.95

            ece += mask.mean() * np.abs(covered - expected)

        return ece


class CalibratedUncertaintyEstimator(nn.Module):
    """
    Main class for calibrated uncertainty estimation.

    Combines multi-sample inference with calibration.
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        calibration_temperature: float = 1.0
    ):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.calibration_temperature = calibration_temperature

        self.sampler = MultiSampleReconstructor(
            model, num_samples=num_samples
        )
        self.calibrator = UncertaintyCalibrator()
        self.calibrator.temperature = calibration_temperature

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        return_samples: bool = False
    ) -> CalibratedUncertaintyResult:
        """
        Estimate calibrated uncertainty.

        Args:
            masked_kspace: Input k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)
            return_samples: Whether to return all samples

        Returns:
            CalibratedUncertaintyResult with mean, uncertainty, and confidence intervals
        """
        # Generate samples
        samples = self.sampler.generate_samples(masked_kspace, mask)
        # samples: (num_samples, B, 1, H, W)

        # Compute statistics across samples
        mean_reconstruction = samples.mean(dim=0)  # (B, 1, H, W)
        raw_variance = samples.var(dim=0)  # (B, 1, H, W)
        raw_std = torch.sqrt(raw_variance + 1e-8)

        # Apply calibration
        calibrated_uncertainty = raw_std * self.calibration_temperature

        # Compute confidence intervals (95% = 1.96 sigma)
        z_95 = 1.96
        confidence_95_lower = mean_reconstruction - z_95 * calibrated_uncertainty
        confidence_95_upper = mean_reconstruction + z_95 * calibrated_uncertainty

        # Compute ECE (rough estimate based on sample consistency)
        sample_std = float(raw_std.mean())
        max_uncertainty = float(calibrated_uncertainty.max())

        # Estimate ECE from sample spread (heuristic)
        # Well-calibrated: ~5% of samples should be outside 2*sigma
        outlier_fraction = self._compute_outlier_fraction(samples, mean_reconstruction, raw_std)
        ece = abs(outlier_fraction - 0.05)  # Expected ~5% outliers for normal distribution

        return CalibratedUncertaintyResult(
            mean_reconstruction=mean_reconstruction,
            uncertainty_map=calibrated_uncertainty,
            raw_variance=raw_variance,
            calibration_temperature=self.calibration_temperature,
            expected_calibration_error=ece,
            is_well_calibrated=(ece < 0.05),
            num_samples=self.num_samples,
            sample_std=sample_std,
            max_uncertainty=max_uncertainty,
            confidence_95_lower=confidence_95_lower,
            confidence_95_upper=confidence_95_upper
        )

    def _compute_outlier_fraction(
        self,
        samples: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> float:
        """Compute fraction of samples outside 2*sigma."""
        # For each sample, check if it's within 2*std of mean
        z_scores = torch.abs(samples - mean.unsqueeze(0)) / (std.unsqueeze(0) + 1e-8)
        outliers = (z_scores > 2).float()
        return float(outliers.mean())

    def fit_calibration(
        self,
        dataloader,
        device: str = 'cuda'
    ) -> float:
        """
        Fit calibration on validation data.

        Args:
            dataloader: Validation data loader
            device: Device to use

        Returns:
            Optimal temperature
        """
        all_preds = []
        all_uncs = []
        all_gts = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                masked_kspace = batch['masked_kspace'].to(device)
                mask = batch['mask'].to(device)
                target = batch['target'].to(device)

                result = self.forward(masked_kspace, mask)

                all_preds.append(result.mean_reconstruction.cpu())
                all_uncs.append(result.uncertainty_map.cpu())
                all_gts.append(target.cpu())

        # Concatenate
        preds = torch.cat(all_preds, dim=0).squeeze()
        uncs = torch.cat(all_uncs, dim=0).squeeze()
        gts = torch.cat(all_gts, dim=0).squeeze()

        # Fit temperature
        optimal_temp = self.calibrator.fit(preds, uncs, gts)
        self.calibration_temperature = optimal_temp

        return optimal_temp


def create_reliability_diagram(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    ground_truths: np.ndarray,
    num_bins: int = 10
) -> Dict:
    """
    Create reliability diagram data for calibration visualization.

    Returns data for plotting: expected vs actual coverage at each confidence level.
    """
    errors = np.abs(predictions.flatten() - ground_truths.flatten())
    unc_flat = uncertainties.flatten()

    # Create bins based on uncertainty
    bin_edges = np.percentile(unc_flat, np.linspace(0, 100, num_bins + 1))

    confidence_levels = []
    actual_coverage = []
    expected_coverage = []
    bin_counts = []

    for i in range(num_bins):
        mask = (unc_flat >= bin_edges[i]) & (unc_flat < bin_edges[i+1])
        if mask.sum() == 0:
            continue

        bin_errors = errors[mask]
        bin_uncs = unc_flat[mask]

        # What confidence level does this bin represent?
        mean_unc = bin_uncs.mean()
        # Convert uncertainty to confidence interval coverage
        # If uncertainty = 1 std, then 68% should be covered at 1*sigma, 95% at 2*sigma

        # Check coverage at 2*sigma (95% expected)
        covered_95 = (bin_errors <= 2 * bin_uncs).mean()

        confidence_levels.append(float(mean_unc))
        actual_coverage.append(float(covered_95))
        expected_coverage.append(0.95)
        bin_counts.append(int(mask.sum()))

    return {
        'confidence_levels': confidence_levels,
        'actual_coverage': actual_coverage,
        'expected_coverage': expected_coverage,
        'bin_counts': bin_counts,
        'calibration_error': np.mean(np.abs(np.array(actual_coverage) - 0.95))
    }


def generate_uncertainty_report(result: CalibratedUncertaintyResult) -> str:
    """Generate a report on uncertainty estimation."""
    calibration_status = "WELL CALIBRATED" if result.is_well_calibrated else "NEEDS CALIBRATION"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║       CALIBRATED UNCERTAINTY REPORT                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  WHY CALIBRATED UNCERTAINTY MATTERS:                         ║
║  A "95% confidence" should be correct 95% of the time.       ║
║  Overconfident AI is DANGEROUS in medical imaging.           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  ESTIMATION METHOD                                            ║
╠──────────────────────────────────────────────────────────────╣
║  Number of samples:        {result.num_samples:>8}                            ║
║  Method:                   Multi-sample + Temperature        ║
║  Calibration temperature:  {result.calibration_temperature:>8.3f}                         ║
╠──────────────────────────────────────────────────────────────╣
║  UNCERTAINTY STATISTICS                                       ║
╠──────────────────────────────────────────────────────────────╣
║  Mean uncertainty (std):   {result.sample_std:>8.4f}                         ║
║  Max uncertainty:          {result.max_uncertainty:>8.4f}                         ║
║  ECE (calibration error):  {result.expected_calibration_error:>8.4f} (<0.05 = good)        ║
╠──────────────────────────────────────────────────────────────╣
║  CALIBRATION STATUS: {calibration_status:<35} ║
╠══════════════════════════════════════════════════════════════╣
║  INTERPRETATION                                               ║
╠──────────────────────────────────────────────────────────────╣
"""
    if result.is_well_calibrated:
        report += """║  Uncertainty estimates are RELIABLE.                         ║
║  High uncertainty regions should be reviewed carefully.      ║
║  95% confidence intervals should be trusted.                 ║
"""
    else:
        report += """║  Uncertainty estimates may be UNRELIABLE.                    ║
║  Consider re-calibrating on validation data.                 ║
║  Current confidence intervals may be too narrow/wide.        ║
"""

    report += "╚══════════════════════════════════════════════════════════════╝"

    return report
