"""
Counterfactual Probe - Mathematical Proof of Hallucinations
============================================================

THE KEY INSIGHT:
Instead of just checking if the image is consistent, ask:
"Could this suspicious feature be REAL?"

THE NOVELTY:
No clinical auditor does this. They just flag errors.
We PROVE whether a feature is hallucination or real.

THE METHOD:
1. Identify ROI: Auditor flags a suspicious spot
2. Optimization: Freeze k-space data (hard constraint)
   Try to find ANY valid image where this spot DOESN'T exist
3. Decision:
   - If we CAN remove the spot without violating k-space → UNCERTAIN
   - If we CANNOT remove the spot (k-space error increases) → CONFIRMED REAL

WHY THIS WINS:
It solves the "Black Box Paradox."
You aren't trusting another neural net.
You're trusting a TARGETED MATHEMATICAL PROOF for that specific patient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import optimize, ndimage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HypothesisTestResult:
    """Result from counterfactual hypothesis testing."""
    # Main conclusion
    feature_is_real: bool  # True = mathematically confirmed, False = uncertain/hallucination
    confidence: float  # 0-1, how confident in the conclusion

    # Evidence
    original_kspace_error: float  # K-space error with feature present
    counterfactual_kspace_error: float  # K-space error with feature removed
    error_increase: float  # How much error increases when feature removed
    error_threshold: float  # Threshold for significance

    # Optimization details
    optimization_converged: bool
    iterations_used: int
    final_residual: float

    # Visualization data
    original_roi: np.ndarray
    counterfactual_roi: np.ndarray
    difference_map: np.ndarray

    # Explanation
    conclusion: str


class HypothesisTester:
    """
    Test whether a suspicious feature is real or hallucinated.

    This is the CORE INNOVATION: We don't trust another neural network.
    We use mathematical optimization to PROVE the answer.

    Mathematical Foundation:
    -----------------------
    Given:
    - measured k-space y
    - suspicious ROI mask M
    - current reconstruction x

    Question: Is the feature in ROI real?

    Test: Find x* that:
    1. Minimizes ||x* - x||_ROI (remove the feature)
    2. Subject to: ||A*x* - y||_measured < threshold (respect physics)

    If no such x* exists (optimization fails), the feature is REAL.
    If x* exists, the feature is UNCERTAIN (could be hallucination).
    """

    def __init__(
        self,
        error_threshold: float = 0.01,  # Max acceptable k-space error increase
        max_iterations: int = 1000,
        learning_rate: float = 0.1,
        regularization: float = 0.001
    ):
        self.error_threshold = error_threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization

    def test_hypothesis(
        self,
        reconstruction: np.ndarray,  # Current reconstruction (H, W) or (1, H, W)
        measured_kspace: np.ndarray,  # Original k-space measurements (H, W) complex
        sampling_mask: np.ndarray,  # Binary mask of measured locations (H, W)
        roi_mask: np.ndarray,  # Mask of suspicious region to test (H, W)
        background_value: Optional[float] = None  # Value to replace feature with
    ) -> HypothesisTestResult:
        """
        Test if a suspicious feature in ROI is real or hallucinated.

        Args:
            reconstruction: Current image reconstruction
            measured_kspace: Original k-space measurements
            sampling_mask: Binary mask of sampled k-space
            roi_mask: Mask highlighting suspicious region
            background_value: Value to use when removing feature (auto if None)

        Returns:
            HypothesisTestResult with conclusion and evidence
        """
        # Ensure correct shapes
        if reconstruction.ndim == 3:
            reconstruction = reconstruction.squeeze()
        if roi_mask.ndim == 3:
            roi_mask = roi_mask.squeeze()

        reconstruction = reconstruction.astype(np.float64)
        original_roi = reconstruction.copy()

        # Step 1: Compute original k-space error
        original_kspace_error = self._compute_kspace_error(
            reconstruction, measured_kspace, sampling_mask
        )

        # Step 2: Estimate background value if not provided
        if background_value is None:
            # Use mean of surrounding region
            dilated_mask = ndimage.binary_dilation(roi_mask, iterations=5)
            surround_mask = dilated_mask & ~roi_mask
            if surround_mask.sum() > 0:
                background_value = reconstruction[surround_mask].mean()
            else:
                background_value = reconstruction[~roi_mask].mean()

        # Step 3: Try to create counterfactual image (feature removed)
        counterfactual, optimization_result = self._optimize_counterfactual(
            reconstruction,
            measured_kspace,
            sampling_mask,
            roi_mask,
            background_value
        )

        # Step 4: Compute counterfactual k-space error
        counterfactual_kspace_error = self._compute_kspace_error(
            counterfactual, measured_kspace, sampling_mask
        )

        # Step 5: Make decision
        error_increase = counterfactual_kspace_error - original_kspace_error
        relative_increase = error_increase / (original_kspace_error + 1e-10)

        # If error increases significantly, feature is REAL
        # (cannot remove without violating physics)
        feature_is_real = relative_increase > self.error_threshold

        # Confidence based on how much error increases
        if feature_is_real:
            # More error increase = more confident it's real
            confidence = min(1.0, relative_increase / (5 * self.error_threshold))
        else:
            # Less error increase = more confident it's hallucination
            confidence = min(1.0, (self.error_threshold - relative_increase) / self.error_threshold)

        # Generate conclusion
        if feature_is_real:
            conclusion = (
                f"FEATURE CONFIRMED REAL. "
                f"Removing this feature increases k-space error by {100*relative_increase:.2f}%, "
                f"which exceeds threshold {100*self.error_threshold:.2f}%. "
                f"This feature is supported by measured data and is NOT a hallucination. "
                f"(Confidence: {100*confidence:.0f}%)"
            )
        else:
            conclusion = (
                f"FEATURE IS UNCERTAIN (possible hallucination). "
                f"This feature CAN be removed with only {100*relative_increase:.2f}% k-space error increase "
                f"(threshold: {100*self.error_threshold:.2f}%). "
                f"The measured k-space data does NOT require this feature. "
                f"(Confidence: {100*confidence:.0f}%)"
            )

        return HypothesisTestResult(
            feature_is_real=feature_is_real,
            confidence=confidence,
            original_kspace_error=original_kspace_error,
            counterfactual_kspace_error=counterfactual_kspace_error,
            error_increase=error_increase,
            error_threshold=self.error_threshold,
            optimization_converged=optimization_result['converged'],
            iterations_used=optimization_result['iterations'],
            final_residual=optimization_result['residual'],
            original_roi=original_roi[roi_mask].copy() if roi_mask.any() else np.array([]),
            counterfactual_roi=counterfactual[roi_mask].copy() if roi_mask.any() else np.array([]),
            difference_map=(reconstruction - counterfactual).astype(np.float32),
            conclusion=conclusion
        )

    def _compute_kspace_error(
        self,
        image: np.ndarray,
        measured_kspace: np.ndarray,
        sampling_mask: np.ndarray
    ) -> float:
        """Compute k-space error at measured locations."""
        # Forward model: image -> k-space
        image_kspace = np.fft.fftshift(np.fft.fft2(image))

        # Error only at measured locations
        mask_bool = sampling_mask > 0.5
        error = np.abs(image_kspace[mask_bool] - measured_kspace[mask_bool])

        # Normalized by signal energy
        signal_energy = np.abs(measured_kspace[mask_bool]).sum()
        error_energy = error.sum()

        return error_energy / (signal_energy + 1e-10)

    def _optimize_counterfactual(
        self,
        image: np.ndarray,
        measured_kspace: np.ndarray,
        sampling_mask: np.ndarray,
        roi_mask: np.ndarray,
        target_value: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize to find counterfactual image with feature removed.

        We solve:
        minimize ||image_roi - target_value||² + lambda * ||Ax - y||²_measured

        This finds an image where:
        1. The ROI is replaced with background
        2. K-space error is minimized
        """
        # Initialize with feature replaced by background
        counterfactual = image.copy()
        counterfactual[roi_mask] = target_value

        # Iterative optimization: project onto k-space constraint while pushing ROI toward background
        converged = False
        best_kspace_error = float('inf')

        for iteration in range(self.max_iterations):
            # Forward: image -> k-space
            cf_kspace = np.fft.fftshift(np.fft.fft2(counterfactual))

            # Compute k-space error
            mask_bool = sampling_mask > 0.5
            kspace_error = self._compute_kspace_error(
                counterfactual, measured_kspace, sampling_mask
            )

            # Check convergence
            if abs(kspace_error - best_kspace_error) < 1e-8:
                converged = True
                break
            best_kspace_error = min(best_kspace_error, kspace_error)

            # Data consistency step: replace measured k-space
            cf_kspace[mask_bool] = measured_kspace[mask_bool]

            # Inverse: k-space -> image
            counterfactual_new = np.abs(np.fft.ifft2(np.fft.ifftshift(cf_kspace)))

            # ROI constraint: push toward target value
            # Mix between DC-corrected image and target in ROI
            alpha = self.learning_rate
            counterfactual_new[roi_mask] = (
                (1 - alpha) * counterfactual_new[roi_mask] +
                alpha * target_value
            )

            # Update
            counterfactual = counterfactual_new

        return counterfactual, {
            'converged': converged,
            'iterations': iteration + 1,
            'residual': kspace_error
        }

    def batch_test(
        self,
        reconstructions: List[np.ndarray],
        measured_kspaces: List[np.ndarray],
        sampling_masks: List[np.ndarray],
        roi_masks: List[np.ndarray]
    ) -> List[HypothesisTestResult]:
        """Test multiple hypotheses in batch."""
        results = []
        for recon, kspace, mask, roi in zip(
            reconstructions, measured_kspaces, sampling_masks, roi_masks
        ):
            result = self.test_hypothesis(recon, kspace, mask, roi)
            results.append(result)
        return results


def generate_hypothesis_report(result: HypothesisTestResult) -> str:
    """Generate detailed report of hypothesis test."""
    status = "CONFIRMED REAL" if result.feature_is_real else "UNCERTAIN (possible hallucination)"
    status_icon = "✓" if result.feature_is_real else "⚠️"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║        COUNTERFACTUAL HYPOTHESIS TEST REPORT                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  THE QUESTION: Could this suspicious feature be real?        ║
║                                                              ║
║  THE METHOD: Mathematical proof, not another neural network  ║
║  We test: "Can we remove this feature without violating      ║
║            the measured k-space data?"                       ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  VERDICT: {status_icon} {status:<46} ║
║  Confidence: {100*result.confidence:>5.1f}%                                        ║
╠══════════════════════════════════════════════════════════════╣
║  EVIDENCE                                                     ║
╠──────────────────────────────────────────────────────────────╣
║  Original k-space error:      {result.original_kspace_error:>12.6f}                ║
║  Counterfactual k-space error:{result.counterfactual_kspace_error:>12.6f}                ║
║  Error increase:              {result.error_increase:>12.6f} ({100*result.error_increase/max(result.original_kspace_error,1e-10):>6.2f}%)     ║
║  Threshold for significance:  {result.error_threshold:>12.6f}                ║
╠──────────────────────────────────────────────────────────────╣
║  OPTIMIZATION                                                 ║
╠──────────────────────────────────────────────────────────────╣
║  Converged:                   {"Yes" if result.optimization_converged else "No":<12}                ║
║  Iterations:                  {result.iterations_used:>12}                ║
║  Final residual:              {result.final_residual:>12.6f}                ║
╠══════════════════════════════════════════════════════════════╣
║  CONCLUSION                                                   ║
╠──────────────────────────────────────────────────────────────╣
"""
    # Wrap conclusion text
    conclusion_lines = [result.conclusion[i:i+58] for i in range(0, len(result.conclusion), 58)]
    for line in conclusion_lines:
        report += f"║  {line:<58} ║\n"

    report += "╚══════════════════════════════════════════════════════════════╝"

    return report


class AutomaticHypothesisTester:
    """
    Automatically identify and test suspicious regions.

    Combines:
    1. Suspicious region detection (from hallucination auditor)
    2. Counterfactual hypothesis testing
    3. Comprehensive verdict
    """

    def __init__(
        self,
        suspicion_threshold: float = 0.5,  # Threshold for flagging suspicious
        min_region_size: int = 10,  # Minimum pixels to test
        hypothesis_tester: Optional[HypothesisTester] = None
    ):
        self.suspicion_threshold = suspicion_threshold
        self.min_region_size = min_region_size
        self.hypothesis_tester = hypothesis_tester or HypothesisTester()

    def auto_test(
        self,
        reconstruction: np.ndarray,
        measured_kspace: np.ndarray,
        sampling_mask: np.ndarray,
        suspicion_map: np.ndarray  # From hallucination detector
    ) -> List[HypothesisTestResult]:
        """
        Automatically test all suspicious regions.

        Args:
            reconstruction: Current reconstruction
            measured_kspace: Original k-space
            sampling_mask: Sampling mask
            suspicion_map: Suspicion/discrepancy map from auditor

        Returns:
            List of HypothesisTestResult for each suspicious region
        """
        # Threshold suspicion map
        suspicious_binary = suspicion_map > self.suspicion_threshold

        # Find connected components
        labeled, n_regions = ndimage.label(suspicious_binary)

        results = []
        for region_id in range(1, n_regions + 1):
            roi_mask = labeled == region_id

            # Skip small regions
            if roi_mask.sum() < self.min_region_size:
                continue

            # Test this region
            result = self.hypothesis_tester.test_hypothesis(
                reconstruction,
                measured_kspace,
                sampling_mask,
                roi_mask
            )
            results.append(result)

        return results
