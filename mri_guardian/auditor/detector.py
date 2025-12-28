"""
Hallucination Detection Module

The CORE of MRI-GUARDIAN's auditing capability.

SCIENTIFIC QUESTION:
===================
"Can we detect when a black-box MRI reconstruction system
is hallucinating structures not supported by the measured data?"

APPROACH:
=========
1. Use Guardian (physics-guided) reconstruction as reference
2. Compare black-box output against Guardian
3. Identify regions where they disagree significantly
4. Flag these as potential hallucinations

WHY THIS WORKS:
==============
- Guardian is constrained by physics (data consistency)
- Guardian can't "make up" k-space values that contradict measurements
- If black-box shows something Guardian doesn't, it might be hallucinated
- If Guardian shows something black-box doesn't, black-box might be missing it

DETECTION MODES:
===============
1. Intensity discrepancy: |I_blackbox - I_guardian|
2. Structural discrepancy: Edge/gradient differences
3. Frequency discrepancy: High-frequency content differences
4. Combined: Weighted combination with learned/optimized weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .discrepancy import (
    compute_intensity_discrepancy,
    compute_structural_discrepancy,
    compute_frequency_discrepancy,
    combine_discrepancy_maps,
)
from .uncertainty import compute_reconstruction_uncertainty


@dataclass
class DetectionResult:
    """Container for hallucination detection results."""
    # Discrepancy maps
    intensity_map: torch.Tensor  # |blackbox - guardian|
    structural_map: torch.Tensor  # Gradient-based discrepancy
    frequency_map: torch.Tensor  # High-frequency differences
    combined_map: torch.Tensor  # Weighted combination

    # Detection
    detection_mask: torch.Tensor  # Binary detection mask
    confidence_map: torch.Tensor  # Confidence scores

    # Statistics
    threshold: float
    num_detected_pixels: int
    detection_rate: float  # Fraction of image flagged

    # Optional uncertainty
    uncertainty_map: Optional[torch.Tensor] = None


class HallucinationDetector(nn.Module):
    """
    Main hallucination detection class.

    Uses Guardian reconstruction to audit black-box outputs.

    USAGE:
    ======
    detector = HallucinationDetector(guardian_model)

    # Given black-box reconstruction and measured k-space
    result = detector.detect(
        blackbox_recon,
        masked_kspace,
        mask
    )

    # Visualize
    plot_detection(blackbox_recon, result.detection_mask)
    """

    def __init__(
        self,
        guardian_model: nn.Module,
        detection_threshold: float = 0.15,
        use_uncertainty: bool = True,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            guardian_model: Trained Guardian reconstruction model
            detection_threshold: Threshold for flagging hallucinations
            use_uncertainty: Include uncertainty estimation
            weights: Weights for combining discrepancy types
                Default: {'intensity': 0.5, 'structural': 0.3, 'frequency': 0.2}
        """
        super().__init__()

        self.guardian = guardian_model
        self.threshold = detection_threshold
        self.use_uncertainty = use_uncertainty

        # Default weights for combining discrepancy maps
        if weights is None:
            weights = {
                'intensity': 0.5,
                'structural': 0.3,
                'frequency': 0.2
            }
        self.weights = weights

        # Optional learned threshold refinement
        self.threshold_net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),  # 4 = combined + 3 individual maps
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    @torch.no_grad()
    def get_guardian_reconstruction(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get Guardian reconstruction for comparison.

        Args:
            masked_kspace: Undersampled k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)

        Returns:
            Guardian reconstruction (B, 1, H, W)
        """
        self.guardian.eval()
        result = self.guardian(masked_kspace, mask)
        return result['output']

    def compute_discrepancy_maps(
        self,
        blackbox_recon: torch.Tensor,
        guardian_recon: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all types of discrepancy maps.

        Args:
            blackbox_recon: Black-box reconstruction (B, 1, H, W)
            guardian_recon: Guardian reconstruction (B, 1, H, W)

        Returns:
            Dict with intensity, structural, frequency, and combined maps
        """
        # Normalize inputs to same scale
        bb_norm = (blackbox_recon - blackbox_recon.mean()) / (blackbox_recon.std() + 1e-8)
        guard_norm = (guardian_recon - guardian_recon.mean()) / (guardian_recon.std() + 1e-8)

        # Compute individual discrepancy maps
        intensity = compute_intensity_discrepancy(bb_norm, guard_norm)
        structural = compute_structural_discrepancy(bb_norm, guard_norm)
        frequency = compute_frequency_discrepancy(bb_norm, guard_norm)

        # Normalize each map to [0, 1]
        intensity = self._normalize_map(intensity)
        structural = self._normalize_map(structural)
        frequency = self._normalize_map(frequency)

        # Combine maps
        combined = combine_discrepancy_maps(
            intensity, structural, frequency,
            self.weights['intensity'],
            self.weights['structural'],
            self.weights['frequency']
        )

        return {
            'intensity': intensity,
            'structural': structural,
            'frequency': frequency,
            'combined': combined
        }

    def _normalize_map(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize map to [0, 1] range."""
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    def detect_hallucinations(
        self,
        discrepancy_maps: Dict[str, torch.Tensor],
        threshold: Optional[float] = None,
        use_learned_threshold: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect hallucinations from discrepancy maps.

        Args:
            discrepancy_maps: Dict of discrepancy maps
            threshold: Detection threshold (uses self.threshold if None)
            use_learned_threshold: Use learned thresholding network

        Returns:
            detection_mask: Binary mask of detected regions
            confidence_map: Confidence scores [0, 1]
        """
        if threshold is None:
            threshold = self.threshold

        combined = discrepancy_maps['combined']

        if use_learned_threshold:
            # Use learned threshold network
            maps_concat = torch.cat([
                combined,
                discrepancy_maps['intensity'],
                discrepancy_maps['structural'],
                discrepancy_maps['frequency']
            ], dim=1)
            confidence_map = self.threshold_net(maps_concat)
        else:
            # Simple thresholding with soft edges
            confidence_map = torch.sigmoid((combined - threshold) / 0.05)

        # Binary detection mask
        detection_mask = (confidence_map > 0.5).float()

        return detection_mask, confidence_map

    def detect(
        self,
        blackbox_recon: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Full hallucination detection pipeline.

        Args:
            blackbox_recon: Black-box reconstruction (B, 1, H, W)
            masked_kspace: Measured k-space (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W)
            threshold: Detection threshold

        Returns:
            DetectionResult with all detection information
        """
        # Get Guardian reconstruction
        guardian_recon = self.get_guardian_reconstruction(masked_kspace, mask)

        # Compute discrepancy maps
        disc_maps = self.compute_discrepancy_maps(blackbox_recon, guardian_recon)

        # Detect hallucinations
        detection_mask, confidence_map = self.detect_hallucinations(
            disc_maps, threshold
        )

        # Optionally compute uncertainty
        uncertainty_map = None
        if self.use_uncertainty:
            uncertainty_map = compute_reconstruction_uncertainty(
                self.guardian, masked_kspace, mask
            )

        # Statistics
        num_detected = int(detection_mask.sum().item())
        total_pixels = detection_mask.numel()
        detection_rate = num_detected / total_pixels

        return DetectionResult(
            intensity_map=disc_maps['intensity'],
            structural_map=disc_maps['structural'],
            frequency_map=disc_maps['frequency'],
            combined_map=disc_maps['combined'],
            detection_mask=detection_mask,
            confidence_map=confidence_map,
            threshold=threshold or self.threshold,
            num_detected_pixels=num_detected,
            detection_rate=detection_rate,
            uncertainty_map=uncertainty_map
        )

    def evaluate_detection(
        self,
        detection_mask: torch.Tensor,
        ground_truth_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate detection performance against ground truth.

        Args:
            detection_mask: Predicted hallucination mask (B, 1, H, W)
            ground_truth_mask: True hallucination mask (B, 1, H, W)

        Returns:
            Dict with precision, recall, F1, IoU
        """
        # Flatten
        pred = detection_mask.flatten()
        true = ground_truth_mask.flatten()

        # True positives, false positives, false negatives
        tp = ((pred == 1) & (true == 1)).sum().float()
        fp = ((pred == 1) & (true == 0)).sum().float()
        fn = ((pred == 0) & (true == 1)).sum().float()
        tn = ((pred == 0) & (true == 0)).sum().float()

        # Metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item(),
            'accuracy': accuracy.item()
        }


class BaselineDetector:
    """
    Baseline hallucination detector for comparison.

    Uses simple methods without physics-guided reference.
    """

    @staticmethod
    def detect_zf_difference(
        blackbox_recon: torch.Tensor,
        zf_recon: torch.Tensor,
        threshold: float = 0.2
    ) -> torch.Tensor:
        """
        Baseline: Compare to zero-filled reconstruction.

        If black-box is very different from zero-filled,
        it might be hallucinating.

        Args:
            blackbox_recon: Black-box output (B, 1, H, W)
            zf_recon: Zero-filled reconstruction (B, 1, H, W)
            threshold: Detection threshold

        Returns:
            Detection mask
        """
        diff = torch.abs(blackbox_recon - zf_recon)
        diff_norm = diff / (diff.max() + 1e-8)
        return (diff_norm > threshold).float()

    @staticmethod
    def detect_edge_anomaly(
        image: torch.Tensor,
        threshold: float = 0.3
    ) -> torch.Tensor:
        """
        Baseline: Detect unusually strong edges.

        Hallucinations often have artificially sharp edges.

        Args:
            image: Image (B, 1, H, W)
            threshold: Detection threshold

        Returns:
            Detection mask
        """
        # Sobel filters
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

        # Compute gradients
        gx = F.conv2d(image, sobel_x, padding=1)
        gy = F.conv2d(image, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(gx**2 + gy**2)

        # Normalize
        edge_norm = edge_magnitude / (edge_magnitude.max() + 1e-8)

        return (edge_norm > threshold).float()

    @staticmethod
    def detect_local_variance_anomaly(
        image: torch.Tensor,
        kernel_size: int = 5,
        threshold: float = 0.25
    ) -> torch.Tensor:
        """
        Baseline: Detect regions with unusual local variance.

        Hallucinated textures often have different statistics.

        Args:
            image: Image (B, 1, H, W)
            kernel_size: Local window size
            threshold: Detection threshold

        Returns:
            Detection mask
        """
        # Local mean
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device)
        kernel = kernel / kernel.numel()

        local_mean = F.conv2d(image, kernel, padding=kernel_size//2)
        local_sq_mean = F.conv2d(image**2, kernel, padding=kernel_size//2)
        local_var = local_sq_mean - local_mean**2

        # Normalize
        var_norm = local_var / (local_var.max() + 1e-8)

        # Detect unusually high OR low variance
        global_var_mean = var_norm.mean()
        anomaly = torch.abs(var_norm - global_var_mean)

        return (anomaly > threshold).float()


class AuditorEnsemble(nn.Module):
    """
    Ensemble of multiple detection methods.

    Combines Guardian-based detection with baselines for
    more robust hallucination detection.
    """

    def __init__(
        self,
        guardian_model: nn.Module,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            guardian_model: Trained Guardian model
            weights: Weights for each detection method
        """
        super().__init__()

        self.guardian_detector = HallucinationDetector(guardian_model)
        self.baseline = BaselineDetector()

        if weights is None:
            weights = {
                'guardian': 0.6,
                'zf_diff': 0.15,
                'edge': 0.15,
                'variance': 0.1
            }
        self.weights = weights

    def detect(
        self,
        blackbox_recon: torch.Tensor,
        zf_recon: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Ensemble detection using multiple methods.

        Args:
            blackbox_recon: Black-box output
            zf_recon: Zero-filled reconstruction
            masked_kspace: Measured k-space
            mask: Sampling mask
            threshold: Final detection threshold

        Returns:
            Dict with individual and combined detections
        """
        # Guardian-based detection
        guardian_result = self.guardian_detector.detect(
            blackbox_recon, masked_kspace, mask
        )

        # Baseline detections
        zf_det = self.baseline.detect_zf_difference(blackbox_recon, zf_recon)
        edge_det = self.baseline.detect_edge_anomaly(blackbox_recon)
        var_det = self.baseline.detect_local_variance_anomaly(blackbox_recon)

        # Weighted combination
        combined = (
            self.weights['guardian'] * guardian_result.confidence_map +
            self.weights['zf_diff'] * zf_det +
            self.weights['edge'] * edge_det +
            self.weights['variance'] * var_det
        )

        # Final detection
        final_mask = (combined > threshold).float()

        return {
            'guardian_confidence': guardian_result.confidence_map,
            'guardian_mask': guardian_result.detection_mask,
            'zf_diff': zf_det,
            'edge': edge_det,
            'variance': var_det,
            'combined_confidence': combined,
            'final_mask': final_mask
        }
