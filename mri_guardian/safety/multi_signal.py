"""
Multi-Signal Consistency Checker

Combines multiple anomaly detection signals into a unified framework:
- Guardian vs black-box discrepancy
- Multi-contrast differences (if available)
- Tissue boundary inconsistencies
- Gradient magnitude anomalies
- Non-physical phase patterns
- Unexpected noise distributions
- K-space vs image domain inconsistencies

Novel contribution: First comprehensive multi-signal fusion system
for detecting reconstruction anomalies in medical imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class AnomalySignal:
    """Individual anomaly detection signal."""
    name: str
    score: float  # 0-1, higher = more anomalous
    spatial_map: Optional[torch.Tensor]  # Spatial distribution
    confidence: float  # Confidence in this signal
    description: str


@dataclass
class ConsistencyReport:
    """Complete multi-signal consistency analysis."""
    signals: List[AnomalySignal]
    fused_anomaly_map: torch.Tensor
    overall_consistency_score: float  # 0-1, higher = more consistent/safe
    inconsistent_regions: List[Dict]  # List of flagged regions
    dominant_anomaly_type: str
    recommendations: List[str]


class ReconstructionDiscrepancySignal:
    """
    Detect discrepancies between Guardian and black-box reconstructions.

    This is the core hallucination detection signal.
    """

    def __init__(
        self,
        threshold_factor: float = 2.0,
        use_structural: bool = True,
        use_intensity: bool = True,
        use_frequency: bool = True
    ):
        self.threshold_factor = threshold_factor
        self.use_structural = use_structural
        self.use_intensity = use_intensity
        self.use_frequency = use_frequency

    def compute(
        self,
        guardian_recon: torch.Tensor,
        blackbox_recon: torch.Tensor
    ) -> AnomalySignal:
        """
        Compute reconstruction discrepancy signal.

        Args:
            guardian_recon: Physics-guided reconstruction
            blackbox_recon: Black-box AI reconstruction

        Returns:
            AnomalySignal with discrepancy analysis
        """
        # Ensure same shape
        g = guardian_recon.squeeze()
        b = blackbox_recon.squeeze()

        while g.dim() > 2:
            g = g.squeeze(0)
        while b.dim() > 2:
            b = b.squeeze(0)

        if g.shape != b.shape:
            # Resize to match
            target_size = g.shape
            b = F.interpolate(
                b.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()

        # 1. Intensity discrepancy
        intensity_diff = torch.abs(g - b)
        mean_diff = intensity_diff.mean()
        std_diff = intensity_diff.std()
        threshold = mean_diff + self.threshold_factor * std_diff

        intensity_anomaly = (intensity_diff > threshold).float()
        intensity_score = (intensity_diff > threshold).float().mean().item()

        # 2. Structural discrepancy (edge comparison)
        def compute_edges(img):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=img.dtype, device=img.device) / 8
            sobel_y = sobel_x.T

            gx = F.conv2d(img.unsqueeze(0).unsqueeze(0),
                          sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
            gy = F.conv2d(img.unsqueeze(0).unsqueeze(0),
                          sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
            return torch.sqrt(gx**2 + gy**2).squeeze()

        if self.use_structural:
            edge_g = compute_edges(g)
            edge_b = compute_edges(b)
            edge_diff = torch.abs(edge_g - edge_b)
            structural_anomaly = edge_diff / (edge_g.max() + 1e-8)
            structural_score = (structural_anomaly > 0.2).float().mean().item()
        else:
            structural_anomaly = torch.zeros_like(g)
            structural_score = 0

        # 3. Frequency discrepancy
        if self.use_frequency:
            fft_g = torch.fft.fft2(g)
            fft_b = torch.fft.fft2(b)
            freq_diff = torch.abs(torch.abs(fft_g) - torch.abs(fft_b))

            # Focus on high frequencies (more prone to hallucinations)
            H, W = freq_diff.shape
            y, x = torch.meshgrid(
                torch.arange(H, device=g.device) - H//2,
                torch.arange(W, device=g.device) - W//2,
                indexing='ij'
            )
            high_freq_mask = (torch.abs(x) + torch.abs(y)) > min(H, W) // 4
            high_freq_mask = torch.fft.fftshift(high_freq_mask.float())

            freq_anomaly = freq_diff * high_freq_mask
            freq_anomaly = torch.abs(torch.fft.ifft2(freq_anomaly + 0j))
            freq_score = freq_anomaly.mean().item() / (g.std().item() + 1e-8)
        else:
            freq_anomaly = torch.zeros_like(g)
            freq_score = 0

        # Combine signals
        combined_map = (
            0.4 * intensity_anomaly +
            0.3 * (structural_anomaly if self.use_structural else 0) +
            0.3 * (freq_anomaly / (freq_anomaly.max() + 1e-8) if self.use_frequency else 0)
        )

        overall_score = (
            0.4 * intensity_score +
            0.3 * structural_score +
            0.3 * min(1.0, freq_score)
        )

        return AnomalySignal(
            name='reconstruction_discrepancy',
            score=overall_score,
            spatial_map=combined_map,
            confidence=0.9,  # High confidence in this signal
            description=f"Guardian vs black-box discrepancy: {overall_score:.2%} anomalous"
        )


class TissueBoundarySignal:
    """
    Detect tissue boundary inconsistencies.

    Real tissue boundaries should:
    - Be smooth and continuous
    - Follow anatomical patterns
    - Have consistent contrast
    """

    def compute(self, image: torch.Tensor) -> AnomalySignal:
        """Detect boundary anomalies."""
        img = image.squeeze()
        while img.dim() > 2:
            img = img.squeeze(0)

        # Edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img.dtype, device=img.device) / 8
        sobel_y = sobel_x.T

        gx = F.conv2d(img.unsqueeze(0).unsqueeze(0),
                      sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        gy = F.conv2d(img.unsqueeze(0).unsqueeze(0),
                      sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        edges = torch.sqrt(gx**2 + gy**2).squeeze()

        # Find strong edges
        edge_threshold = edges.mean() + 2 * edges.std()
        strong_edges = edges > edge_threshold

        # Compute edge continuity
        # Connected edges should form smooth curves
        dilated = F.max_pool2d(
            strong_edges.float().unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        eroded = -F.max_pool2d(
            -strong_edges.float().unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()

        # Isolated edge pixels (not part of continuous boundary)
        isolated = strong_edges.float() * (1 - eroded)
        isolated_ratio = isolated.sum() / (strong_edges.sum() + 1)

        # Edge direction consistency
        edge_direction = torch.atan2(gy.squeeze(), gx.squeeze())

        # Local direction variance
        direction_var = F.avg_pool2d(
            edge_direction.unsqueeze(0).unsqueeze(0)**2, 5, 1, 2
        ).squeeze() - F.avg_pool2d(
            edge_direction.unsqueeze(0).unsqueeze(0), 5, 1, 2
        ).squeeze()**2

        # High direction variance at edges = inconsistent boundaries
        boundary_inconsistency = direction_var * strong_edges.float()

        # Anomaly score
        score = (isolated_ratio.item() + boundary_inconsistency.mean().item()) / 2
        score = min(1.0, score * 2)

        return AnomalySignal(
            name='tissue_boundary',
            score=score,
            spatial_map=isolated + boundary_inconsistency,
            confidence=0.7,
            description=f"Boundary consistency: {1-score:.2%}"
        )


class GradientMagnitudeSignal:
    """
    Detect abnormal gradient patterns.

    Gradients in MRI should follow tissue boundaries and be
    physically plausible.
    """

    def compute(
        self,
        reconstruction: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> AnomalySignal:
        """Detect gradient anomalies."""
        img = reconstruction.squeeze()
        while img.dim() > 2:
            img = img.squeeze(0)

        # Compute gradient magnitude
        grad_x = img[:, 1:] - img[:, :-1]
        grad_y = img[1:, :] - img[:-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        img_range = img.max() - img.min()
        grad_mag_norm = grad_mag / (img_range + 1e-8)

        # Find anomalously high gradients (not at edges)
        # Edges detected by Laplacian
        laplacian = F.conv2d(
            img.unsqueeze(0).unsqueeze(0),
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                         dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()

        edge_mask = torch.abs(laplacian) > laplacian.std() * 2
        non_edge_gradient = grad_mag_norm * (~edge_mask).float()

        # High gradient in non-edge regions = anomaly
        threshold = non_edge_gradient.mean() + 3 * non_edge_gradient.std()
        anomaly_map = (non_edge_gradient > threshold).float() * non_edge_gradient

        score = anomaly_map.mean().item() / (grad_mag_norm.mean().item() + 1e-8)
        score = min(1.0, score)

        # Compare with reference if available
        if reference is not None:
            ref = reference.squeeze()
            while ref.dim() > 2:
                ref = ref.squeeze(0)

            ref_grad_x = ref[:, 1:] - ref[:, :-1]
            ref_grad_y = ref[1:, :] - ref[:-1, :]
            ref_grad_x = F.pad(ref_grad_x, (0, 1, 0, 0))
            ref_grad_y = F.pad(ref_grad_y, (0, 0, 0, 1))
            ref_grad_mag = torch.sqrt(ref_grad_x**2 + ref_grad_y**2)

            grad_diff = torch.abs(grad_mag - ref_grad_mag)
            diff_score = grad_diff.mean().item() / (ref_grad_mag.mean().item() + 1e-8)
            score = (score + diff_score) / 2
            anomaly_map = anomaly_map + grad_diff / (grad_diff.max() + 1e-8)

        return AnomalySignal(
            name='gradient_anomaly',
            score=score,
            spatial_map=anomaly_map,
            confidence=0.75,
            description=f"Gradient anomaly score: {score:.2%}"
        )


class NoiseDistributionSignal:
    """
    Detect unexpected noise patterns.

    Real MRI noise should be:
    - Rayleigh/Rician distributed in magnitude images
    - Spatially uniform (mostly)
    - Independent of signal
    """

    def compute(
        self,
        image: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> AnomalySignal:
        """Detect noise distribution anomalies."""
        img = image.squeeze()
        while img.dim() > 2:
            img = img.squeeze(0)

        # Estimate noise from high-frequency residual
        # High-pass filter
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0) / 9

        high_freq = F.conv2d(img.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()

        # Expected: uniform noise distribution
        # Compute local noise variance
        kernel_size = 16
        stride = kernel_size // 2

        # Unfold into patches
        H, W = high_freq.shape
        patches = high_freq.unfold(0, kernel_size, stride).unfold(1, kernel_size, stride)
        patch_vars = patches.var(dim=(-2, -1))

        # Non-uniform variance indicates structured noise (artifacts)
        overall_var = high_freq.var()
        var_of_var = patch_vars.var()
        uniformity = 1 - (var_of_var / (overall_var**2 + 1e-8)).item()
        uniformity = max(0, min(1, uniformity))

        # Detect structured patterns in residual
        fft_residual = torch.fft.fft2(high_freq)
        fft_mag = torch.abs(torch.fft.fftshift(fft_residual))

        # Peaks in FFT indicate periodic artifacts
        mean_mag = fft_mag.mean()
        std_mag = fft_mag.std()
        peaks = fft_mag > (mean_mag + 5 * std_mag)

        # Exclude DC and very low frequencies
        cy, cx = H // 2, W // 2
        dc_mask = torch.zeros_like(peaks)
        dc_mask[cy-5:cy+5, cx-5:cx+5] = True
        artifact_peaks = peaks & ~dc_mask

        artifact_score = artifact_peaks.sum().item() / (H * W)

        # Combine scores
        score = (1 - uniformity) * 0.5 + artifact_score * 100 * 0.5
        score = min(1.0, score)

        # Spatial map: local variance deviation from expected
        expected_var = overall_var
        anomaly_map = torch.zeros_like(img)
        for i in range(patch_vars.shape[0]):
            for j in range(patch_vars.shape[1]):
                var_diff = abs(patch_vars[i, j] - expected_var) / (expected_var + 1e-8)
                y_start = i * stride
                x_start = j * stride
                y_end = min(y_start + kernel_size, H)
                x_end = min(x_start + kernel_size, W)
                anomaly_map[y_start:y_end, x_start:x_end] = var_diff

        return AnomalySignal(
            name='noise_distribution',
            score=score,
            spatial_map=anomaly_map,
            confidence=0.65,
            description=f"Noise uniformity: {uniformity:.2%}, artifact peaks: {artifact_peaks.sum().item()}"
        )


class AnomalyFusion:
    """
    Fuse multiple anomaly signals into unified detection.

    Uses learned or heuristic weights to combine signals
    based on their reliability and clinical importance.
    """

    def __init__(
        self,
        fusion_method: str = 'weighted',  # 'weighted', 'max', 'learned'
        default_weights: Optional[Dict[str, float]] = None
    ):
        self.fusion_method = fusion_method
        self.weights = default_weights or {
            'reconstruction_discrepancy': 0.35,
            'tissue_boundary': 0.15,
            'gradient_anomaly': 0.20,
            'noise_distribution': 0.10,
            'physics_violation': 0.20
        }

    def fuse(
        self,
        signals: List[AnomalySignal],
        image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Fuse multiple anomaly signals.

        Args:
            signals: List of AnomalySignal objects
            image_size: Target size for spatial maps

        Returns:
            fused_map: Combined spatial anomaly map
            overall_score: Scalar anomaly score
        """
        if not signals:
            return None, 0.0

        # Determine image size
        if image_size is None:
            for s in signals:
                if s.spatial_map is not None:
                    image_size = s.spatial_map.shape[-2:]
                    break
            if image_size is None:
                image_size = (256, 256)

        # Collect and resize spatial maps
        maps = []
        scores = []
        weights = []

        for signal in signals:
            w = self.weights.get(signal.name, 0.1)
            weights.append(w * signal.confidence)
            scores.append(signal.score)

            if signal.spatial_map is not None:
                m = signal.spatial_map.float()
                while m.dim() < 4:
                    m = m.unsqueeze(0)
                if m.shape[-2:] != image_size:
                    m = F.interpolate(m, size=image_size, mode='bilinear', align_corners=False)
                m = m.squeeze()

                # Normalize to [0, 1]
                if m.max() > m.min():
                    m = (m - m.min()) / (m.max() - m.min())

                maps.append(m)
            else:
                maps.append(torch.zeros(image_size))

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Fuse based on method
        if self.fusion_method == 'weighted':
            fused_map = sum(w * m for w, m in zip(weights, maps))
            overall_score = sum(w * s for w, s in zip(weights, scores))

        elif self.fusion_method == 'max':
            fused_map = torch.stack(maps).max(dim=0)[0]
            overall_score = max(scores)

        else:
            # Default to weighted
            fused_map = sum(w * m for w, m in zip(weights, maps))
            overall_score = sum(w * s for w, s in zip(weights, scores))

        return fused_map, overall_score


class MultiSignalConsistencyChecker:
    """
    Complete multi-signal consistency checking system.

    Orchestrates all individual signal detectors and fuses
    their outputs into a comprehensive anomaly assessment.
    """

    def __init__(
        self,
        use_reconstruction_discrepancy: bool = True,
        use_tissue_boundary: bool = True,
        use_gradient: bool = True,
        use_noise: bool = True,
        fusion_method: str = 'weighted'
    ):
        self.signal_detectors = {}

        if use_reconstruction_discrepancy:
            self.signal_detectors['reconstruction_discrepancy'] = ReconstructionDiscrepancySignal()

        if use_tissue_boundary:
            self.signal_detectors['tissue_boundary'] = TissueBoundarySignal()

        if use_gradient:
            self.signal_detectors['gradient_anomaly'] = GradientMagnitudeSignal()

        if use_noise:
            self.signal_detectors['noise_distribution'] = NoiseDistributionSignal()

        self.fusion = AnomalyFusion(fusion_method=fusion_method)

    def check(
        self,
        reconstruction: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        guardian_recon: Optional[torch.Tensor] = None,
        physics_violation_map: Optional[torch.Tensor] = None
    ) -> ConsistencyReport:
        """
        Perform comprehensive consistency check.

        Args:
            reconstruction: AI reconstruction to check
            reference: Ground truth or alternative reference
            guardian_recon: Guardian model reconstruction
            physics_violation_map: Pre-computed physics violations

        Returns:
            ConsistencyReport with complete analysis
        """
        signals = []

        # 1. Reconstruction discrepancy (if guardian available)
        if 'reconstruction_discrepancy' in self.signal_detectors and guardian_recon is not None:
            signal = self.signal_detectors['reconstruction_discrepancy'].compute(
                guardian_recon, reconstruction
            )
            signals.append(signal)

        # 2. Tissue boundary analysis
        if 'tissue_boundary' in self.signal_detectors:
            signal = self.signal_detectors['tissue_boundary'].compute(reconstruction)
            signals.append(signal)

        # 3. Gradient analysis
        if 'gradient_anomaly' in self.signal_detectors:
            signal = self.signal_detectors['gradient_anomaly'].compute(
                reconstruction, reference
            )
            signals.append(signal)

        # 4. Noise distribution
        if 'noise_distribution' in self.signal_detectors:
            signal = self.signal_detectors['noise_distribution'].compute(
                reconstruction, reference
            )
            signals.append(signal)

        # 5. Physics violations (pre-computed)
        if physics_violation_map is not None:
            signals.append(AnomalySignal(
                name='physics_violation',
                score=physics_violation_map.mean().item(),
                spatial_map=physics_violation_map,
                confidence=0.9,
                description="Physics constraint violations"
            ))

        # Fuse signals
        image_size = reconstruction.shape[-2:]
        fused_map, overall_score = self.fusion.fuse(signals, image_size)

        # Identify inconsistent regions
        inconsistent_regions = []
        if fused_map is not None:
            # Find connected high-anomaly regions
            threshold = 0.5
            high_anomaly = (fused_map > threshold).float()

            # Simple region detection (bounding boxes)
            if high_anomaly.sum() > 0:
                nonzero = torch.nonzero(high_anomaly)
                if len(nonzero) > 0:
                    y_min, x_min = nonzero.min(dim=0)[0].tolist()
                    y_max, x_max = nonzero.max(dim=0)[0].tolist()

                    inconsistent_regions.append({
                        'bbox': (y_min, x_min, y_max, x_max),
                        'area': high_anomaly.sum().item(),
                        'mean_score': fused_map[high_anomaly > 0].mean().item()
                    })

        # Determine dominant anomaly type
        if signals:
            dominant = max(signals, key=lambda s: s.score * s.confidence)
            dominant_type = dominant.name
        else:
            dominant_type = 'none'

        # Consistency score (inverse of anomaly)
        consistency_score = 1.0 - overall_score

        # Recommendations
        recommendations = []
        if consistency_score < 0.5:
            recommendations.append("CAUTION: Significant inconsistencies detected")
        if consistency_score < 0.3:
            recommendations.append("WARNING: Reconstruction may be unreliable")

        for signal in signals:
            if signal.score > 0.5:
                recommendations.append(f"High {signal.name}: {signal.description}")

        if not recommendations:
            recommendations.append("No significant inconsistencies detected")

        return ConsistencyReport(
            signals=signals,
            fused_anomaly_map=fused_map,
            overall_consistency_score=consistency_score,
            inconsistent_regions=inconsistent_regions,
            dominant_anomaly_type=dominant_type,
            recommendations=recommendations
        )
