"""
Distribution Shift Detection Module

Detects when an MRI scan is "unusual" compared to the training distribution.
Critical for AI safety as models may fail silently on out-of-distribution data.

Detects:
- Scanner/protocol differences
- Rare pathologies
- Unusual anatomy
- Different field strengths
- Motion artifacts
- Coil configurations

Novel contribution: First comprehensive distribution shift detector
specifically designed for MRI reconstruction pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.covariance import EmpiricalCovariance, MinCovDet


@dataclass
class DistributionShiftResult:
    """Container for distribution shift detection results."""
    is_ood: bool  # Out-of-distribution flag
    ood_score: float  # Continuous OOD score [0, 1]
    confidence: float  # Confidence in OOD detection
    shift_type: str  # Type of detected shift
    details: Dict  # Detailed breakdown


class FeatureExtractor(nn.Module):
    """
    Extract features for distribution shift detection.

    Uses multiple levels of abstraction:
    - Low-level: intensity statistics, gradients
    - Mid-level: texture features, frequency content
    - High-level: learned representations
    """

    def __init__(self, pretrained_encoder: Optional[nn.Module] = None):
        super().__init__()

        self.pretrained_encoder = pretrained_encoder

        # Learnable feature extractor if no pretrained model
        if pretrained_encoder is None:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256)
            )

    def extract_statistical_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract hand-crafted statistical features."""
        B = image.shape[0]
        features = []

        for b in range(B):
            img = image[b, 0]  # [H, W]

            # Intensity statistics
            mean_val = img.mean().item()
            std_val = img.std().item()
            min_val = img.min().item()
            max_val = img.max().item()

            # Percentiles
            flat = img.flatten()
            p10 = torch.quantile(flat, 0.1).item()
            p25 = torch.quantile(flat, 0.25).item()
            p50 = torch.quantile(flat, 0.5).item()
            p75 = torch.quantile(flat, 0.75).item()
            p90 = torch.quantile(flat, 0.9).item()

            # Gradient statistics
            grad_x = torch.abs(img[:, 1:] - img[:, :-1])
            grad_y = torch.abs(img[1:, :] - img[:-1, :])
            grad_mean = (grad_x.mean() + grad_y.mean()).item() / 2
            grad_std = (grad_x.std() + grad_y.std()).item() / 2

            # Frequency content (simplified)
            fft = torch.fft.fft2(img)
            fft_mag = torch.abs(fft)
            center_h, center_w = img.shape[0] // 2, img.shape[1] // 2

            # Low/mid/high frequency energy ratios
            low_freq_mask = torch.zeros_like(fft_mag)
            low_freq_mask[center_h-16:center_h+16, center_w-16:center_w+16] = 1
            low_freq_energy = (fft_mag * low_freq_mask).sum().item()
            total_energy = fft_mag.sum().item() + 1e-8
            low_freq_ratio = low_freq_energy / total_energy

            mid_freq_mask = torch.zeros_like(fft_mag)
            mid_freq_mask[center_h-64:center_h+64, center_w-64:center_w+64] = 1
            mid_freq_mask = mid_freq_mask - low_freq_mask
            mid_freq_energy = (fft_mag * mid_freq_mask).sum().item()
            mid_freq_ratio = mid_freq_energy / total_energy

            high_freq_ratio = 1.0 - low_freq_ratio - mid_freq_ratio

            # Entropy (texture measure)
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            hist = torch.histc(img_norm, bins=64, min=0, max=1)
            hist = hist / hist.sum()
            entropy = -torch.sum(hist * torch.log(hist + 1e-8)).item()

            # Laplacian (edge measure)
            laplacian_kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
            laplacian = F.conv2d(img.unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1)
            laplacian_var = laplacian.var().item()

            features.append([
                mean_val, std_val, min_val, max_val,
                p10, p25, p50, p75, p90,
                grad_mean, grad_std,
                low_freq_ratio, mid_freq_ratio, high_freq_ratio,
                entropy, laplacian_var
            ])

        return torch.tensor(features, dtype=image.dtype, device=image.device)

    def extract_deep_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract learned deep features."""
        if self.pretrained_encoder is not None:
            with torch.no_grad():
                return self.pretrained_encoder(image)
        else:
            return self.encoder(image)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract all features."""
        stat_features = self.extract_statistical_features(image)
        deep_features = self.extract_deep_features(image)

        return {
            'statistical': stat_features,
            'deep': deep_features,
            'combined': torch.cat([stat_features, deep_features], dim=1)
        }


class MahalanobisDetector:
    """
    Mahalanobis distance-based OOD detector.

    Computes distance from the training distribution using
    robust covariance estimation.
    """

    def __init__(self, use_robust: bool = True):
        self.use_robust = use_robust
        self.mean = None
        self.covariance = None
        self.inv_covariance = None
        self.threshold = None

    def fit(self, features: np.ndarray, contamination: float = 0.05):
        """
        Fit the detector to training features.

        Args:
            features: Training features [N, D]
            contamination: Expected proportion of outliers
        """
        if self.use_robust:
            # Robust covariance estimation
            cov_estimator = MinCovDet(support_fraction=1.0 - contamination)
            cov_estimator.fit(features)
            self.mean = cov_estimator.location_
            self.covariance = cov_estimator.covariance_
        else:
            cov_estimator = EmpiricalCovariance()
            cov_estimator.fit(features)
            self.mean = cov_estimator.location_
            self.covariance = cov_estimator.covariance_

        # Regularize covariance for numerical stability
        self.covariance += np.eye(self.covariance.shape[0]) * 1e-6
        self.inv_covariance = np.linalg.inv(self.covariance)

        # Compute threshold from training data
        train_distances = self.compute_distance(features)
        self.threshold = np.percentile(train_distances, 95)

    def compute_distance(self, features: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance from training distribution."""
        if self.mean is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        diff = features - self.mean
        distances = np.sqrt(np.sum(diff @ self.inv_covariance * diff, axis=1))
        return distances

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict OOD status.

        Returns:
            is_ood: Boolean array
            scores: Normalized OOD scores [0, 1]
        """
        distances = self.compute_distance(features)
        is_ood = distances > self.threshold
        scores = 1.0 / (1.0 + np.exp(-(distances - self.threshold) / (self.threshold * 0.5)))
        return is_ood, scores


class KSpaceDistributionAnalyzer:
    """
    Analyze k-space distribution for scanner/protocol shifts.

    Different scanners and protocols produce characteristic
    k-space patterns that can be detected.
    """

    def __init__(self):
        self.reference_stats = None

    def extract_kspace_features(self, kspace: torch.Tensor) -> Dict[str, float]:
        """Extract features from k-space data."""
        # Magnitude
        if kspace.is_complex():
            magnitude = torch.abs(kspace)
        else:
            # Assume last dim is real/imag
            magnitude = torch.sqrt(kspace[..., 0]**2 + kspace[..., 1]**2)

        # Remove batch dim if present
        while magnitude.dim() > 2:
            magnitude = magnitude.squeeze(0)

        H, W = magnitude.shape

        # Center
        center_h, center_w = H // 2, W // 2

        # Radial profile
        y, x = torch.meshgrid(
            torch.arange(H, device=magnitude.device) - center_h,
            torch.arange(W, device=magnitude.device) - center_w,
            indexing='ij'
        )
        radius = torch.sqrt(x.float()**2 + y.float()**2)

        # Energy in radial bands
        max_radius = min(center_h, center_w)
        num_bands = 10
        band_energies = []

        for i in range(num_bands):
            r_min = i * max_radius / num_bands
            r_max = (i + 1) * max_radius / num_bands
            mask = (radius >= r_min) & (radius < r_max)
            if mask.sum() > 0:
                band_energy = magnitude[mask].mean().item()
            else:
                band_energy = 0
            band_energies.append(band_energy)

        # Peak location and shape
        peak_val = magnitude.max().item()
        peak_idx = torch.argmax(magnitude)
        peak_y, peak_x = peak_idx // W, peak_idx % W

        # DC component strength
        dc_strength = magnitude[center_h, center_w].item()

        # Symmetry measure
        flipped = torch.flip(magnitude, dims=[0, 1])
        symmetry = 1.0 - (torch.abs(magnitude - flipped).mean() / (magnitude.mean() + 1e-8)).item()

        features = {
            'band_energies': band_energies,
            'peak_val': peak_val,
            'peak_offset': np.sqrt((peak_y - center_h)**2 + (peak_x - center_w)**2),
            'dc_strength': dc_strength,
            'symmetry': symmetry,
            'total_energy': magnitude.sum().item()
        }

        return features

    def fit(self, kspace_samples: List[torch.Tensor]):
        """Fit to training k-space samples."""
        all_features = []
        for ks in kspace_samples:
            features = self.extract_kspace_features(ks)
            all_features.append(features)

        # Compute reference statistics
        self.reference_stats = {}
        keys = all_features[0].keys()

        for key in keys:
            if key == 'band_energies':
                values = np.array([f[key] for f in all_features])
                self.reference_stats[key] = {
                    'mean': values.mean(axis=0),
                    'std': values.std(axis=0) + 1e-6
                }
            else:
                values = [f[key] for f in all_features]
                self.reference_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values) + 1e-6
                }

    def compute_shift_score(self, kspace: torch.Tensor) -> Tuple[float, Dict]:
        """Compute distribution shift score for new k-space."""
        if self.reference_stats is None:
            return 0.0, {}

        features = self.extract_kspace_features(kspace)
        z_scores = {}
        total_z = 0
        count = 0

        for key, value in features.items():
            ref = self.reference_stats[key]
            if key == 'band_energies':
                z = np.abs(np.array(value) - ref['mean']) / ref['std']
                z_scores[key] = z.mean()
            else:
                z = abs(value - ref['mean']) / ref['std']
                z_scores[key] = z

            total_z += z_scores[key] if isinstance(z_scores[key], float) else z_scores[key].mean()
            count += 1

        avg_z = total_z / count
        shift_score = 1.0 / (1.0 + np.exp(-avg_z + 2))  # Sigmoid centered at z=2

        return shift_score, z_scores


class DistributionShiftDetector:
    """
    Comprehensive distribution shift detector for MRI.

    Combines multiple detection strategies:
    1. Feature-space Mahalanobis distance
    2. K-space distribution analysis
    3. Intensity histogram comparison
    4. Anatomy structure analysis
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        device: str = 'cuda'
    ):
        self.device = device
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.feature_extractor = self.feature_extractor.to(device)

        self.mahalanobis_detector = MahalanobisDetector()
        self.kspace_analyzer = KSpaceDistributionAnalyzer()

        self.reference_histograms = None
        self.is_fitted = False

    def fit(
        self,
        images: List[torch.Tensor],
        kspace_samples: Optional[List[torch.Tensor]] = None
    ):
        """
        Fit detector to training distribution.

        Args:
            images: List of training images
            kspace_samples: Optional list of k-space data
        """
        # Extract features from all training images
        all_features = []
        for img in images:
            if img.dim() == 2:
                img = img.unsqueeze(0).unsqueeze(0)
            elif img.dim() == 3:
                img = img.unsqueeze(0)

            img = img.to(self.device)
            features = self.feature_extractor(img)
            all_features.append(features['combined'].cpu().numpy())

        all_features = np.vstack(all_features)

        # Fit Mahalanobis detector
        self.mahalanobis_detector.fit(all_features)

        # Fit k-space analyzer if data provided
        if kspace_samples:
            self.kspace_analyzer.fit(kspace_samples)

        # Compute reference histogram
        all_pixels = []
        for img in images:
            img_np = img.cpu().numpy().flatten()
            all_pixels.extend(img_np.tolist())

        hist, bin_edges = np.histogram(all_pixels, bins=100, density=True)
        self.reference_histograms = {
            'hist': hist,
            'bin_edges': bin_edges
        }

        self.is_fitted = True

    def detect(
        self,
        image: torch.Tensor,
        kspace: Optional[torch.Tensor] = None,
        return_details: bool = True
    ) -> DistributionShiftResult:
        """
        Detect distribution shift in input image.

        Args:
            image: Input image [B, 1, H, W] or [1, H, W] or [H, W]
            kspace: Optional k-space data
            return_details: Whether to return detailed breakdown

        Returns:
            DistributionShiftResult with OOD status and scores
        """
        if not self.is_fitted:
            # Return neutral result if not fitted
            return DistributionShiftResult(
                is_ood=False,
                ood_score=0.0,
                confidence=0.0,
                shift_type='unknown',
                details={'warning': 'Detector not fitted'}
            )

        # Ensure proper shape
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Feature-based detection
        with torch.no_grad():
            features = self.feature_extractor(image)
            feature_vec = features['combined'].cpu().numpy()

        is_ood_feature, feature_score = self.mahalanobis_detector.predict(feature_vec)
        feature_score = feature_score[0]

        # K-space detection
        kspace_score = 0.0
        kspace_details = {}
        if kspace is not None:
            kspace_score, kspace_details = self.kspace_analyzer.compute_shift_score(kspace)

        # Histogram comparison
        hist_score = 0.0
        if self.reference_histograms is not None:
            img_np = image.cpu().numpy().flatten()
            test_hist, _ = np.histogram(
                img_np,
                bins=self.reference_histograms['bin_edges'],
                density=True
            )
            # Jensen-Shannon divergence
            ref_hist = self.reference_histograms['hist']
            m = (ref_hist + test_hist) / 2
            js_div = 0.5 * stats.entropy(ref_hist + 1e-10, m + 1e-10) + \
                     0.5 * stats.entropy(test_hist + 1e-10, m + 1e-10)
            hist_score = min(1.0, js_div / 0.5)  # Normalize

        # Combine scores
        weights = {'feature': 0.5, 'kspace': 0.3, 'histogram': 0.2}
        combined_score = (
            weights['feature'] * feature_score +
            weights['kspace'] * kspace_score +
            weights['histogram'] * hist_score
        )

        # Determine shift type
        if feature_score > 0.7:
            shift_type = 'structural_anomaly'
        elif kspace_score > 0.7:
            shift_type = 'scanner_protocol'
        elif hist_score > 0.7:
            shift_type = 'intensity_distribution'
        else:
            shift_type = 'unknown'

        is_ood = combined_score > 0.5
        confidence = abs(combined_score - 0.5) * 2  # Confidence in decision

        details = {
            'feature_score': feature_score,
            'kspace_score': kspace_score,
            'histogram_score': hist_score,
            'kspace_details': kspace_details,
            'statistical_features': features['statistical'].cpu().numpy().tolist()
        } if return_details else {}

        return DistributionShiftResult(
            is_ood=is_ood,
            ood_score=combined_score,
            confidence=confidence,
            shift_type=shift_type,
            details=details
        )


class ScannerFingerprintDetector:
    """
    Detect scanner/protocol from k-space characteristics.

    Different MRI scanners leave characteristic "fingerprints"
    in k-space that can be detected and classified.
    """

    def __init__(self):
        self.known_fingerprints = {}
        self.classifier = None

    def register_scanner(
        self,
        scanner_id: str,
        kspace_samples: List[torch.Tensor]
    ):
        """Register a known scanner's fingerprint."""
        features = []
        analyzer = KSpaceDistributionAnalyzer()

        for ks in kspace_samples:
            feat = analyzer.extract_kspace_features(ks)
            # Flatten features
            flat_feat = []
            for v in feat.values():
                if isinstance(v, list):
                    flat_feat.extend(v)
                else:
                    flat_feat.append(v)
            features.append(flat_feat)

        self.known_fingerprints[scanner_id] = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0) + 1e-6,
            'samples': features
        }

    def identify_scanner(
        self,
        kspace: torch.Tensor
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Identify which scanner produced the k-space data.

        Returns:
            best_match: Scanner ID of best match
            confidence: Confidence in match
            all_scores: Scores for all known scanners
        """
        if not self.known_fingerprints:
            return 'unknown', 0.0, {}

        analyzer = KSpaceDistributionAnalyzer()
        feat = analyzer.extract_kspace_features(kspace)

        # Flatten
        flat_feat = []
        for v in feat.values():
            if isinstance(v, list):
                flat_feat.extend(v)
            else:
                flat_feat.append(v)
        flat_feat = np.array(flat_feat)

        # Compare to all known fingerprints
        all_scores = {}
        for scanner_id, fingerprint in self.known_fingerprints.items():
            z_score = np.abs(flat_feat - fingerprint['mean']) / fingerprint['std']
            avg_z = z_score.mean()
            # Convert to similarity score
            similarity = np.exp(-avg_z / 2)
            all_scores[scanner_id] = similarity

        # Find best match
        best_match = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_match]

        # Confidence based on separation from second best
        sorted_scores = sorted(all_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            confidence = (sorted_scores[0] - sorted_scores[1]) / (sorted_scores[0] + 1e-8)
        else:
            confidence = best_score

        return best_match, confidence, all_scores


class AnatomyOutlierDetector:
    """
    Detect unusual anatomy that may cause AI reconstruction failures.

    Detects:
    - Abnormal organ sizes
    - Unusual structural patterns
    - Rare anatomical variants
    - Post-surgical changes
    """

    def __init__(self):
        self.anatomy_stats = {}

    def extract_anatomy_features(self, image: torch.Tensor) -> Dict[str, float]:
        """Extract anatomy-related features."""
        if image.dim() > 2:
            image = image.squeeze()

        H, W = image.shape

        # Binarize to find major structures
        threshold = image.mean() + 0.5 * image.std()
        binary = (image > threshold).float()

        # Connected component-like analysis (simplified)
        # Find bounding box of main structure
        nonzero_y, nonzero_x = torch.where(binary > 0)

        if len(nonzero_y) == 0:
            return {
                'structure_area': 0,
                'bbox_ratio': 0,
                'centroid_offset': 0,
                'symmetry': 0,
                'complexity': 0
            }

        y_min, y_max = nonzero_y.min().item(), nonzero_y.max().item()
        x_min, x_max = nonzero_x.min().item(), nonzero_x.max().item()

        # Structure area ratio
        structure_area = binary.sum().item() / (H * W)

        # Bounding box aspect ratio
        bbox_height = y_max - y_min + 1
        bbox_width = x_max - x_min + 1
        bbox_ratio = bbox_height / (bbox_width + 1e-8)

        # Centroid offset from image center
        centroid_y = nonzero_y.float().mean().item()
        centroid_x = nonzero_x.float().mean().item()
        centroid_offset = np.sqrt(
            ((centroid_y - H/2) / H)**2 +
            ((centroid_x - W/2) / W)**2
        )

        # Left-right symmetry
        left_half = image[:, :W//2]
        right_half = torch.flip(image[:, W//2:], dims=[1])
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry = 1.0 - torch.abs(
            left_half[:, :min_width] - right_half[:, :min_width]
        ).mean().item() / (image.max().item() + 1e-8)

        # Boundary complexity (perimeter / sqrt(area))
        # Approximate perimeter using gradient
        grad_x = torch.abs(binary[:, 1:] - binary[:, :-1])
        grad_y = torch.abs(binary[1:, :] - binary[:-1, :])
        perimeter = grad_x.sum().item() + grad_y.sum().item()
        area = binary.sum().item() + 1e-8
        complexity = perimeter / np.sqrt(area)

        return {
            'structure_area': structure_area,
            'bbox_ratio': bbox_ratio,
            'centroid_offset': centroid_offset,
            'symmetry': symmetry,
            'complexity': complexity
        }

    def fit(self, images: List[torch.Tensor]):
        """Fit to training anatomy distribution."""
        all_features = []
        for img in images:
            feat = self.extract_anatomy_features(img)
            all_features.append(feat)

        # Compute statistics for each feature
        for key in all_features[0].keys():
            values = [f[key] for f in all_features]
            self.anatomy_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values) + 1e-6,
                'min': np.min(values),
                'max': np.max(values)
            }

    def detect_outlier(self, image: torch.Tensor) -> Tuple[bool, float, Dict]:
        """
        Detect if anatomy is an outlier.

        Returns:
            is_outlier: Boolean
            outlier_score: Continuous score [0, 1]
            details: Feature-wise breakdown
        """
        if not self.anatomy_stats:
            return False, 0.0, {}

        feat = self.extract_anatomy_features(image)
        z_scores = {}
        flags = []

        for key, value in feat.items():
            ref = self.anatomy_stats[key]
            z = abs(value - ref['mean']) / ref['std']
            z_scores[key] = z
            flags.append(z > 3)  # 3 sigma rule

        avg_z = np.mean(list(z_scores.values()))
        outlier_score = 1.0 / (1.0 + np.exp(-avg_z + 2))

        is_outlier = outlier_score > 0.5 or any(flags)

        return is_outlier, outlier_score, {
            'z_scores': z_scores,
            'features': feat
        }
