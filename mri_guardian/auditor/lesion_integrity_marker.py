"""
Lesion Integrity Marker (LIM) - Advanced Bioengineering Metric

This module implements a comprehensive lesion fingerprinting system that quantifies
how well AI reconstruction preserves clinically critical lesion characteristics.

The LIM is a single score (0-1) that summarizes "how intact is this lesion?"
- LIM = 1.0: Lesion perfectly preserved
- LIM = 0.0: Lesion completely corrupted

Key Features:
1. Multi-feature lesion fingerprinting (intensity, shape, texture, edges)
2. Comparison between ground truth and AI reconstructions
3. Per-lesion and aggregate analysis
4. Correlation with auditor suspicion scores
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.filters import sobel
import warnings


@dataclass
class LesionFingerprint:
    """
    Complete feature fingerprint of a lesion.

    This captures all measurable characteristics that define a lesion's
    appearance and should be preserved after reconstruction.
    """
    # Intensity & Contrast Features
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    min_intensity: float = 0.0
    max_intensity: float = 0.0
    background_mean: float = 0.0
    background_std: float = 0.0
    cnr: float = 0.0  # Contrast-to-Noise Ratio

    # Size & Shape Features
    area_pixels: float = 0.0
    perimeter: float = 0.0
    compactness: float = 0.0  # P² / (4π * A) - circle = 1
    eccentricity: float = 0.0  # 0 = circle, 1 = line
    solidity: float = 0.0  # area / convex_hull_area
    extent: float = 0.0  # area / bounding_box_area
    major_axis: float = 0.0
    minor_axis: float = 0.0

    # Texture Features (GLCM-based)
    texture_contrast: float = 0.0
    texture_dissimilarity: float = 0.0
    texture_homogeneity: float = 0.0
    texture_energy: float = 0.0
    texture_correlation: float = 0.0
    texture_entropy: float = 0.0

    # Edge Features
    edge_sharpness: float = 0.0  # Mean gradient magnitude at boundary
    edge_consistency: float = 0.0  # Std of gradient magnitudes
    boundary_gradient_mean: float = 0.0
    boundary_gradient_max: float = 0.0

    # Location (for verification)
    centroid_x: float = 0.0
    centroid_y: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert fingerprint to feature vector for comparison."""
        return np.array([
            self.mean_intensity, self.std_intensity,
            self.cnr, self.area_pixels, self.perimeter,
            self.compactness, self.eccentricity, self.solidity,
            self.texture_contrast, self.texture_homogeneity,
            self.texture_energy, self.texture_entropy,
            self.edge_sharpness, self.edge_consistency
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Names of features in the vector."""
        return [
            'mean_intensity', 'std_intensity', 'cnr', 'area',
            'perimeter', 'compactness', 'eccentricity', 'solidity',
            'texture_contrast', 'texture_homogeneity',
            'texture_energy', 'texture_entropy',
            'edge_sharpness', 'edge_consistency'
        ]


@dataclass
class LIMResult:
    """Result of Lesion Integrity Marker computation."""
    lim_score: float  # Overall LIM (0-1, higher = better preserved)

    # Component scores (each 0-1)
    intensity_score: float = 0.0
    contrast_score: float = 0.0
    shape_score: float = 0.0
    texture_score: float = 0.0
    edge_score: float = 0.0
    location_score: float = 0.0

    # Raw fingerprints
    gt_fingerprint: Optional[LesionFingerprint] = None
    recon_fingerprint: Optional[LesionFingerprint] = None

    # Detailed component metrics
    component_details: Dict[str, float] = field(default_factory=dict)


class LesionFingerprintExtractor:
    """
    Extracts comprehensive feature fingerprints from lesion regions.

    This is the core of the LIM system - it captures all measurable
    characteristics of a lesion that should be preserved after reconstruction.
    """

    def __init__(
        self,
        background_dilation: int = 5,
        glcm_distances: List[int] = None,
        glcm_angles: List[float] = None
    ):
        """
        Args:
            background_dilation: Pixels to dilate mask for background estimation
            glcm_distances: Distances for GLCM computation
            glcm_angles: Angles for GLCM computation
        """
        self.background_dilation = background_dilation
        self.glcm_distances = glcm_distances or [1, 2, 3]
        self.glcm_angles = glcm_angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]

    def extract(
        self,
        image: np.ndarray,
        lesion_mask: np.ndarray
    ) -> LesionFingerprint:
        """
        Extract complete fingerprint from a lesion region.

        Args:
            image: 2D image array (H, W), normalized to [0, 1]
            lesion_mask: Binary mask of lesion region (H, W)

        Returns:
            LesionFingerprint containing all extracted features
        """
        fp = LesionFingerprint()

        # Ensure mask is binary
        mask = lesion_mask > 0.5

        if mask.sum() < 4:  # Minimum viable lesion size
            return fp

        # Get lesion pixels
        lesion_pixels = image[mask]

        # === Intensity Features ===
        fp.mean_intensity = float(np.mean(lesion_pixels))
        fp.std_intensity = float(np.std(lesion_pixels))
        fp.min_intensity = float(np.min(lesion_pixels))
        fp.max_intensity = float(np.max(lesion_pixels))

        # Background estimation (dilated region minus lesion)
        dilated = ndimage.binary_dilation(
            mask,
            iterations=self.background_dilation
        )
        background_mask = dilated & ~mask

        if background_mask.sum() > 0:
            background_pixels = image[background_mask]
            fp.background_mean = float(np.mean(background_pixels))
            fp.background_std = float(np.std(background_pixels)) + 1e-8

            # Contrast-to-Noise Ratio
            fp.cnr = abs(fp.mean_intensity - fp.background_mean) / fp.background_std

        # === Shape Features ===
        # Use regionprops for robust shape analysis
        labeled_mask = label(mask.astype(int))
        regions = regionprops(labeled_mask, intensity_image=image)

        if len(regions) > 0:
            region = regions[0]  # Take largest region

            fp.area_pixels = float(region.area)
            fp.perimeter = float(region.perimeter) if region.perimeter > 0 else 0

            # Compactness (circularity): 4π * area / perimeter²
            if fp.perimeter > 0:
                fp.compactness = (4 * np.pi * fp.area_pixels) / (fp.perimeter ** 2)

            fp.eccentricity = float(region.eccentricity)
            fp.solidity = float(region.solidity)
            fp.extent = float(region.extent)
            fp.major_axis = float(region.major_axis_length)
            fp.minor_axis = float(region.minor_axis_length)

            # Centroid
            fp.centroid_y, fp.centroid_x = region.centroid

        # === Texture Features (GLCM) ===
        fp = self._extract_texture_features(image, mask, fp)

        # === Edge Features ===
        fp = self._extract_edge_features(image, mask, fp)

        return fp

    def _extract_texture_features(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        fp: LesionFingerprint
    ) -> LesionFingerprint:
        """Extract GLCM-based texture features from lesion region."""

        # Get bounding box of lesion
        coords = np.where(mask)
        if len(coords[0]) < 4:
            return fp

        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        # Extract lesion patch (with some padding)
        pad = 2
        y_min = max(0, y_min - pad)
        y_max = min(image.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(image.shape[1], x_max + pad)

        patch = image[y_min:y_max, x_min:x_max]

        if patch.size < 16:  # Too small for texture analysis
            return fp

        # Quantize to 16 levels for GLCM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            patch_quantized = (patch * 15).astype(np.uint8)
            patch_quantized = np.clip(patch_quantized, 0, 15)

        try:
            # Compute GLCM
            glcm = graycomatrix(
                patch_quantized,
                distances=self.glcm_distances,
                angles=self.glcm_angles,
                levels=16,
                symmetric=True,
                normed=True
            )

            # Extract properties (averaged over distances and angles)
            fp.texture_contrast = float(np.mean(graycoprops(glcm, 'contrast')))
            fp.texture_dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
            fp.texture_homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
            fp.texture_energy = float(np.mean(graycoprops(glcm, 'energy')))
            fp.texture_correlation = float(np.mean(graycoprops(glcm, 'correlation')))

            # Entropy from GLCM
            glcm_flat = glcm.flatten()
            glcm_flat = glcm_flat[glcm_flat > 0]
            fp.texture_entropy = float(-np.sum(glcm_flat * np.log2(glcm_flat + 1e-10)))

        except Exception:
            pass  # Keep default values if GLCM fails

        return fp

    def _extract_edge_features(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        fp: LesionFingerprint
    ) -> LesionFingerprint:
        """Extract edge/boundary features from lesion."""

        # Compute gradient magnitude
        gradient = sobel(image)

        # Get boundary pixels (edge of mask)
        eroded = ndimage.binary_erosion(mask, iterations=1)
        boundary = mask & ~eroded

        if boundary.sum() > 0:
            boundary_gradients = gradient[boundary]

            fp.edge_sharpness = float(np.mean(boundary_gradients))
            fp.edge_consistency = float(np.std(boundary_gradients))
            fp.boundary_gradient_mean = float(np.mean(boundary_gradients))
            fp.boundary_gradient_max = float(np.max(boundary_gradients))

        return fp


class LesionIntegrityMarker:
    """
    Computes the Lesion Integrity Marker (LIM) by comparing
    lesion fingerprints between ground truth and reconstruction.

    The LIM is a single score (0-1) that tells clinicians:
    "How well did AI preserve this lesion?"
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        danger_threshold: float = 0.7
    ):
        """
        Args:
            weights: Component weights for final LIM score
            danger_threshold: LIM below this triggers warning
        """
        # Default weights (clinically justified)
        self.weights = weights or {
            'intensity': 0.15,    # Important for diagnosis
            'contrast': 0.25,    # Critical for visibility
            'shape': 0.20,       # Lesion morphology matters
            'texture': 0.15,     # Internal structure
            'edge': 0.15,        # Boundary definition
            'location': 0.10     # Should stay in place
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        self.danger_threshold = danger_threshold
        self.extractor = LesionFingerprintExtractor()

    def compute_lim(
        self,
        gt_image: np.ndarray,
        recon_image: np.ndarray,
        lesion_mask: np.ndarray
    ) -> LIMResult:
        """
        Compute LIM by comparing lesion in ground truth vs reconstruction.

        Args:
            gt_image: Ground truth image (H, W), normalized [0, 1]
            recon_image: Reconstructed image (H, W), normalized [0, 1]
            lesion_mask: Binary lesion mask (H, W)

        Returns:
            LIMResult with overall score and component breakdown
        """
        # Extract fingerprints
        gt_fp = self.extractor.extract(gt_image, lesion_mask)
        recon_fp = self.extractor.extract(recon_image, lesion_mask)

        # Compute component scores
        intensity_score = self._compare_intensity(gt_fp, recon_fp)
        contrast_score = self._compare_contrast(gt_fp, recon_fp)
        shape_score = self._compare_shape(gt_fp, recon_fp)
        texture_score = self._compare_texture(gt_fp, recon_fp)
        edge_score = self._compare_edges(gt_fp, recon_fp)
        location_score = self._compare_location(gt_fp, recon_fp)

        # Weighted combination
        lim_score = (
            self.weights['intensity'] * intensity_score +
            self.weights['contrast'] * contrast_score +
            self.weights['shape'] * shape_score +
            self.weights['texture'] * texture_score +
            self.weights['edge'] * edge_score +
            self.weights['location'] * location_score
        )

        return LIMResult(
            lim_score=float(lim_score),
            intensity_score=float(intensity_score),
            contrast_score=float(contrast_score),
            shape_score=float(shape_score),
            texture_score=float(texture_score),
            edge_score=float(edge_score),
            location_score=float(location_score),
            gt_fingerprint=gt_fp,
            recon_fingerprint=recon_fp,
            component_details={
                'gt_cnr': gt_fp.cnr,
                'recon_cnr': recon_fp.cnr,
                'gt_area': gt_fp.area_pixels,
                'recon_area': recon_fp.area_pixels,
                'gt_edge_sharpness': gt_fp.edge_sharpness,
                'recon_edge_sharpness': recon_fp.edge_sharpness
            }
        )

    def _compare_intensity(
        self,
        gt_fp: LesionFingerprint,
        recon_fp: LesionFingerprint
    ) -> float:
        """Compare intensity preservation (0-1)."""
        if gt_fp.mean_intensity < 1e-6:
            return 1.0

        # Ratio of means (capped to [0, 1])
        mean_ratio = min(recon_fp.mean_intensity, gt_fp.mean_intensity) / \
                     max(recon_fp.mean_intensity, gt_fp.mean_intensity + 1e-8)

        # Ratio of stds
        if gt_fp.std_intensity > 1e-6:
            std_ratio = min(recon_fp.std_intensity, gt_fp.std_intensity) / \
                        max(recon_fp.std_intensity, gt_fp.std_intensity + 1e-8)
        else:
            std_ratio = 1.0

        return 0.7 * mean_ratio + 0.3 * std_ratio

    def _compare_contrast(
        self,
        gt_fp: LesionFingerprint,
        recon_fp: LesionFingerprint
    ) -> float:
        """Compare contrast-to-noise ratio preservation (0-1)."""
        if gt_fp.cnr < 1e-6:
            return 1.0

        # CNR ratio (capped)
        cnr_ratio = min(recon_fp.cnr, gt_fp.cnr) / max(recon_fp.cnr, gt_fp.cnr + 1e-8)

        # Penalize if CNR drops significantly
        if recon_fp.cnr < gt_fp.cnr * 0.5:
            cnr_ratio *= 0.5  # Heavy penalty for major CNR loss

        return float(np.clip(cnr_ratio, 0, 1))

    def _compare_shape(
        self,
        gt_fp: LesionFingerprint,
        recon_fp: LesionFingerprint
    ) -> float:
        """Compare shape preservation (0-1)."""
        if gt_fp.area_pixels < 1:
            return 1.0

        # Area ratio
        area_ratio = 1 - abs(recon_fp.area_pixels - gt_fp.area_pixels) / \
                     max(gt_fp.area_pixels, 1)
        area_ratio = max(0, area_ratio)

        # Compactness similarity
        compact_diff = abs(recon_fp.compactness - gt_fp.compactness)
        compact_score = max(0, 1 - compact_diff)

        # Eccentricity similarity
        ecc_diff = abs(recon_fp.eccentricity - gt_fp.eccentricity)
        ecc_score = max(0, 1 - ecc_diff)

        # Solidity similarity
        solid_diff = abs(recon_fp.solidity - gt_fp.solidity)
        solid_score = max(0, 1 - solid_diff)

        return 0.4 * area_ratio + 0.2 * compact_score + 0.2 * ecc_score + 0.2 * solid_score

    def _compare_texture(
        self,
        gt_fp: LesionFingerprint,
        recon_fp: LesionFingerprint
    ) -> float:
        """Compare texture preservation (0-1)."""
        # Compare each texture feature
        texture_scores = []

        # Contrast
        if gt_fp.texture_contrast > 1e-6:
            ratio = min(recon_fp.texture_contrast, gt_fp.texture_contrast) / \
                    max(recon_fp.texture_contrast, gt_fp.texture_contrast + 1e-8)
            texture_scores.append(ratio)

        # Homogeneity
        if gt_fp.texture_homogeneity > 1e-6:
            ratio = min(recon_fp.texture_homogeneity, gt_fp.texture_homogeneity) / \
                    max(recon_fp.texture_homogeneity, gt_fp.texture_homogeneity + 1e-8)
            texture_scores.append(ratio)

        # Energy
        if gt_fp.texture_energy > 1e-6:
            ratio = min(recon_fp.texture_energy, gt_fp.texture_energy) / \
                    max(recon_fp.texture_energy, gt_fp.texture_energy + 1e-8)
            texture_scores.append(ratio)

        # Entropy difference
        entropy_diff = abs(recon_fp.texture_entropy - gt_fp.texture_entropy)
        entropy_score = max(0, 1 - entropy_diff / (gt_fp.texture_entropy + 1e-8))
        texture_scores.append(entropy_score)

        if len(texture_scores) == 0:
            return 1.0

        return float(np.mean(texture_scores))

    def _compare_edges(
        self,
        gt_fp: LesionFingerprint,
        recon_fp: LesionFingerprint
    ) -> float:
        """Compare edge/boundary preservation (0-1)."""
        if gt_fp.edge_sharpness < 1e-6:
            return 1.0

        # Edge sharpness ratio
        sharp_ratio = min(recon_fp.edge_sharpness, gt_fp.edge_sharpness) / \
                      max(recon_fp.edge_sharpness, gt_fp.edge_sharpness + 1e-8)

        # Edge consistency (lower std = more consistent)
        if gt_fp.edge_consistency > 1e-6:
            consist_ratio = min(recon_fp.edge_consistency, gt_fp.edge_consistency) / \
                           max(recon_fp.edge_consistency, gt_fp.edge_consistency + 1e-8)
        else:
            consist_ratio = 1.0

        return 0.7 * sharp_ratio + 0.3 * consist_ratio

    def _compare_location(
        self,
        gt_fp: LesionFingerprint,
        recon_fp: LesionFingerprint
    ) -> float:
        """Compare location preservation (0-1)."""
        # Compute centroid distance
        dx = recon_fp.centroid_x - gt_fp.centroid_x
        dy = recon_fp.centroid_y - gt_fp.centroid_y
        distance = np.sqrt(dx**2 + dy**2)

        # Normalize by lesion size (major axis)
        norm_factor = max(gt_fp.major_axis, 1)
        normalized_distance = distance / norm_factor

        # Score (1 if no shift, decreases with distance)
        return float(np.exp(-normalized_distance))

    def is_lesion_at_risk(self, lim_result: LIMResult) -> bool:
        """Check if lesion integrity is below danger threshold."""
        return lim_result.lim_score < self.danger_threshold

    def get_risk_category(self, lim_result: LIMResult) -> str:
        """Categorize lesion preservation risk."""
        score = lim_result.lim_score
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.8:
            return "GOOD"
        elif score >= 0.7:
            return "ACCEPTABLE"
        elif score >= 0.5:
            return "WARNING"
        else:
            return "CRITICAL"


class LIMAggregator:
    """
    Aggregates LIM results across multiple lesions and reconstructions.

    This provides the statistical analysis needed for the ISEF experiment.
    """

    def __init__(self):
        self.results: Dict[str, List[LIMResult]] = {
            'blackbox': [],
            'guardian': []
        }

    def add_result(self, method: str, result: LIMResult):
        """Add a LIM result for a reconstruction method."""
        if method not in self.results:
            self.results[method] = []
        self.results[method].append(result)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics for each method."""
        stats = {}

        for method, results in self.results.items():
            if len(results) == 0:
                continue

            lim_scores = [r.lim_score for r in results]

            stats[method] = {
                'mean_lim': float(np.mean(lim_scores)),
                'std_lim': float(np.std(lim_scores)),
                'median_lim': float(np.median(lim_scores)),
                'min_lim': float(np.min(lim_scores)),
                'max_lim': float(np.max(lim_scores)),
                'num_lesions': len(results),

                # Risk analysis
                'pct_excellent': sum(1 for s in lim_scores if s >= 0.9) / len(lim_scores) * 100,
                'pct_good': sum(1 for s in lim_scores if 0.8 <= s < 0.9) / len(lim_scores) * 100,
                'pct_acceptable': sum(1 for s in lim_scores if 0.7 <= s < 0.8) / len(lim_scores) * 100,
                'pct_warning': sum(1 for s in lim_scores if 0.5 <= s < 0.7) / len(lim_scores) * 100,
                'pct_critical': sum(1 for s in lim_scores if s < 0.5) / len(lim_scores) * 100,

                # Component breakdown
                'mean_intensity_score': float(np.mean([r.intensity_score for r in results])),
                'mean_contrast_score': float(np.mean([r.contrast_score for r in results])),
                'mean_shape_score': float(np.mean([r.shape_score for r in results])),
                'mean_texture_score': float(np.mean([r.texture_score for r in results])),
                'mean_edge_score': float(np.mean([r.edge_score for r in results])),
                'mean_location_score': float(np.mean([r.location_score for r in results]))
            }

        return stats

    def compare_methods(self) -> Dict[str, Any]:
        """Statistical comparison between Guardian and black-box."""
        if 'blackbox' not in self.results or 'guardian' not in self.results:
            return {}

        bb_scores = [r.lim_score for r in self.results['blackbox']]
        g_scores = [r.lim_score for r in self.results['guardian']]

        if len(bb_scores) == 0 or len(g_scores) == 0:
            return {}

        # Mean improvement
        mean_improvement = np.mean(g_scores) - np.mean(bb_scores)

        # Percentage where Guardian is better
        paired_comparisons = min(len(bb_scores), len(g_scores))
        guardian_better = sum(
            1 for i in range(paired_comparisons)
            if g_scores[i] > bb_scores[i]
        )
        pct_guardian_better = guardian_better / paired_comparisons * 100

        # Critical lesion rate
        bb_critical = sum(1 for s in bb_scores if s < 0.5) / len(bb_scores) * 100
        g_critical = sum(1 for s in g_scores if s < 0.5) / len(g_scores) * 100

        return {
            'mean_improvement': float(mean_improvement),
            'pct_guardian_better': float(pct_guardian_better),
            'blackbox_critical_rate': float(bb_critical),
            'guardian_critical_rate': float(g_critical),
            'critical_rate_reduction': float(bb_critical - g_critical)
        }


class AuditorLIMCorrelator:
    """
    Analyzes correlation between auditor suspicion scores and LIM.

    This validates whether the auditor can predict lesion integrity issues.
    """

    def __init__(self):
        self.suspicion_scores: List[float] = []
        self.lim_scores: List[float] = []

    def add_pair(self, suspicion: float, lim: float):
        """Add a (suspicion, LIM) pair for correlation analysis."""
        self.suspicion_scores.append(suspicion)
        self.lim_scores.append(lim)

    def compute_correlation(self) -> Dict[str, float]:
        """Compute correlation between suspicion and LIM."""
        if len(self.suspicion_scores) < 3:
            return {'pearson': 0.0, 'spearman': 0.0, 'n_samples': 0}

        suspicion = np.array(self.suspicion_scores)
        lim = np.array(self.lim_scores)

        # We expect NEGATIVE correlation: high suspicion = low LIM
        pearson_r, pearson_p = pearsonr(suspicion, lim)
        spearman_r, spearman_p = spearmanr(suspicion, lim)

        return {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'n_samples': len(self.suspicion_scores),
            'expected_negative': pearson_r < 0  # Should be negative
        }

    def compute_detection_from_lim(
        self,
        lim_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Evaluate if auditor suspicion can predict low-LIM lesions.

        Args:
            lim_threshold: LIM below this is considered "at risk"

        Returns:
            Detection metrics for predicting low-LIM lesions
        """
        if len(self.suspicion_scores) < 3:
            return {}

        # Ground truth: LIM < threshold means lesion is at risk
        lim_labels = np.array([1 if lim < lim_threshold else 0
                               for lim in self.lim_scores])

        suspicion = np.array(self.suspicion_scores)

        # Use suspicion > median as "detected"
        threshold = np.median(suspicion)
        predictions = (suspicion > threshold).astype(int)

        # Compute metrics
        tp = np.sum((predictions == 1) & (lim_labels == 1))
        fp = np.sum((predictions == 1) & (lim_labels == 0))
        fn = np.sum((predictions == 0) & (lim_labels == 1))
        tn = np.sum((predictions == 0) & (lim_labels == 0))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'num_at_risk': int(np.sum(lim_labels)),
            'num_detected': int(np.sum(predictions)),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }


def create_lim_visualization(
    gt_image: np.ndarray,
    bb_image: np.ndarray,
    guardian_image: np.ndarray,
    lesion_mask: np.ndarray,
    bb_lim: LIMResult,
    guardian_lim: LIMResult,
    save_path: str = None
) -> 'plt.Figure':
    """
    Create comprehensive LIM visualization figure.

    Shows:
    - Ground truth, black-box, and Guardian reconstructions
    - LIM score breakdown for each
    - Component radar charts
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(16, 10))

    # Row 1: Images
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(gt_image, cmap='gray')
    ax1.contour(lesion_mask, colors='lime', linewidths=1)
    ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(bb_image, cmap='gray')
    ax2.contour(lesion_mask, colors='red', linewidths=1)
    ax2.set_title(f'Black-box (LIM={bb_lim.lim_score:.3f})', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(guardian_image, cmap='gray')
    ax3.contour(lesion_mask, colors='cyan', linewidths=1)
    ax3.set_title(f'Guardian (LIM={guardian_lim.lim_score:.3f})', fontsize=12)
    ax3.axis('off')

    # Difference maps
    ax4 = fig.add_subplot(2, 4, 4)
    diff_bb = np.abs(gt_image - bb_image)
    diff_g = np.abs(gt_image - guardian_image)
    combined = np.stack([diff_bb, diff_g, np.zeros_like(diff_bb)], axis=-1)
    combined = combined / (combined.max() + 1e-8)
    ax4.imshow(combined)
    ax4.contour(lesion_mask, colors='white', linewidths=1)
    ax4.set_title('Difference (R=BB, G=Guardian)', fontsize=12)
    ax4.axis('off')

    # Row 2: Component breakdown bars
    ax5 = fig.add_subplot(2, 4, 5)
    components = ['Intensity', 'Contrast', 'Shape', 'Texture', 'Edge', 'Location']
    bb_scores = [bb_lim.intensity_score, bb_lim.contrast_score, bb_lim.shape_score,
                 bb_lim.texture_score, bb_lim.edge_score, bb_lim.location_score]
    g_scores = [guardian_lim.intensity_score, guardian_lim.contrast_score, guardian_lim.shape_score,
                guardian_lim.texture_score, guardian_lim.edge_score, guardian_lim.location_score]

    x = np.arange(len(components))
    width = 0.35
    ax5.bar(x - width/2, bb_scores, width, label='Black-box', color='red', alpha=0.7)
    ax5.bar(x + width/2, g_scores, width, label='Guardian', color='green', alpha=0.7)
    ax5.set_ylabel('Score (0-1)')
    ax5.set_title('LIM Component Breakdown')
    ax5.set_xticks(x)
    ax5.set_xticklabels(components, rotation=45, ha='right')
    ax5.legend()
    ax5.set_ylim(0, 1.1)
    ax5.axhline(y=0.7, color='orange', linestyle='--', label='Danger threshold')

    # Radar chart for Guardian
    ax6 = fig.add_subplot(2, 4, 6, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    bb_radar = bb_scores + [bb_scores[0]]
    g_radar = g_scores + [g_scores[0]]

    ax6.plot(angles, bb_radar, 'o-', linewidth=2, color='red', label='Black-box')
    ax6.fill(angles, bb_radar, alpha=0.25, color='red')
    ax6.plot(angles, g_radar, 'o-', linewidth=2, color='green', label='Guardian')
    ax6.fill(angles, g_radar, alpha=0.25, color='green')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(components, size=8)
    ax6.set_title('Component Radar', fontsize=12)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Summary text
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.axis('off')

    summary_text = f"""
    LESION INTEGRITY MARKER SUMMARY
    ═══════════════════════════════

    Black-box Reconstruction:
      Overall LIM: {bb_lim.lim_score:.3f}
      Risk Level: {'⚠️ AT RISK' if bb_lim.lim_score < 0.7 else '✓ OK'}
      CNR: {bb_lim.component_details.get('recon_cnr', 0):.2f}

    Guardian Reconstruction:
      Overall LIM: {guardian_lim.lim_score:.3f}
      Risk Level: {'⚠️ AT RISK' if guardian_lim.lim_score < 0.7 else '✓ OK'}
      CNR: {guardian_lim.component_details.get('recon_cnr', 0):.2f}

    Improvement: {(guardian_lim.lim_score - bb_lim.lim_score)*100:+.1f}%
    """

    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Clinical interpretation
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')

    if bb_lim.lim_score < 0.5:
        bb_interpretation = "CRITICAL: Major lesion corruption"
    elif bb_lim.lim_score < 0.7:
        bb_interpretation = "WARNING: Significant degradation"
    else:
        bb_interpretation = "Acceptable preservation"

    if guardian_lim.lim_score < 0.5:
        g_interpretation = "CRITICAL: Major lesion corruption"
    elif guardian_lim.lim_score < 0.7:
        g_interpretation = "WARNING: Significant degradation"
    else:
        g_interpretation = "Acceptable preservation"

    clinical_text = f"""
    CLINICAL INTERPRETATION
    ═══════════════════════

    Black-box: {bb_interpretation}
    Guardian:  {g_interpretation}

    Key Finding:
    {'Guardian preserves lesion integrity significantly better'
     if guardian_lim.lim_score > bb_lim.lim_score + 0.1
     else 'Both methods show similar preservation'}

    Recommendation:
    {'Review black-box reconstruction with caution'
     if bb_lim.lim_score < 0.7
     else 'Lesion appears well-preserved'}
    """

    ax8.text(0.1, 0.9, clinical_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
