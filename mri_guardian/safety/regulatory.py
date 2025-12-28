"""
FDA-Style Regulatory Metrics Module

Computes metrics aligned with FDA guidelines for AI medical devices.
Provides a framework for how future MRI AI systems could be evaluated
before clinical deployment.

Metrics include:
- Residual error analysis
- Artifact severity scoring
- Uncertainty calibration
- Lesion detectability curves
- ROC/AUC for pathology preservation
- Performance stratification
- Failure mode analysis

Novel contribution: First ISEF-level implementation of FDA-aligned
safety metrics for AI medical imaging systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json
from datetime import datetime


@dataclass
class RegulatoryMetrics:
    """FDA-aligned metrics for a single evaluation."""
    # Image quality metrics
    psnr: float
    ssim: float
    nrmse: float

    # Clinical metrics
    lesion_sensitivity: float  # True positive rate for lesions
    lesion_specificity: float  # True negative rate
    artifact_severity: float  # Severity of reconstruction artifacts

    # Uncertainty metrics
    uncertainty_calibration_error: float
    uncertainty_coverage: float  # Fraction within confidence interval

    # Robustness metrics
    worst_case_psnr: float
    performance_std: float

    # Failure metrics
    failure_rate: float  # Rate of unacceptable reconstructions
    mean_time_to_failure: float  # Average samples until failure


@dataclass
class SafetyBenchmarkResult:
    """Complete safety benchmark evaluation result."""
    overall_pass: bool
    metrics: RegulatoryMetrics
    stratified_results: Dict[str, RegulatoryMetrics]
    failure_modes: List[Dict]
    recommendations: List[str]
    certification_level: str  # "cleared", "limited", "not_recommended"


@dataclass
class RegulatoryReport:
    """Complete regulatory compliance report."""
    device_name: str
    evaluation_date: str
    dataset_info: Dict
    benchmark_results: SafetyBenchmarkResult
    statistical_analysis: Dict
    clinical_evaluation: Dict
    conclusions: List[str]
    appendices: Dict


class ResidualErrorAnalyzer:
    """
    Analyze residual errors for regulatory compliance.

    Examines:
    - Error distribution characteristics
    - Spatial error patterns
    - Error correlation with anatomy
    - Systematic vs random errors
    """

    def analyze(
        self,
        predictions: List[torch.Tensor],
        ground_truths: List[torch.Tensor]
    ) -> Dict:
        """
        Analyze residual errors across a dataset.

        Args:
            predictions: List of predicted images
            ground_truths: List of ground truth images

        Returns:
            Dictionary with error analysis results
        """
        all_errors = []
        spatial_errors = []

        for pred, gt in zip(predictions, ground_truths):
            pred = pred.squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()

            error = pred - gt
            all_errors.append(error.flatten())
            spatial_errors.append(np.abs(error))

        # Aggregate errors
        all_errors_flat = np.concatenate(all_errors)

        # Error distribution statistics
        error_stats = {
            'mean': np.mean(all_errors_flat),
            'std': np.std(all_errors_flat),
            'median': np.median(all_errors_flat),
            'iqr': np.percentile(all_errors_flat, 75) - np.percentile(all_errors_flat, 25),
            'skewness': stats.skew(all_errors_flat),
            'kurtosis': stats.kurtosis(all_errors_flat),
            'p99': np.percentile(np.abs(all_errors_flat), 99),
            'max': np.max(np.abs(all_errors_flat))
        }

        # Test for normality (important for uncertainty quantification)
        if len(all_errors_flat) > 5000:
            # Use random sample for large datasets
            sample = np.random.choice(all_errors_flat, 5000, replace=False)
        else:
            sample = all_errors_flat

        _, normality_pvalue = stats.normaltest(sample)
        error_stats['is_normal'] = normality_pvalue > 0.05
        error_stats['normality_pvalue'] = normality_pvalue

        # Spatial error patterns
        avg_spatial_error = np.mean(spatial_errors, axis=0)

        # Check for systematic spatial bias
        H, W = avg_spatial_error.shape
        top_half = avg_spatial_error[:H//2, :].mean()
        bottom_half = avg_spatial_error[H//2:, :].mean()
        left_half = avg_spatial_error[:, :W//2].mean()
        right_half = avg_spatial_error[:, W//2:].mean()

        spatial_bias = {
            'top_bottom_ratio': top_half / (bottom_half + 1e-8),
            'left_right_ratio': left_half / (right_half + 1e-8),
            'center_periphery_ratio': self._center_periphery_ratio(avg_spatial_error)
        }

        # Detect systematic error regions
        high_error_threshold = np.percentile(avg_spatial_error, 90)
        systematic_regions = avg_spatial_error > high_error_threshold
        systematic_error_fraction = systematic_regions.sum() / systematic_regions.size

        return {
            'error_statistics': error_stats,
            'spatial_bias': spatial_bias,
            'systematic_error_fraction': systematic_error_fraction,
            'average_spatial_error': avg_spatial_error
        }

    def _center_periphery_ratio(self, error_map: np.ndarray) -> float:
        """Compute ratio of center to periphery errors."""
        H, W = error_map.shape
        cy, cx = H // 2, W // 2

        # Center region (inner 50%)
        margin_h, margin_w = H // 4, W // 4
        center = error_map[cy-margin_h:cy+margin_h, cx-margin_w:cx+margin_w]

        # Periphery
        periphery_mask = np.ones_like(error_map, dtype=bool)
        periphery_mask[cy-margin_h:cy+margin_h, cx-margin_w:cx+margin_w] = False
        periphery = error_map[periphery_mask]

        return center.mean() / (periphery.mean() + 1e-8)


class ArtifactSeverityScorer:
    """
    Score severity of reconstruction artifacts.

    Detects and quantifies:
    - Ringing/Gibbs artifacts
    - Aliasing
    - Blurring
    - Noise amplification
    - Structured artifacts
    """

    def __init__(self):
        self.artifact_weights = {
            'ringing': 0.3,
            'aliasing': 0.25,
            'blurring': 0.2,
            'noise': 0.15,
            'structured': 0.1
        }

    def score(
        self,
        reconstruction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Score artifact severity.

        Args:
            reconstruction: Reconstructed image
            ground_truth: Ground truth image

        Returns:
            Dictionary with artifact scores (0-1, higher = worse)
        """
        recon = reconstruction.squeeze().cpu().numpy()
        gt = ground_truth.squeeze().cpu().numpy()

        scores = {}

        # 1. Ringing detection (high-frequency oscillations near edges)
        scores['ringing'] = self._detect_ringing(recon, gt)

        # 2. Aliasing detection (wrap-around artifacts)
        scores['aliasing'] = self._detect_aliasing(recon, gt)

        # 3. Blurring (loss of high-frequency content)
        scores['blurring'] = self._detect_blurring(recon, gt)

        # 4. Noise amplification
        scores['noise'] = self._detect_noise_amplification(recon, gt)

        # 5. Structured artifacts (periodic patterns)
        scores['structured'] = self._detect_structured_artifacts(recon, gt)

        # Overall severity (weighted sum)
        scores['overall'] = sum(
            self.artifact_weights[k] * scores[k]
            for k in self.artifact_weights
        )

        return scores

    def _detect_ringing(self, recon: np.ndarray, gt: np.ndarray) -> float:
        """Detect ringing artifacts near edges."""
        # Find edges in ground truth
        from scipy import ndimage
        edges = ndimage.sobel(gt)
        edge_mask = np.abs(edges) > np.percentile(np.abs(edges), 90)

        # Dilate edge mask
        edge_region = ndimage.binary_dilation(edge_mask, iterations=5)

        # Compute error oscillation in edge regions
        error = recon - gt
        edge_error = error * edge_region

        # Ringing = high-frequency oscillations
        # Use Laplacian as proxy
        laplacian = ndimage.laplace(edge_error)
        ringing_score = np.abs(laplacian[edge_region]).mean()

        # Normalize
        return min(1.0, ringing_score / (gt.std() + 1e-8))

    def _detect_aliasing(self, recon: np.ndarray, gt: np.ndarray) -> float:
        """Detect aliasing artifacts."""
        error = recon - gt

        # Check for wrap-around patterns
        H, W = error.shape

        # Compare opposite sides
        top_error = error[:H//4, :]
        bottom_error = error[-H//4:, :]

        # Correlation between opposite sides (aliasing signature)
        correlation = np.corrcoef(top_error.flatten(), bottom_error.flatten())[0, 1]

        aliasing_score = max(0, correlation)  # Positive correlation = aliasing
        return min(1.0, aliasing_score)

    def _detect_blurring(self, recon: np.ndarray, gt: np.ndarray) -> float:
        """Detect blurring (loss of high frequencies)."""
        # Compare high-frequency content
        fft_recon = np.fft.fftshift(np.fft.fft2(recon))
        fft_gt = np.fft.fftshift(np.fft.fft2(gt))

        H, W = recon.shape
        cy, cx = H // 2, W // 2

        # High frequency region (outer 50% of k-space)
        y, x = np.ogrid[:H, :W]
        high_freq_mask = np.sqrt((x - cx)**2 + (y - cy)**2) > min(H, W) // 4

        hf_recon = np.abs(fft_recon[high_freq_mask]).sum()
        hf_gt = np.abs(fft_gt[high_freq_mask]).sum()

        # Ratio of high-frequency energy
        hf_ratio = hf_recon / (hf_gt + 1e-8)

        # Blurring = loss of HF = ratio < 1
        blurring_score = max(0, 1 - hf_ratio)
        return min(1.0, blurring_score)

    def _detect_noise_amplification(self, recon: np.ndarray, gt: np.ndarray) -> float:
        """Detect noise amplification."""
        # Estimate noise in both images
        from scipy import ndimage

        # High-pass filter to isolate noise
        hp_recon = recon - ndimage.gaussian_filter(recon, sigma=2)
        hp_gt = gt - ndimage.gaussian_filter(gt, sigma=2)

        noise_recon = hp_recon.std()
        noise_gt = hp_gt.std()

        # Amplification ratio
        amplification = noise_recon / (noise_gt + 1e-8)

        # Score: amplification > 1 is bad
        return min(1.0, max(0, amplification - 1))

    def _detect_structured_artifacts(self, recon: np.ndarray, gt: np.ndarray) -> float:
        """Detect structured/periodic artifacts."""
        error = recon - gt

        # FFT of error
        fft_error = np.abs(np.fft.fftshift(np.fft.fft2(error)))

        # Look for peaks (excluding DC)
        H, W = error.shape
        cy, cx = H // 2, W // 2

        # Mask out DC region
        fft_error[cy-3:cy+3, cx-3:cx+3] = 0

        # Count significant peaks
        threshold = fft_error.mean() + 5 * fft_error.std()
        peaks = fft_error > threshold

        # Structured artifacts have isolated peaks
        num_peaks = peaks.sum()

        return min(1.0, num_peaks / 20)  # Normalize


class UncertaintyCalibrationAnalyzer:
    """
    Analyze calibration of uncertainty estimates.

    Well-calibrated uncertainty should:
    - Cover true values at specified confidence levels
    - Be neither over- nor under-confident
    """

    def analyze(
        self,
        predictions: List[torch.Tensor],
        uncertainties: List[torch.Tensor],
        ground_truths: List[torch.Tensor],
        confidence_levels: List[float] = [0.5, 0.75, 0.9, 0.95, 0.99]
    ) -> Dict:
        """
        Analyze uncertainty calibration.

        Args:
            predictions: List of predictions
            uncertainties: List of uncertainty estimates (std dev)
            ground_truths: List of ground truths
            confidence_levels: Confidence levels to evaluate

        Returns:
            Dictionary with calibration results
        """
        from scipy.stats import norm

        coverage_results = {level: [] for level in confidence_levels}
        all_z_scores = []

        for pred, unc, gt in zip(predictions, uncertainties, ground_truths):
            pred = pred.squeeze().cpu().numpy()
            unc = unc.squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()

            # Z-scores
            z = (gt - pred) / (unc + 1e-8)
            all_z_scores.append(z.flatten())

            # Check coverage at each confidence level
            for level in confidence_levels:
                z_critical = norm.ppf((1 + level) / 2)
                covered = np.abs(z) <= z_critical
                coverage_results[level].append(covered.mean())

        # Aggregate results
        all_z_flat = np.concatenate(all_z_scores)

        # Expected calibration error
        expected_coverage = {}
        actual_coverage = {}
        ece = 0

        for level in confidence_levels:
            expected_coverage[level] = level
            actual_coverage[level] = np.mean(coverage_results[level])
            ece += abs(actual_coverage[level] - level)

        ece /= len(confidence_levels)

        # Sharpness (average uncertainty width)
        all_unc = np.concatenate([u.squeeze().cpu().numpy().flatten() for u in uncertainties])
        sharpness = all_unc.mean()

        # Overconfidence/underconfidence
        mean_coverage_gap = np.mean([
            actual_coverage[l] - l for l in confidence_levels
        ])

        if mean_coverage_gap < -0.05:
            confidence_status = 'overconfident'
        elif mean_coverage_gap > 0.05:
            confidence_status = 'underconfident'
        else:
            confidence_status = 'well_calibrated'

        return {
            'calibration_error': ece,
            'expected_coverage': expected_coverage,
            'actual_coverage': actual_coverage,
            'sharpness': sharpness,
            'confidence_status': confidence_status,
            'z_score_mean': np.mean(all_z_flat),
            'z_score_std': np.std(all_z_flat)
        }


class LesionDetectabilityCurveGenerator:
    """
    Generate lesion detectability curves.

    Shows relationship between:
    - Lesion size vs detection rate
    - Lesion contrast vs detection rate
    - Detection threshold vs sensitivity/specificity
    """

    def generate(
        self,
        lesion_detections: List[Dict],  # Each has size, contrast, detected, ground_truth
    ) -> Dict:
        """
        Generate detectability analysis.

        Args:
            lesion_detections: List of detection results with metadata

        Returns:
            Dictionary with detectability curves and metrics
        """
        # Organize by lesion characteristics
        by_size = {'small': [], 'medium': [], 'large': []}
        by_contrast = {'low': [], 'medium': [], 'high': []}

        all_scores = []
        all_labels = []

        for det in lesion_detections:
            # Categorize by size
            if det['size'] < 5:
                by_size['small'].append(det)
            elif det['size'] < 15:
                by_size['medium'].append(det)
            else:
                by_size['large'].append(det)

            # Categorize by contrast
            if det['contrast'] < 0.1:
                by_contrast['low'].append(det)
            elif det['contrast'] < 0.3:
                by_contrast['medium'].append(det)
            else:
                by_contrast['high'].append(det)

            all_scores.append(det.get('detection_score', det['detected']))
            all_labels.append(det['ground_truth'])

        # Detection rates by category
        size_detection_rates = {}
        for size, detections in by_size.items():
            if detections:
                rate = sum(1 for d in detections if d['detected']) / len(detections)
                size_detection_rates[size] = rate
            else:
                size_detection_rates[size] = None

        contrast_detection_rates = {}
        for contrast, detections in by_contrast.items():
            if detections:
                rate = sum(1 for d in detections if d['detected']) / len(detections)
                contrast_detection_rates[contrast] = rate
            else:
                contrast_detection_rates[contrast] = None

        # ROC curve
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        if len(np.unique(all_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            roc_auc = auc(fpr, tpr)

            # Find optimal threshold (Youden's J)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
        else:
            fpr, tpr, thresholds = [0, 1], [0, 1], [0.5]
            roc_auc = 0.5
            optimal_threshold = 0.5

        # Precision-recall curve
        if len(np.unique(all_labels)) > 1:
            precision, recall, _ = precision_recall_curve(all_labels, all_scores)
            pr_auc = auc(recall, precision)
        else:
            precision, recall = [1], [1]
            pr_auc = 1.0

        return {
            'size_detection_rates': size_detection_rates,
            'contrast_detection_rates': contrast_detection_rates,
            'roc': {
                'fpr': list(fpr),
                'tpr': list(tpr),
                'auc': roc_auc,
                'optimal_threshold': optimal_threshold
            },
            'precision_recall': {
                'precision': list(precision),
                'recall': list(recall),
                'auc': pr_auc
            },
            'overall_sensitivity': sum(all_labels == 1) / len(all_labels) if all_labels.sum() > 0 else 0,
            'num_lesions_evaluated': len(lesion_detections)
        }


class FDAMetricsCalculator:
    """
    Calculate complete FDA-aligned metrics suite.

    Combines all regulatory metrics into a comprehensive evaluation.
    """

    def __init__(
        self,
        psnr_threshold: float = 30.0,  # Minimum acceptable PSNR
        ssim_threshold: float = 0.85,  # Minimum acceptable SSIM
        artifact_threshold: float = 0.3,  # Maximum acceptable artifact score
        lesion_sensitivity_threshold: float = 0.95  # Minimum lesion sensitivity
    ):
        self.thresholds = {
            'psnr': psnr_threshold,
            'ssim': ssim_threshold,
            'artifact': artifact_threshold,
            'lesion_sensitivity': lesion_sensitivity_threshold
        }

        self.residual_analyzer = ResidualErrorAnalyzer()
        self.artifact_scorer = ArtifactSeverityScorer()
        self.calibration_analyzer = UncertaintyCalibrationAnalyzer()
        self.detectability_generator = LesionDetectabilityCurveGenerator()

    def calculate(
        self,
        predictions: List[torch.Tensor],
        ground_truths: List[torch.Tensor],
        uncertainties: Optional[List[torch.Tensor]] = None,
        lesion_detections: Optional[List[Dict]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> SafetyBenchmarkResult:
        """
        Calculate complete FDA metrics.

        Args:
            predictions: List of reconstructed images
            ground_truths: List of ground truth images
            uncertainties: Optional uncertainty estimates
            lesion_detections: Optional lesion detection results
            metadata: Optional per-sample metadata for stratification

        Returns:
            SafetyBenchmarkResult with complete evaluation
        """
        from mri_guardian.metrics.image_quality import compute_psnr, compute_ssim, compute_nrmse

        # Calculate base metrics
        psnr_values = []
        ssim_values = []
        nrmse_values = []
        artifact_scores = []

        for pred, gt in zip(predictions, ground_truths):
            psnr_values.append(compute_psnr(pred, gt))
            ssim_values.append(compute_ssim(pred, gt))
            nrmse_values.append(compute_nrmse(pred, gt))
            artifact_scores.append(self.artifact_scorer.score(pred, gt)['overall'])

        # Aggregate metrics
        metrics = RegulatoryMetrics(
            psnr=np.mean(psnr_values),
            ssim=np.mean(ssim_values),
            nrmse=np.mean(nrmse_values),
            lesion_sensitivity=0.0,  # Updated below if available
            lesion_specificity=0.0,
            artifact_severity=np.mean(artifact_scores),
            uncertainty_calibration_error=0.0,  # Updated below if available
            uncertainty_coverage=0.0,
            worst_case_psnr=np.percentile(psnr_values, 5),  # 5th percentile
            performance_std=np.std(psnr_values),
            failure_rate=sum(1 for p in psnr_values if p < self.thresholds['psnr']) / len(psnr_values),
            mean_time_to_failure=self._calculate_mttf(psnr_values)
        )

        # Uncertainty calibration
        if uncertainties is not None:
            cal_results = self.calibration_analyzer.analyze(
                predictions, uncertainties, ground_truths
            )
            metrics.uncertainty_calibration_error = cal_results['calibration_error']
            metrics.uncertainty_coverage = cal_results['actual_coverage'].get(0.95, 0)

        # Lesion detectability
        if lesion_detections is not None:
            detect_results = self.detectability_generator.generate(lesion_detections)
            metrics.lesion_sensitivity = detect_results['roc']['tpr'][-2] if len(detect_results['roc']['tpr']) > 1 else 0
            # Specificity at optimal threshold
            opt_idx = np.argmin(np.abs(
                np.array(detect_results['roc']['tpr']) - detect_results['roc']['tpr'][-2]
            ))
            metrics.lesion_specificity = 1 - detect_results['roc']['fpr'][opt_idx]

        # Stratified analysis
        stratified = {}
        if metadata is not None:
            # Group by metadata categories
            categories = {}
            for i, meta in enumerate(metadata):
                for key, value in meta.items():
                    if key not in categories:
                        categories[key] = {}
                    if value not in categories[key]:
                        categories[key][value] = []
                    categories[key][value].append(i)

            for key, value_indices in categories.items():
                stratified[key] = {}
                for value, indices in value_indices.items():
                    subset_psnr = [psnr_values[i] for i in indices]
                    subset_ssim = [ssim_values[i] for i in indices]
                    stratified[key][value] = RegulatoryMetrics(
                        psnr=np.mean(subset_psnr),
                        ssim=np.mean(subset_ssim),
                        nrmse=np.mean([nrmse_values[i] for i in indices]),
                        lesion_sensitivity=0,
                        lesion_specificity=0,
                        artifact_severity=np.mean([artifact_scores[i] for i in indices]),
                        uncertainty_calibration_error=0,
                        uncertainty_coverage=0,
                        worst_case_psnr=np.percentile(subset_psnr, 5),
                        performance_std=np.std(subset_psnr),
                        failure_rate=sum(1 for p in subset_psnr if p < self.thresholds['psnr']) / len(subset_psnr),
                        mean_time_to_failure=0
                    )

        # Identify failure modes
        failure_modes = self._analyze_failure_modes(
            predictions, ground_truths, psnr_values, artifact_scores
        )

        # Overall pass/fail
        overall_pass = (
            metrics.psnr >= self.thresholds['psnr'] and
            metrics.ssim >= self.thresholds['ssim'] and
            metrics.artifact_severity <= self.thresholds['artifact'] and
            metrics.failure_rate < 0.05
        )

        if lesion_detections is not None:
            overall_pass = overall_pass and metrics.lesion_sensitivity >= self.thresholds['lesion_sensitivity']

        # Certification level
        if overall_pass and metrics.failure_rate < 0.01:
            certification = 'cleared'
        elif overall_pass:
            certification = 'limited'
        else:
            certification = 'not_recommended'

        # Recommendations
        recommendations = self._generate_recommendations(metrics, overall_pass)

        return SafetyBenchmarkResult(
            overall_pass=overall_pass,
            metrics=metrics,
            stratified_results=stratified,
            failure_modes=failure_modes,
            recommendations=recommendations,
            certification_level=certification
        )

    def _calculate_mttf(self, psnr_values: List[float]) -> float:
        """Calculate mean time (samples) to failure."""
        failures = [i for i, p in enumerate(psnr_values) if p < self.thresholds['psnr']]
        if not failures:
            return len(psnr_values)  # No failures
        return failures[0]  # First failure index

    def _analyze_failure_modes(
        self,
        predictions: List[torch.Tensor],
        ground_truths: List[torch.Tensor],
        psnr_values: List[float],
        artifact_scores: List[float]
    ) -> List[Dict]:
        """Analyze and categorize failure modes."""
        failure_modes = []

        for i, (pred, gt, psnr, artifact) in enumerate(zip(
            predictions, ground_truths, psnr_values, artifact_scores
        )):
            if psnr < self.thresholds['psnr'] or artifact > self.thresholds['artifact']:
                # This is a failure case
                mode = {
                    'index': i,
                    'psnr': psnr,
                    'artifact_score': artifact
                }

                # Categorize failure type
                if artifact > 0.5:
                    mode['type'] = 'severe_artifacts'
                elif psnr < 25:
                    mode['type'] = 'reconstruction_failure'
                else:
                    mode['type'] = 'moderate_degradation'

                failure_modes.append(mode)

        return failure_modes

    def _generate_recommendations(
        self,
        metrics: RegulatoryMetrics,
        overall_pass: bool
    ) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        if not overall_pass:
            recommendations.append(
                "CRITICAL: System does not meet minimum safety requirements for clinical use"
            )

        if metrics.psnr < self.thresholds['psnr']:
            recommendations.append(
                f"Improve reconstruction quality: PSNR {metrics.psnr:.1f} dB < "
                f"threshold {self.thresholds['psnr']} dB"
            )

        if metrics.artifact_severity > self.thresholds['artifact']:
            recommendations.append(
                f"Reduce artifact severity: {metrics.artifact_severity:.2f} > "
                f"threshold {self.thresholds['artifact']}"
            )

        if metrics.failure_rate > 0.05:
            recommendations.append(
                f"Reduce failure rate: {metrics.failure_rate:.1%} > 5% threshold"
            )

        if metrics.uncertainty_calibration_error > 0.1:
            recommendations.append(
                "Improve uncertainty calibration for reliable confidence estimates"
            )

        if metrics.performance_std > 3:
            recommendations.append(
                "Reduce performance variability for consistent results"
            )

        if not recommendations:
            recommendations.append("System meets all safety benchmarks")

        return recommendations


class SafetyBenchmark:
    """
    Run complete safety benchmark evaluation.

    Provides standardized testing protocol aligned with
    regulatory requirements.
    """

    def __init__(self, fda_calculator: Optional[FDAMetricsCalculator] = None):
        self.fda_calculator = fda_calculator or FDAMetricsCalculator()

    def run(
        self,
        model: nn.Module,
        test_dataloader,
        device: str = 'cuda'
    ) -> RegulatoryReport:
        """
        Run complete benchmark.

        Args:
            model: Model to evaluate
            test_dataloader: DataLoader with test data
            device: Device to run on

        Returns:
            RegulatoryReport with complete evaluation
        """
        model.eval()

        predictions = []
        ground_truths = []
        metadata = []

        with torch.no_grad():
            for batch in test_dataloader:
                masked_kspace = batch['masked_kspace'].to(device)
                mask = batch['mask'].to(device)
                target = batch['target']

                result = model(masked_kspace, mask)
                output = result['output'] if isinstance(result, dict) else result

                predictions.append(output.cpu())
                ground_truths.append(target)

                # Collect metadata if available
                if 'metadata' in batch:
                    metadata.extend(batch['metadata'])

        # Run FDA metrics
        benchmark_results = self.fda_calculator.calculate(
            predictions, ground_truths,
            metadata=metadata if metadata else None
        )

        # Generate report
        report = RegulatoryReport(
            device_name="MRI-GUARDIAN Reconstruction System",
            evaluation_date=datetime.now().isoformat(),
            dataset_info={
                'num_samples': len(predictions),
                'image_size': predictions[0].shape[-2:]
            },
            benchmark_results=benchmark_results,
            statistical_analysis={
                'residual_analysis': self.fda_calculator.residual_analyzer.analyze(
                    predictions, ground_truths
                )
            },
            clinical_evaluation={
                'certification_level': benchmark_results.certification_level,
                'clinical_readiness': benchmark_results.overall_pass
            },
            conclusions=self._generate_conclusions(benchmark_results),
            appendices={}
        )

        return report

    def _generate_conclusions(self, results: SafetyBenchmarkResult) -> List[str]:
        """Generate conclusions from benchmark results."""
        conclusions = []

        if results.overall_pass:
            conclusions.append(
                "The AI reconstruction system meets FDA safety benchmarks "
                "and is suitable for clinical evaluation"
            )
        else:
            conclusions.append(
                "The system does not meet minimum safety requirements "
                "and requires further development"
            )

        conclusions.append(
            f"Certification recommendation: {results.certification_level.upper()}"
        )

        if results.failure_modes:
            conclusions.append(
                f"Identified {len(results.failure_modes)} failure cases requiring attention"
            )

        return conclusions

    def export_report(
        self,
        report: RegulatoryReport,
        output_path: str,
        format: str = 'json'
    ):
        """Export report to file."""
        if format == 'json':
            # Convert to JSON-serializable format
            report_dict = {
                'device_name': report.device_name,
                'evaluation_date': report.evaluation_date,
                'dataset_info': report.dataset_info,
                'benchmark_results': {
                    'overall_pass': report.benchmark_results.overall_pass,
                    'certification_level': report.benchmark_results.certification_level,
                    'metrics': {
                        'psnr': report.benchmark_results.metrics.psnr,
                        'ssim': report.benchmark_results.metrics.ssim,
                        'nrmse': report.benchmark_results.metrics.nrmse,
                        'lesion_sensitivity': report.benchmark_results.metrics.lesion_sensitivity,
                        'artifact_severity': report.benchmark_results.metrics.artifact_severity,
                        'failure_rate': report.benchmark_results.metrics.failure_rate
                    },
                    'recommendations': report.benchmark_results.recommendations
                },
                'conclusions': report.conclusions
            }

            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)

        elif format == 'txt':
            with open(output_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("FDA-ALIGNED SAFETY BENCHMARK REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Device: {report.device_name}\n")
                f.write(f"Date: {report.evaluation_date}\n\n")

                f.write("-" * 40 + "\n")
                f.write("BENCHMARK RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Overall Pass: {report.benchmark_results.overall_pass}\n")
                f.write(f"Certification: {report.benchmark_results.certification_level}\n\n")

                f.write("Metrics:\n")
                m = report.benchmark_results.metrics
                f.write(f"  PSNR: {m.psnr:.2f} dB\n")
                f.write(f"  SSIM: {m.ssim:.4f}\n")
                f.write(f"  Artifact Severity: {m.artifact_severity:.4f}\n")
                f.write(f"  Failure Rate: {m.failure_rate:.2%}\n\n")

                f.write("-" * 40 + "\n")
                f.write("CONCLUSIONS\n")
                f.write("-" * 40 + "\n")
                for conclusion in report.conclusions:
                    f.write(f"• {conclusion}\n")
