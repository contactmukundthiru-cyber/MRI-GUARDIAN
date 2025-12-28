"""
Clinical Significance Metrics
==============================

A pixel difference of 0.01 is mathematical error;
a missing meniscus tear is a clinical error.

This module provides:
1. Lesion-specific SNR/CNR (not global metrics)
2. Tumor Conspicuity Index
3. FROC curve analysis (what clinicians care about)
4. Alert fatigue quantification
5. Clinical detection metrics

CRITICAL FOR ISEF: These metrics matter more than global PSNR.
"""

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ClinicalMetrics:
    """Clinical significance metrics for a reconstruction."""
    # Lesion-specific metrics (THE IMPORTANT ONES)
    lesion_snr: float = 0.0
    lesion_cnr: float = 0.0
    tumor_conspicuity_index: float = 0.0
    lesion_edge_preservation: float = 0.0

    # Detection metrics
    detection_sensitivity: float = 0.0
    detection_specificity: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

    # Clinical risk
    clinical_significance_score: float = 0.0
    risk_level: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            'lesion_snr': self.lesion_snr,
            'lesion_cnr': self.lesion_cnr,
            'tumor_conspicuity_index': self.tumor_conspicuity_index,
            'lesion_edge_preservation': self.lesion_edge_preservation,
            'detection_sensitivity': self.detection_sensitivity,
            'detection_specificity': self.detection_specificity,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'clinical_significance_score': self.clinical_significance_score,
            'risk_level': self.risk_level
        }


def compute_lesion_specific_snr(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    lesion_mask: np.ndarray
) -> float:
    """
    Compute SNR specifically on the lesion region.

    Global SNR is meaningless if the tumor is blurred out.
    """
    if lesion_mask.sum() < 10:
        return 0.0

    mask = lesion_mask > 0.5

    # Signal: mean intensity in lesion
    lesion_signal = reconstruction[mask]
    signal_power = np.mean(lesion_signal ** 2)

    # Noise: difference from ground truth in lesion
    lesion_noise = reconstruction[mask] - ground_truth[mask]
    noise_power = np.mean(lesion_noise ** 2) + 1e-10

    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def compute_lesion_cnr(
    reconstruction: np.ndarray,
    lesion_mask: np.ndarray,
    background_dilation: int = 10
) -> float:
    """
    Compute Contrast-to-Noise Ratio for lesion.

    CNR = |μ_lesion - μ_background| / σ_background

    This is what radiologists actually use.
    """
    if lesion_mask.sum() < 10:
        return 0.0

    mask = lesion_mask > 0.5

    # Background: dilated region minus lesion
    dilated = ndimage.binary_dilation(mask, iterations=background_dilation)
    background_mask = dilated & ~mask

    if background_mask.sum() < 10:
        return 0.0

    lesion_mean = np.mean(reconstruction[mask])
    background_mean = np.mean(reconstruction[background_mask])
    background_std = np.std(reconstruction[background_mask]) + 1e-8

    cnr = abs(lesion_mean - background_mean) / background_std
    return float(cnr)


def compute_tumor_conspicuity_index(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    lesion_mask: np.ndarray
) -> float:
    """
    Tumor Conspicuity Index: How visible is the tumor after reconstruction?

    TCI = CNR_reconstruction / CNR_ground_truth

    TCI = 1.0: Tumor as visible as in ground truth
    TCI < 1.0: Tumor less visible (dangerous!)
    TCI > 1.0: Tumor more visible (could be enhancement or hallucination)
    """
    cnr_recon = compute_lesion_cnr(reconstruction, lesion_mask)
    cnr_gt = compute_lesion_cnr(ground_truth, lesion_mask)

    if cnr_gt < 0.1:
        return 1.0  # No meaningful lesion

    tci = cnr_recon / cnr_gt
    return float(np.clip(tci, 0, 2))


def compute_edge_preservation(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    lesion_mask: np.ndarray
) -> float:
    """
    Measure how well lesion boundaries are preserved.

    Sharp edges are critical for surgical planning.
    """
    if lesion_mask.sum() < 10:
        return 0.0

    mask = lesion_mask > 0.5

    # Get boundary
    eroded = ndimage.binary_erosion(mask, iterations=2)
    boundary = mask & ~eroded

    if boundary.sum() < 5:
        return 0.0

    # Edge magnitude
    from scipy.ndimage import sobel
    recon_edges = np.sqrt(sobel(reconstruction, axis=0)**2 + sobel(reconstruction, axis=1)**2)
    gt_edges = np.sqrt(sobel(ground_truth, axis=0)**2 + sobel(ground_truth, axis=1)**2)

    # Correlation at boundary
    recon_boundary = recon_edges[boundary]
    gt_boundary = gt_edges[boundary]

    if np.std(recon_boundary) < 1e-6 or np.std(gt_boundary) < 1e-6:
        return 0.0

    correlation = np.corrcoef(recon_boundary, gt_boundary)[0, 1]
    return float(max(0, correlation))


def compute_froc_curve(
    predictions: np.ndarray,
    ground_truth_masks: List[np.ndarray],
    thresholds: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Compute Free-Response ROC (FROC) curve.

    This is what clinical judges care about:
    - Sensitivity vs False Positives per Image

    Unlike ROC, FROC accounts for localization accuracy.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    sensitivities = []
    fps_per_image = []

    for thresh in thresholds:
        tp = 0
        fn = 0
        fp = 0
        n_images = len(ground_truth_masks)

        # For each image
        for i, gt_mask in enumerate(ground_truth_masks):
            pred = predictions[i] > thresh
            gt = gt_mask > 0.5

            # True lesion detection
            if gt.sum() > 0:
                # Lesion detected if any prediction overlaps
                overlap = pred & gt
                if overlap.sum() > 0:
                    tp += 1
                else:
                    fn += 1

            # False positives: predictions outside lesions
            fp_mask = pred & ~gt
            fp += fp_mask.sum() / 1000  # Normalize by area

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fps = fp / n_images

        sensitivities.append(sensitivity)
        fps_per_image.append(fps)

    return {
        'thresholds': thresholds,
        'sensitivity': np.array(sensitivities),
        'fps_per_image': np.array(fps_per_image)
    }


def compute_alert_fatigue_metrics(
    suspicion_scores: np.ndarray,
    has_lesion: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Quantify alert fatigue - CRITICAL for clinical deployment.

    If the auditor flags >5% of perfectly good scans as "Suspicious",
    radiologists will ignore ALL alerts.

    Args:
        suspicion_scores: Auditor output (0-1) for each scan
        has_lesion: Binary array indicating if scan has real lesion
        threshold: Suspicion threshold for alerting

    Returns:
        Alert fatigue metrics
    """
    # Separate healthy and pathological scans
    healthy_mask = has_lesion == 0
    pathological_mask = has_lesion == 1

    healthy_scores = suspicion_scores[healthy_mask]
    pathological_scores = suspicion_scores[pathological_mask]

    # False alerts on healthy scans (BAD!)
    false_alerts = (healthy_scores > threshold).sum()
    false_alert_rate = false_alerts / len(healthy_scores) if len(healthy_scores) > 0 else 0

    # Missed pathology (VERY BAD!)
    missed_pathology = (pathological_scores < threshold).sum()
    miss_rate = missed_pathology / len(pathological_scores) if len(pathological_scores) > 0 else 0

    # True alerts
    true_alerts = (pathological_scores > threshold).sum()
    true_alert_rate = true_alerts / len(pathological_scores) if len(pathological_scores) > 0 else 0

    # Alert precision (of all alerts, how many are real?)
    total_alerts = false_alerts + true_alerts
    alert_precision = true_alerts / total_alerts if total_alerts > 0 else 0

    # Clinical acceptability
    # False alert rate must be < 5% and miss rate < 2%
    is_acceptable = false_alert_rate < 0.05 and miss_rate < 0.02

    return {
        'false_alert_rate': float(false_alert_rate),
        'miss_rate': float(miss_rate),
        'true_alert_rate': float(true_alert_rate),
        'alert_precision': float(alert_precision),
        'n_healthy_scans': int(len(healthy_scores)),
        'n_pathological_scans': int(len(pathological_scores)),
        'is_clinically_acceptable': is_acceptable,
        'recommendation': 'PASS' if is_acceptable else 'FAIL - Adjust threshold'
    }


def compute_clinical_significance(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    lesion_mask: np.ndarray
) -> ClinicalMetrics:
    """
    Compute all clinical significance metrics for a reconstruction.

    Returns a complete clinical assessment.
    """
    metrics = ClinicalMetrics()

    # Lesion-specific metrics
    metrics.lesion_snr = compute_lesion_specific_snr(reconstruction, ground_truth, lesion_mask)
    metrics.lesion_cnr = compute_lesion_cnr(reconstruction, lesion_mask)
    metrics.tumor_conspicuity_index = compute_tumor_conspicuity_index(
        reconstruction, ground_truth, lesion_mask
    )
    metrics.lesion_edge_preservation = compute_edge_preservation(
        reconstruction, ground_truth, lesion_mask
    )

    # Overall clinical significance score
    # Weighted combination focused on what matters clinically
    metrics.clinical_significance_score = (
        0.25 * min(metrics.lesion_snr / 20, 1.0) +  # SNR up to 20 dB
        0.30 * min(metrics.lesion_cnr / 5, 1.0) +    # CNR up to 5
        0.30 * metrics.tumor_conspicuity_index +      # TCI around 1
        0.15 * metrics.lesion_edge_preservation       # Edge preservation
    )

    # Risk level
    if metrics.clinical_significance_score >= 0.9:
        metrics.risk_level = "LOW"
    elif metrics.clinical_significance_score >= 0.7:
        metrics.risk_level = "MODERATE"
    elif metrics.clinical_significance_score >= 0.5:
        metrics.risk_level = "HIGH"
    else:
        metrics.risk_level = "CRITICAL"

    return metrics


def compare_methods_clinically(
    ground_truth: np.ndarray,
    lesion_mask: np.ndarray,
    reconstructions: Dict[str, np.ndarray]
) -> Dict[str, ClinicalMetrics]:
    """
    Compare multiple reconstruction methods using clinical metrics.

    Args:
        ground_truth: Ground truth image
        lesion_mask: Lesion segmentation mask
        reconstructions: Dict mapping method name to reconstruction

    Returns:
        Dict mapping method name to clinical metrics
    """
    results = {}

    for method_name, recon in reconstructions.items():
        results[method_name] = compute_clinical_significance(
            recon, ground_truth, lesion_mask
        )

    return results


def generate_clinical_report(
    metrics: ClinicalMetrics,
    method_name: str = "Unknown"
) -> str:
    """Generate a clinical-style report for a reconstruction."""

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║          CLINICAL SIGNIFICANCE REPORT                        ║
╠══════════════════════════════════════════════════════════════╣
║ Method: {method_name:<52} ║
╠══════════════════════════════════════════════════════════════╣
║ LESION-SPECIFIC METRICS (What Actually Matters)              ║
╠──────────────────────────────────────────────────────────────╣
║ Lesion SNR:              {metrics.lesion_snr:>8.2f} dB                    ║
║ Lesion CNR:              {metrics.lesion_cnr:>8.2f}                       ║
║ Tumor Conspicuity Index: {metrics.tumor_conspicuity_index:>8.2f} (1.0 = ideal)          ║
║ Edge Preservation:       {metrics.lesion_edge_preservation:>8.2f}                       ║
╠──────────────────────────────────────────────────────────────╣
║ OVERALL ASSESSMENT                                           ║
╠──────────────────────────────────────────────────────────────╣
║ Clinical Significance:   {metrics.clinical_significance_score:>8.2f}                       ║
║ Risk Level:              {metrics.risk_level:<8}                         ║
╠──────────────────────────────────────────────────────────────╣
"""

    if metrics.risk_level == "LOW":
        report += "║ ✓ SAFE FOR CLINICAL USE                                     ║\n"
    elif metrics.risk_level == "MODERATE":
        report += "║ ⚠ REVIEW RECOMMENDED                                        ║\n"
    elif metrics.risk_level == "HIGH":
        report += "║ ⚠ EXPERT REVIEW REQUIRED                                    ║\n"
    else:
        report += "║ ✗ DO NOT USE FOR CLINICAL DECISION                          ║\n"

    report += "╚══════════════════════════════════════════════════════════════╝"

    return report
