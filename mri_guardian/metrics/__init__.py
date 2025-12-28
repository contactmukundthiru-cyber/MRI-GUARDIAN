"""
Metrics Module for MRI-GUARDIAN

Provides evaluation metrics for:
- Image quality (PSNR, SSIM, NRMSE, HFEN)
- Detection performance (ROC, AUC, F1, precision, recall)
- Statistical analysis (t-tests, confidence intervals)
- Clinical significance (lesion SNR, CNR, tumor conspicuity)
- Downstream task preservation (Dice, detection accuracy)
"""

from .image_quality import (
    compute_psnr,
    compute_ssim,
    compute_nrmse,
    compute_hfen,
    compute_vif,
    compute_all_image_metrics,
    ImageQualityMetrics,
)
from .detection import (
    compute_roc_curve,
    compute_auc,
    compute_precision_recall,
    compute_f1_score,
    compute_confusion_matrix,
    compute_detection_metrics,
    DetectionMetrics,
)
from .statistical import (
    paired_ttest,
    wilcoxon_test,
    compute_confidence_interval,
    bootstrap_ci,
    StatisticalResults,
)
from .clinical_significance import (
    ClinicalMetrics,
    compute_lesion_specific_snr,
    compute_lesion_cnr,
    compute_tumor_conspicuity_index,
    compute_edge_preservation,
    compute_froc_curve,
    compute_alert_fatigue_metrics,
    compute_clinical_significance,
    generate_clinical_report,
)
from .downstream_task import (
    DownstreamTaskResult,
    dice_coefficient,
    iou_score,
    hausdorff_distance,
    evaluate_pathology_preservation,
    compare_reconstructions_clinically,
    generate_downstream_report,
)

__all__ = [
    # Image quality
    "compute_psnr",
    "compute_ssim",
    "compute_nrmse",
    "compute_hfen",
    "compute_vif",
    "compute_all_image_metrics",
    "ImageQualityMetrics",
    # Detection
    "compute_roc_curve",
    "compute_auc",
    "compute_precision_recall",
    "compute_f1_score",
    "compute_confusion_matrix",
    "compute_detection_metrics",
    "DetectionMetrics",
    # Statistical
    "paired_ttest",
    "wilcoxon_test",
    "compute_confidence_interval",
    "bootstrap_ci",
    "StatisticalResults",
    # Clinical Significance (what actually matters)
    "ClinicalMetrics",
    "compute_lesion_specific_snr",
    "compute_lesion_cnr",
    "compute_tumor_conspicuity_index",
    "compute_edge_preservation",
    "compute_froc_curve",
    "compute_alert_fatigue_metrics",
    "compute_clinical_significance",
    "generate_clinical_report",
    # Downstream Task (the REAL test)
    "DownstreamTaskResult",
    "dice_coefficient",
    "iou_score",
    "hausdorff_distance",
    "evaluate_pathology_preservation",
    "compare_reconstructions_clinically",
    "generate_downstream_report",
]
