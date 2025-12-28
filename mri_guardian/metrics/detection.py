"""
Detection Metrics for Hallucination Detection

Metrics for evaluating how well we detect hallucinations:
- ROC curve and AUC
- Precision, Recall, F1
- Confusion matrix
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    roc_curve as sklearn_roc_curve,
    auc as sklearn_auc,
    precision_recall_curve,
    average_precision_score,
    f1_score as sklearn_f1,
    confusion_matrix as sklearn_confusion_matrix,
)


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""
    auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    threshold: float
    average_precision: float


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().flatten()
    return np.array(tensor).flatten()


def compute_roc_curve(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.

    ROC = Receiver Operating Characteristic
    Plots True Positive Rate vs False Positive Rate at different thresholds.

    Args:
        predictions: Predicted scores/probabilities
        ground_truth: Binary ground truth (0 or 1)

    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Threshold values
    """
    pred_np = to_numpy(predictions)
    gt_np = to_numpy(ground_truth)

    # Binarize ground truth if needed
    gt_np = (gt_np > 0.5).astype(np.int32)

    fpr, tpr, thresholds = sklearn_roc_curve(gt_np, pred_np)

    return fpr, tpr, thresholds


def compute_auc(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor
) -> float:
    """
    Compute Area Under ROC Curve (AUC).

    AUC measures overall detection performance.
    Range: [0, 1], where:
    - 0.5 = random (useless)
    - 1.0 = perfect
    - > 0.7 = acceptable
    - > 0.8 = good
    - > 0.9 = excellent

    Args:
        predictions: Predicted scores
        ground_truth: Binary ground truth

    Returns:
        AUC value
    """
    fpr, tpr, _ = compute_roc_curve(predictions, ground_truth)
    return sklearn_auc(fpr, tpr)


def compute_precision_recall(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute precision and recall at a threshold.

    Precision = TP / (TP + FP) = "Of detected positives, how many are correct?"
    Recall = TP / (TP + FN) = "Of actual positives, how many did we detect?"

    Args:
        predictions: Predicted scores
        ground_truth: Binary ground truth
        threshold: Decision threshold

    Returns:
        precision, recall
    """
    pred_np = to_numpy(predictions)
    gt_np = to_numpy(ground_truth)

    pred_binary = (pred_np > threshold).astype(np.int32)
    gt_binary = (gt_np > 0.5).astype(np.int32)

    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return float(precision), float(recall)


def compute_f1_score(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)

    Harmonic mean of precision and recall.
    Good single metric when you care about both.

    Args:
        predictions: Predicted scores
        ground_truth: Binary ground truth
        threshold: Decision threshold

    Returns:
        F1 score
    """
    precision, recall = compute_precision_recall(predictions, ground_truth, threshold)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def compute_confusion_matrix(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix.

    [[TN, FP],
     [FN, TP]]

    Args:
        predictions: Predicted scores
        ground_truth: Binary ground truth
        threshold: Decision threshold

    Returns:
        2x2 confusion matrix
    """
    pred_np = to_numpy(predictions)
    gt_np = to_numpy(ground_truth)

    pred_binary = (pred_np > threshold).astype(np.int32)
    gt_binary = (gt_np > 0.5).astype(np.int32)

    return sklearn_confusion_matrix(gt_binary, pred_binary)


def compute_optimal_threshold(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    criterion: str = "f1"
) -> float:
    """
    Find optimal threshold for detection.

    Args:
        predictions: Predicted scores
        ground_truth: Binary ground truth
        criterion: "f1", "youden" (TPR - FPR), or "precision_recall_balance"

    Returns:
        Optimal threshold
    """
    pred_np = to_numpy(predictions)
    gt_np = to_numpy(ground_truth)
    gt_binary = (gt_np > 0.5).astype(np.int32)

    if criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(gt_binary, pred_np)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # Last point is always (1, 0)
        return float(thresholds[best_idx])

    elif criterion == "youden":
        fpr, tpr, thresholds = sklearn_roc_curve(gt_binary, pred_np)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        return float(thresholds[best_idx])

    elif criterion == "precision_recall_balance":
        precision, recall, thresholds = precision_recall_curve(gt_binary, pred_np)
        diff = np.abs(precision[:-1] - recall[:-1])
        best_idx = np.argmin(diff)
        return float(thresholds[best_idx])

    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def compute_detection_metrics(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: Optional[float] = None
) -> DetectionMetrics:
    """
    Compute all detection metrics.

    Args:
        predictions: Predicted scores
        ground_truth: Binary ground truth
        threshold: Decision threshold (computed optimally if None)

    Returns:
        DetectionMetrics with all metrics
    """
    # Compute optimal threshold if not provided
    if threshold is None:
        threshold = compute_optimal_threshold(predictions, ground_truth)

    pred_np = to_numpy(predictions)
    gt_np = to_numpy(ground_truth)
    gt_binary = (gt_np > 0.5).astype(np.int32)

    # AUC
    auc = compute_auc(predictions, ground_truth)

    # Precision, Recall, F1
    precision, recall = compute_precision_recall(predictions, ground_truth, threshold)
    f1 = compute_f1_score(predictions, ground_truth, threshold)

    # Confusion matrix
    cm = compute_confusion_matrix(predictions, ground_truth, threshold)
    tn, fp, fn, tp = cm.ravel()

    # Accuracy and Specificity
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    # Average Precision (area under PR curve)
    ap = average_precision_score(gt_binary, pred_np)

    return DetectionMetrics(
        auc=auc,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        specificity=specificity,
        threshold=threshold,
        average_precision=ap
    )


def compute_pixel_wise_metrics(
    detection_map: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute pixel-wise detection metrics.

    For spatial hallucination detection where we have
    pixel-level predictions and ground truth.

    Args:
        detection_map: Predicted detection map (B, 1, H, W)
        ground_truth_mask: Binary hallucination mask (B, 1, H, W)
        threshold: Detection threshold

    Returns:
        Dict with pixel-wise metrics
    """
    # Flatten spatial dimensions
    det_flat = detection_map.flatten()
    gt_flat = ground_truth_mask.flatten()

    # Binary predictions
    det_binary = (det_flat > threshold).float()
    gt_binary = (gt_flat > 0.5).float()

    # Metrics
    tp = ((det_binary == 1) & (gt_binary == 1)).sum().float()
    fp = ((det_binary == 1) & (gt_binary == 0)).sum().float()
    fn = ((det_binary == 0) & (gt_binary == 1)).sum().float()
    tn = ((det_binary == 0) & (gt_binary == 0)).sum().float()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    iou = (tp / (tp + fp + fn + 1e-8)).item()
    accuracy = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
    dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'accuracy': accuracy,
        'dice': dice,
    }


class DetectionAggregator:
    """
    Aggregates detection scores and ground truths for final evaluation.
    """

    def __init__(self):
        self.predictions = []
        self.ground_truths = []

    def add(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ):
        """Add predictions and ground truth for a sample."""
        self.predictions.append(to_numpy(prediction))
        self.ground_truths.append(to_numpy(ground_truth))

    def compute_metrics(
        self,
        threshold: Optional[float] = None
    ) -> DetectionMetrics:
        """Compute aggregate metrics."""
        all_preds = np.concatenate(self.predictions)
        all_gt = np.concatenate(self.ground_truths)

        return compute_detection_metrics(
            torch.from_numpy(all_preds),
            torch.from_numpy(all_gt),
            threshold
        )

    def reset(self):
        """Reset accumulated data."""
        self.predictions = []
        self.ground_truths = []
