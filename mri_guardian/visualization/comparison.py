"""
Comparison and Publication-Quality Figures
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, Union, List, Dict
from pathlib import Path
from .plotting import to_numpy, normalize_image


def create_comparison_figure(
    ground_truth: Union[torch.Tensor, np.ndarray],
    reconstructions: Dict[str, Union[torch.Tensor, np.ndarray]],
    error_maps: bool = True,
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create publication-quality reconstruction comparison figure.

    Args:
        ground_truth: Ground truth image
        reconstructions: Dict of {method_name: reconstruction}
        error_maps: Show error maps
        metrics: Dict of {method_name: {metric: value}}
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    n_methods = len(reconstructions)
    n_rows = 2 if error_maps else 1
    n_cols = n_methods + 1  # +1 for ground truth

    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.1, wspace=0.1)

    gt = to_numpy(ground_truth)
    while gt.ndim > 2:
        gt = gt.squeeze(0)
    gt = normalize_image(gt)

    # Ground truth
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(gt, cmap='gray')
    ax.set_title('Ground Truth', fontsize=10)
    ax.axis('off')

    if error_maps:
        ax = fig.add_subplot(gs[1, 0])
        ax.axis('off')

    # Reconstructions
    for i, (name, recon) in enumerate(reconstructions.items(), 1):
        r = to_numpy(recon)
        while r.ndim > 2:
            r = r.squeeze(0)
        r = normalize_image(r)

        # Reconstruction
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(r, cmap='gray')

        # Title with metrics if available
        if metrics and name in metrics:
            m = metrics[name]
            title = f"{name}\nPSNR: {m.get('psnr', 0):.2f} SSIM: {m.get('ssim', 0):.3f}"
        else:
            title = name
        ax.set_title(title, fontsize=9)
        ax.axis('off')

        # Error map
        if error_maps:
            error = np.abs(gt - r)
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.3)
            ax.set_title(f'Error (×3)', fontsize=9)
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_hallucination_detection_figure(
    original: Union[torch.Tensor, np.ndarray],
    hallucinated: Union[torch.Tensor, np.ndarray],
    guardian_recon: Union[torch.Tensor, np.ndarray],
    detection_map: Union[torch.Tensor, np.ndarray],
    ground_truth_mask: Union[torch.Tensor, np.ndarray],
    metrics: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (18, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create figure showing hallucination detection results.

    Args:
        original: Clean reconstruction
        hallucinated: Reconstruction with hallucinations
        guardian_recon: Guardian model reconstruction
        detection_map: Predicted detection map
        ground_truth_mask: True hallucination locations
        metrics: Detection metrics
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, hspace=0.2, wspace=0.1)

    # Convert all to numpy
    orig = to_numpy(original)
    hall = to_numpy(hallucinated)
    guard = to_numpy(guardian_recon)
    det = to_numpy(detection_map)
    gt_mask = to_numpy(ground_truth_mask)

    # Squeeze dimensions
    for arr in [orig, hall, guard, det, gt_mask]:
        while arr.ndim > 2:
            arr = arr.squeeze(0)

    orig = normalize_image(to_numpy(original).squeeze())
    hall = normalize_image(to_numpy(hallucinated).squeeze())
    guard = normalize_image(to_numpy(guardian_recon).squeeze())
    det = to_numpy(detection_map).squeeze()
    gt_mask = to_numpy(ground_truth_mask).squeeze()

    # Row 1: Images
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(orig, cmap='gray')
    ax.set_title('Original', fontsize=12)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(hall, cmap='gray')
    ax.set_title('Hallucinated (Black-Box)', fontsize=12)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(guard, cmap='gray')
    ax.set_title('Guardian Reconstruction', fontsize=12)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 3])
    diff = np.abs(hall - guard)
    ax.imshow(diff, cmap='hot')
    ax.set_title('|Black-Box - Guardian|', fontsize=12)
    ax.axis('off')

    # Row 2: Detection
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(gt_mask, cmap='Reds', vmin=0, vmax=1)
    ax.set_title('Ground Truth Mask', fontsize=12)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    det_norm = (det - det.min()) / (det.max() - det.min() + 1e-8)
    ax.imshow(det_norm, cmap='Reds', vmin=0, vmax=1)
    ax.set_title('Detection Map', fontsize=12)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    detection_binary = (det_norm > 0.5).astype(float)
    ax.imshow(detection_binary, cmap='Reds', vmin=0, vmax=1)
    ax.set_title('Detection (Thresholded)', fontsize=12)
    ax.axis('off')

    # Overlay
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(hall, cmap='gray')
    # Overlay detections in red, ground truth in green
    overlay = np.zeros((*hall.shape, 3))
    overlay[..., 0] = detection_binary * 0.5  # Red: detected
    overlay[..., 1] = gt_mask * 0.5  # Green: ground truth
    ax.imshow(overlay, alpha=0.5)
    ax.set_title('Overlay (Red=Det, Green=GT)', fontsize=12)
    ax.axis('off')

    # Add metrics text if provided
    if metrics:
        text = f"AUC: {metrics.get('auc', 0):.3f}  |  F1: {metrics.get('f1', 0):.3f}  |  Precision: {metrics.get('precision', 0):.3f}  |  Recall: {metrics.get('recall', 0):.3f}"
        fig.text(0.5, 0.02, text, ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_robustness_figure(
    accelerations: List[int],
    metrics_by_accel: Dict[int, Dict[str, Dict[str, float]]],
    metric_names: List[str] = ['psnr', 'ssim'],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create robustness study figure showing performance vs acceleration.

    Args:
        accelerations: List of acceleration factors
        metrics_by_accel: {accel: {method: {metric: value}}}
        metric_names: Which metrics to plot
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, metric in zip(axes, metric_names):
        # Get method names from first acceleration
        methods = list(metrics_by_accel[accelerations[0]].keys())

        for i, method in enumerate(methods):
            values = []
            stds = []
            for acc in accelerations:
                m = metrics_by_accel[acc][method]
                values.append(m.get(metric, 0))
                stds.append(m.get(f'{metric}_std', 0))

            ax.errorbar(accelerations, values, yerr=stds, marker='o',
                        label=method, color=colors[i], linewidth=2, capsize=3)

        ax.set_xlabel('Acceleration Factor', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} vs Acceleration', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(accelerations)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_poster_summary_figure(
    results: Dict,
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create summary figure suitable for ISEF poster.

    Args:
        results: Dict containing all experiment results
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('MRI-GUARDIAN: Physics-Guided Hallucination Detection',
                 fontsize=16, fontweight='bold', y=0.98)

    # Section 1: Example reconstruction (top row)
    if 'example_images' in results:
        imgs = results['example_images']
        for i, (name, img) in enumerate(imgs.items()):
            if i >= 4:
                break
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(to_numpy(img).squeeze(), cmap='gray')
            ax.set_title(name, fontsize=10)
            ax.axis('off')

    # Section 2: Metrics comparison (middle left)
    if 'reconstruction_metrics' in results:
        ax = fig.add_subplot(gs[1, 0:2])
        metrics = results['reconstruction_metrics']
        methods = list(metrics.keys())
        x = np.arange(len(methods))
        width = 0.35

        psnr_vals = [metrics[m]['psnr'] for m in methods]
        ssim_vals = [metrics[m]['ssim'] for m in methods]

        ax.bar(x - width/2, psnr_vals, width, label='PSNR', color='steelblue')
        ax2 = ax.twinx()
        ax2.bar(x + width/2, ssim_vals, width, label='SSIM', color='coral')

        ax.set_ylabel('PSNR (dB)', color='steelblue')
        ax2.set_ylabel('SSIM', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title('Reconstruction Quality', fontsize=12)

    # Section 3: ROC curve (middle right)
    if 'roc_curves' in results:
        ax = fig.add_subplot(gs[1, 2:4])
        for name, (fpr, tpr, auc) in results['roc_curves'].items():
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Hallucination Detection ROC', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    # Section 4: Robustness (bottom row)
    if 'robustness' in results:
        ax = fig.add_subplot(gs[2, :])
        rob = results['robustness']
        accels = sorted(rob.keys())
        for method in rob[accels[0]].keys():
            psnr_vals = [rob[a][method]['psnr'] for a in accels]
            ax.plot(accels, psnr_vals, marker='o', linewidth=2, label=method)
        ax.set_xlabel('Acceleration Factor')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Reconstruction Quality vs Acceleration', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
