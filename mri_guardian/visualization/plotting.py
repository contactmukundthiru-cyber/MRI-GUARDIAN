"""
General Plotting Utilities for MRI-GUARDIAN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, List, Tuple, Union, Dict
from pathlib import Path


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_image(
    img: np.ndarray,
    percentile: float = 99
) -> np.ndarray:
    """Normalize image for display."""
    vmin = np.percentile(img, 100 - percentile)
    vmax = np.percentile(img, percentile)
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def plot_image(
    image: Union[torch.Tensor, np.ndarray],
    title: str = "",
    cmap: str = "gray",
    figsize: Tuple[int, int] = (6, 6),
    colorbar: bool = False,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot a single MRI image.

    Args:
        image: Image tensor/array (H, W) or (1, H, W) or (1, 1, H, W)
        title: Figure title
        cmap: Colormap
        figsize: Figure size
        colorbar: Show colorbar
        save_path: Path to save figure
        ax: Existing axes to plot on

    Returns:
        matplotlib Figure
    """
    img = to_numpy(image)

    # Remove extra dimensions
    while img.ndim > 2:
        img = img.squeeze(0)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=12)
    ax.axis('off')

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison(
    images: List[Union[torch.Tensor, np.ndarray]],
    titles: List[str],
    main_title: str = "",
    cmap: str = "gray",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple images side by side for comparison.

    Args:
        images: List of images to compare
        titles: List of titles for each image
        main_title: Overall figure title
        cmap: Colormap
        figsize: Figure size (auto if None)
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        img_np = to_numpy(img)
        while img_np.ndim > 2:
            img_np = img_np.squeeze(0)

        ax.imshow(img_np, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    if main_title:
        fig.suptitle(main_title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_kspace(
    kspace: Union[torch.Tensor, np.ndarray],
    title: str = "K-space",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot k-space magnitude.

    Args:
        kspace: K-space data (complex or 2-channel)
        title: Figure title
        log_scale: Use log scale for better visualization
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    k = to_numpy(kspace)

    # Handle different formats
    if k.ndim == 4 and k.shape[1] == 2:  # (B, 2, H, W)
        k = k[0, 0] + 1j * k[0, 1]
    elif k.ndim == 3:
        if k.shape[0] == 2:  # (2, H, W)
            k = k[0] + 1j * k[1]
        else:
            k = k[0]

    # Compute magnitude
    mag = np.abs(k)

    if log_scale:
        mag = np.log1p(mag)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(mag, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_metrics_bar(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str = "",
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar plot comparing methods.

    Args:
        metrics: Dict of {method_name: {metric_name: value, 'std': std}}
        metric_name: Name of metric to plot
        title: Figure title
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    methods = list(metrics.keys())
    values = [metrics[m].get(metric_name, 0) for m in methods]
    stds = [metrics[m].get('std', 0) for m in methods]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    bars = ax.bar(methods, values, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title if title else f'{metric_name} Comparison', fontsize=14)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves.

    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        val_metrics: Dict of validation metrics per epoch
        title: Figure title
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    n_plots = 1 + (val_metrics is not None and len(val_metrics) > 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Loss plot
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metrics plot
    if val_metrics and len(val_metrics) > 0:
        ax = axes[1]
        for name, values in val_metrics.items():
            ax.plot(range(1, len(values) + 1), values, label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('Validation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curves",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple methods.

    Args:
        roc_data: Dict of {method: (fpr, tpr, auc)}
        title: Figure title
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {auc_val:.3f})')

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    dpi: int = 150,
    formats: List[str] = ['png', 'pdf']
):
    """
    Save figure in multiple formats.

    Args:
        fig: matplotlib Figure
        path: Base path (without extension)
        dpi: Resolution
        formats: List of formats to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(str(path) + f'.{fmt}', dpi=dpi, bbox_inches='tight')
