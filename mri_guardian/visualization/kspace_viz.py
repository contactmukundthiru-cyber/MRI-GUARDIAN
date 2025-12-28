"""
K-Space Visualization Utilities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from .plotting import to_numpy


def visualize_kspace_magnitude(
    kspace: Union[torch.Tensor, np.ndarray],
    title: str = "K-space Magnitude",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize k-space magnitude with optional log scaling.

    Args:
        kspace: K-space data (complex or 2-channel)
        title: Figure title
        log_scale: Apply log scaling
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    k = to_numpy(kspace)

    # Convert to complex if needed
    if k.ndim == 4 and k.shape[1] == 2:  # (B, 2, H, W)
        k = k[0, 0] + 1j * k[0, 1]
    elif k.ndim == 3 and k.shape[0] == 2:  # (2, H, W)
        k = k[0] + 1j * k[1]
    elif k.ndim == 3:
        k = k[0]

    # Magnitude
    mag = np.abs(k)

    if log_scale:
        mag = np.log1p(mag)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mag, cmap='hot')
    ax.set_title(title, fontsize=14)
    ax.axis('off')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Log Magnitude' if log_scale else 'Magnitude')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_sampling_mask(
    mask: Union[torch.Tensor, np.ndarray],
    title: str = "Sampling Mask",
    show_percentage: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize k-space sampling mask.

    Args:
        mask: Sampling mask
        title: Figure title
        show_percentage: Show sampling percentage
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    m = to_numpy(mask)

    while m.ndim > 2:
        m = m.squeeze(0)

    # Calculate sampling percentage
    percentage = 100 * m.sum() / m.size

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(m, cmap='gray', vmin=0, vmax=1)

    if show_percentage:
        ax.set_title(f'{title}\nSampling: {percentage:.1f}%', fontsize=14)
    else:
        ax.set_title(title, fontsize=14)

    ax.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_kspace_difference(
    kspace1: Union[torch.Tensor, np.ndarray],
    kspace2: Union[torch.Tensor, np.ndarray],
    title: str = "K-space Difference",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize difference between two k-space data.

    Args:
        kspace1: First k-space
        kspace2: Second k-space
        title: Figure title
        log_scale: Apply log scaling
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    k1 = to_numpy(kspace1)
    k2 = to_numpy(kspace2)

    # Convert to complex
    def to_complex(k):
        if k.ndim == 4 and k.shape[1] == 2:
            return k[0, 0] + 1j * k[0, 1]
        elif k.ndim == 3 and k.shape[0] == 2:
            return k[0] + 1j * k[1]
        elif k.ndim == 3:
            return k[0]
        return k

    k1 = to_complex(k1)
    k2 = to_complex(k2)

    diff = np.abs(k1 - k2)

    if log_scale:
        diff = np.log1p(diff)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(diff, cmap='hot')
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_kspace_with_mask(
    kspace: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize full k-space, mask, and masked k-space side by side.

    Args:
        kspace: Full k-space data
        mask: Sampling mask
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    k = to_numpy(kspace)
    m = to_numpy(mask)

    # Convert to complex
    if k.ndim == 4 and k.shape[1] == 2:
        k = k[0, 0] + 1j * k[0, 1]
    elif k.ndim == 3 and k.shape[0] == 2:
        k = k[0] + 1j * k[1]
    elif k.ndim == 3:
        k = k[0]

    while m.ndim > 2:
        m = m.squeeze(0)

    # Masked k-space
    k_masked = k * m

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Full k-space
    axes[0].imshow(np.log1p(np.abs(k)), cmap='hot')
    axes[0].set_title('Full K-space', fontsize=12)
    axes[0].axis('off')

    # Mask
    axes[1].imshow(m, cmap='gray')
    axes[1].set_title(f'Mask ({100*m.mean():.1f}% sampled)', fontsize=12)
    axes[1].axis('off')

    # Masked k-space
    axes[2].imshow(np.log1p(np.abs(k_masked)), cmap='hot')
    axes[2].set_title('Masked K-space', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
