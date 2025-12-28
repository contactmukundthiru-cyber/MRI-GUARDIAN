"""
Hallucination Injection Module

Re-exports hallucination utilities from the blackbox model
plus additional specialized generation functions.
"""

# Re-export from blackbox
from ..models.blackbox import HallucinationInjector, HallucinationConfig

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def generate_synthetic_lesions(
    image: torch.Tensor,
    num_lesions: int = 3,
    size_range: Tuple[int, int] = (5, 15),
    intensity_range: Tuple[float, float] = (0.3, 0.8),
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic lesions for testing.

    Creates elliptical bright spots that simulate pathology.
    Returns both the modified image and ground truth mask.

    Args:
        image: Input image (B, 1, H, W)
        num_lesions: Number of lesions to add
        size_range: (min, max) size in pixels
        intensity_range: (min, max) relative intensity
        seed: Random seed

    Returns:
        modified_image: Image with lesions
        lesion_mask: Binary mask of lesion locations
    """
    if seed is not None:
        np.random.seed(seed)

    B, C, H, W = image.shape
    device = image.device

    output = image.clone()
    mask = torch.zeros(B, 1, H, W, device=device)

    for b in range(B):
        for _ in range(num_lesions):
            # Random position
            margin = size_range[1]
            cy = np.random.randint(margin, H - margin)
            cx = np.random.randint(margin, W - margin)

            # Random size
            size = np.random.randint(*size_range)
            a = size / 2 * np.random.uniform(0.7, 1.3)
            b_axis = size / 2 * np.random.uniform(0.7, 1.3)

            # Random intensity
            local_mean = output[b, 0, max(0, cy-10):min(H, cy+10),
                                max(0, cx-10):min(W, cx+10)].mean()
            intensity = local_mean * np.random.uniform(*intensity_range)

            # Create elliptical mask
            y, x = torch.meshgrid(
                torch.arange(H, device=device).float(),
                torch.arange(W, device=device).float(),
                indexing='ij'
            )

            dist = ((x - cx) / a) ** 2 + ((y - cy) / b_axis) ** 2
            lesion = (dist <= 1).float()

            # Smooth edges
            lesion = F.avg_pool2d(lesion.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze()

            # Add lesion
            output[b, 0] = output[b, 0] + intensity * lesion
            mask[b, 0] = torch.maximum(mask[b, 0], lesion)

    return output, mask


def generate_texture_artifacts(
    image: torch.Tensor,
    num_patches: int = 2,
    patch_size: int = 32,
    strength: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate texture hallucination artifacts.

    Adds high-frequency noise patterns to local regions.

    Args:
        image: Input image (B, 1, H, W)
        num_patches: Number of patches to affect
        patch_size: Size of each patch
        strength: Noise strength
        seed: Random seed

    Returns:
        modified_image: Image with texture artifacts
        artifact_mask: Binary mask of affected regions
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    B, C, H, W = image.shape
    device = image.device

    output = image.clone()
    mask = torch.zeros(B, 1, H, W, device=device)

    for b in range(B):
        for _ in range(num_patches):
            # Random position
            py = np.random.randint(0, H - patch_size)
            px = np.random.randint(0, W - patch_size)

            # Generate high-frequency noise
            noise = torch.randn(patch_size, patch_size, device=device)

            # Scale by local intensity
            local_std = output[b, 0, py:py+patch_size, px:px+patch_size].std()
            noise = noise * strength * local_std

            # Apply with soft edges
            edge_mask = torch.ones(patch_size, patch_size, device=device)
            margin = patch_size // 4
            for i in range(margin):
                factor = i / margin
                edge_mask[i, :] *= factor
                edge_mask[-(i+1), :] *= factor
                edge_mask[:, i] *= factor
                edge_mask[:, -(i+1)] *= factor

            output[b, 0, py:py+patch_size, px:px+patch_size] += noise * edge_mask
            mask[b, 0, py:py+patch_size, px:px+patch_size] = edge_mask

    return output, mask


def generate_edge_artifacts(
    image: torch.Tensor,
    num_edges: int = 3,
    length_range: Tuple[int, int] = (20, 50),
    thickness: int = 2,
    intensity: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate false edge artifacts.

    Creates artificial linear structures that don't exist in the original.

    Args:
        image: Input image (B, 1, H, W)
        num_edges: Number of edges to add
        length_range: (min, max) edge length
        thickness: Edge thickness in pixels
        intensity: Edge intensity relative to local region
        seed: Random seed

    Returns:
        modified_image: Image with edge artifacts
        artifact_mask: Binary mask of edge locations
    """
    if seed is not None:
        np.random.seed(seed)

    B, C, H, W = image.shape
    device = image.device

    output = image.clone()
    mask = torch.zeros(B, 1, H, W, device=device)

    for b in range(B):
        for _ in range(num_edges):
            # Random start point
            y1 = np.random.randint(20, H - 20)
            x1 = np.random.randint(20, W - 20)

            # Random angle and length
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.randint(*length_range)

            # End point
            y2 = int(y1 + length * np.sin(angle))
            x2 = int(x1 + length * np.cos(angle))

            # Clip to image bounds
            y2 = np.clip(y2, 0, H - 1)
            x2 = np.clip(x2, 0, W - 1)

            # Draw line (Bresenham's algorithm simplified)
            num_points = max(abs(y2 - y1), abs(x2 - x1)) + 1
            ys = np.linspace(y1, y2, num_points).astype(int)
            xs = np.linspace(x1, x2, num_points).astype(int)

            # Get local intensity
            local_mean = output[b, 0, max(0, y1-10):min(H, y1+10),
                                max(0, x1-10):min(W, x1+10)].mean()
            edge_intensity = local_mean * intensity

            # Add edge with thickness
            for y, x in zip(ys, xs):
                for dy in range(-thickness//2, thickness//2 + 1):
                    for dx in range(-thickness//2, thickness//2 + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            output[b, 0, ny, nx] = output[b, 0, ny, nx] + edge_intensity
                            mask[b, 0, ny, nx] = 1.0

    return output, mask


def generate_mixed_hallucinations(
    image: torch.Tensor,
    lesion_prob: float = 0.5,
    texture_prob: float = 0.3,
    edge_prob: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate mixed hallucination types.

    Randomly selects and applies different hallucination types.

    Args:
        image: Input image
        lesion_prob: Probability of adding lesions
        texture_prob: Probability of adding texture
        edge_prob: Probability of adding edges
        seed: Random seed

    Returns:
        modified_image: Image with hallucinations
        combined_mask: Combined hallucination mask
        info: Dict with which hallucinations were applied
    """
    if seed is not None:
        np.random.seed(seed)

    output = image.clone()
    mask = torch.zeros_like(image)
    info = {'lesion': False, 'texture': False, 'edge': False}

    if np.random.random() < lesion_prob:
        output, lesion_mask = generate_synthetic_lesions(output, seed=seed)
        mask = torch.maximum(mask, lesion_mask)
        info['lesion'] = True

    if np.random.random() < texture_prob:
        output, texture_mask = generate_texture_artifacts(output, seed=seed)
        mask = torch.maximum(mask, texture_mask)
        info['texture'] = True

    if np.random.random() < edge_prob:
        output, edge_mask = generate_edge_artifacts(output, seed=seed)
        mask = torch.maximum(mask, edge_mask)
        info['edge'] = True

    return output, mask, info
