"""
K-Space Sampling Patterns for MRI

This module implements various undersampling patterns used in accelerated MRI.

SAMPLING INTUITION:
==================
Full k-space acquisition is slow. To speed up scans, we skip some measurements.
But which points to skip matters a LOT for reconstruction quality.

Key principles:
1. ALWAYS sample the center (low frequencies = most image energy)
2. Sample more densely near center, sparser at edges
3. Avoid regular patterns (causes coherent aliasing)
4. Random/pseudo-random patterns work well with compressed sensing

COMMON PATTERNS:
===============
1. Cartesian: Skip entire lines (most common in practice)
   - Easy to implement on MRI hardware
   - Creates directional aliasing

2. Radial: Sample along spokes through center
   - More robust to motion
   - Incoherent aliasing

3. Spiral: Sample along spiral trajectories
   - Very efficient but complex
   - Sensitive to off-resonance

4. Variable Density: Dense center, sparse edges
   - Optimal for compressed sensing
   - Matches MRI signal distribution
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass


@dataclass
class SamplingMask:
    """
    Container for sampling mask with metadata.

    Attributes:
        mask: Binary mask tensor (H, W)
        acceleration: Achieved acceleration factor
        center_fraction: Fraction of center sampled
        pattern_type: Type of sampling pattern
    """
    mask: Union[torch.Tensor, np.ndarray]
    acceleration: float
    center_fraction: float
    pattern_type: str

    def to_tensor(self) -> torch.Tensor:
        """Convert mask to tensor."""
        if isinstance(self.mask, np.ndarray):
            return torch.from_numpy(self.mask).float()
        return self.mask.float()

    def to_numpy(self) -> np.ndarray:
        """Convert mask to numpy array."""
        if isinstance(self.mask, torch.Tensor):
            return self.mask.cpu().numpy()
        return self.mask


class CartesianMask(nn.Module):
    """
    Cartesian (line-by-line) undersampling mask generator.

    This is the most common pattern in clinical MRI.
    Samples entire k-space lines (columns), skips others.
    """

    def __init__(
        self,
        acceleration: int = 4,
        center_fraction: float = 0.08,
        pattern: str = "random"  # "random", "equispaced", "gaussian"
    ):
        """
        Args:
            acceleration: Target acceleration factor
            center_fraction: Fraction of center k-space to always sample
            pattern: Sampling pattern type
        """
        super().__init__()
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.pattern = pattern

    def forward(
        self,
        shape: Tuple[int, int],
        seed: Optional[int] = None
    ) -> SamplingMask:
        """
        Generate Cartesian sampling mask.

        Args:
            shape: (height, width) of k-space
            seed: Random seed for reproducibility

        Returns:
            SamplingMask object
        """
        h, w = shape
        rng = np.random.default_rng(seed)

        # Initialize mask (sample along columns)
        mask_1d = np.zeros(w, dtype=np.float32)

        # Always sample center
        num_center = max(1, int(w * self.center_fraction))
        center_start = (w - num_center) // 2
        mask_1d[center_start:center_start + num_center] = 1

        # Number of additional samples needed
        target_samples = max(1, int(w / self.acceleration))
        current_samples = int(mask_1d.sum())
        additional_needed = max(0, target_samples - current_samples)

        # Get indices of unsampled columns (excluding center)
        unsampled = np.where(mask_1d == 0)[0]

        if self.pattern == "random":
            # Uniform random sampling
            if len(unsampled) > 0 and additional_needed > 0:
                selected = rng.choice(
                    unsampled,
                    size=min(additional_needed, len(unsampled)),
                    replace=False
                )
                mask_1d[selected] = 1

        elif self.pattern == "equispaced":
            # Regular spacing
            if additional_needed > 0:
                step = max(1, len(unsampled) // additional_needed)
                selected = unsampled[::step][:additional_needed]
                mask_1d[selected] = 1

        elif self.pattern == "gaussian":
            # Variable density (denser near center)
            if len(unsampled) > 0 and additional_needed > 0:
                # Probability proportional to Gaussian centered at k-space center
                probs = np.exp(-0.5 * ((unsampled - w/2) / (w/4)) ** 2)
                probs = probs / probs.sum()
                selected = rng.choice(
                    unsampled,
                    size=min(additional_needed, len(unsampled)),
                    replace=False,
                    p=probs
                )
                mask_1d[selected] = 1

        # Expand to 2D (same mask for all rows)
        mask_2d = np.repeat(mask_1d[np.newaxis, :], h, axis=0)

        # Compute actual acceleration
        actual_accel = w / mask_1d.sum() if mask_1d.sum() > 0 else float('inf')

        return SamplingMask(
            mask=mask_2d,
            acceleration=actual_accel,
            center_fraction=num_center / w,
            pattern_type=f"cartesian_{self.pattern}"
        )


class RadialMask(nn.Module):
    """
    Radial undersampling mask generator.

    Samples k-space along spokes through the center.
    Common in cardiac and real-time MRI.
    """

    def __init__(
        self,
        num_spokes: int = 32,
        golden_angle: bool = True
    ):
        """
        Args:
            num_spokes: Number of radial spokes
            golden_angle: Use golden angle spacing (more uniform coverage)
        """
        super().__init__()
        self.num_spokes = num_spokes
        self.golden_angle = golden_angle
        # Golden angle in radians (≈111.246°)
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.golden_angle_rad = np.pi / self.golden_ratio

    def forward(
        self,
        shape: Tuple[int, int],
        seed: Optional[int] = None
    ) -> SamplingMask:
        """
        Generate radial sampling mask.

        Args:
            shape: (height, width) of k-space
            seed: Random seed (used for starting angle if not golden)

        Returns:
            SamplingMask object
        """
        h, w = shape
        rng = np.random.default_rng(seed)

        mask = np.zeros((h, w), dtype=np.float32)

        # Center of k-space
        cy, cx = h // 2, w // 2

        # Maximum radius
        max_radius = min(h, w) // 2

        # Generate spoke angles
        if self.golden_angle:
            angles = np.array([i * self.golden_angle_rad for i in range(self.num_spokes)])
        else:
            start_angle = rng.uniform(0, np.pi / self.num_spokes)
            angles = np.linspace(start_angle, start_angle + np.pi, self.num_spokes, endpoint=False)

        # Draw each spoke
        for angle in angles:
            for r in range(max_radius):
                y = int(cy + r * np.sin(angle))
                x = int(cx + r * np.cos(angle))
                if 0 <= y < h and 0 <= x < w:
                    mask[y, x] = 1
                # Also draw negative direction
                y = int(cy - r * np.sin(angle))
                x = int(cx - r * np.cos(angle))
                if 0 <= y < h and 0 <= x < w:
                    mask[y, x] = 1

        # Compute acceleration
        actual_accel = (h * w) / mask.sum() if mask.sum() > 0 else float('inf')

        return SamplingMask(
            mask=mask,
            acceleration=actual_accel,
            center_fraction=1.0,  # Radial always samples center
            pattern_type="radial_golden" if self.golden_angle else "radial_uniform"
        )


class SpiralMask(nn.Module):
    """
    Spiral undersampling mask generator.

    Samples k-space along Archimedean spiral trajectories.
    Very efficient but sensitive to system imperfections.
    """

    def __init__(
        self,
        num_interleaves: int = 4,
        turns: int = 24
    ):
        """
        Args:
            num_interleaves: Number of spiral arms
            turns: Number of turns in each spiral
        """
        super().__init__()
        self.num_interleaves = num_interleaves
        self.turns = turns

    def forward(
        self,
        shape: Tuple[int, int],
        seed: Optional[int] = None
    ) -> SamplingMask:
        """
        Generate spiral sampling mask.

        Args:
            shape: (height, width) of k-space
            seed: Random seed for starting angle

        Returns:
            SamplingMask object
        """
        h, w = shape
        rng = np.random.default_rng(seed)

        mask = np.zeros((h, w), dtype=np.float32)

        cy, cx = h // 2, w // 2
        max_radius = min(h, w) // 2

        # Archimedean spiral: r = a * theta
        # a chosen so r reaches max_radius after 'turns' rotations
        a = max_radius / (2 * np.pi * self.turns)

        for interleave in range(self.num_interleaves):
            # Starting angle for this interleave
            theta_offset = 2 * np.pi * interleave / self.num_interleaves

            # Sample along spiral
            num_points = self.turns * 100  # Points per spiral
            thetas = np.linspace(0, 2 * np.pi * self.turns, num_points)

            for theta in thetas:
                r = a * theta
                if r > max_radius:
                    break

                angle = theta + theta_offset
                y = int(cy + r * np.sin(angle))
                x = int(cx + r * np.cos(angle))

                if 0 <= y < h and 0 <= x < w:
                    mask[y, x] = 1

        # Compute acceleration
        actual_accel = (h * w) / mask.sum() if mask.sum() > 0 else float('inf')

        return SamplingMask(
            mask=mask,
            acceleration=actual_accel,
            center_fraction=1.0,
            pattern_type="spiral"
        )


class VariableDensityMask(nn.Module):
    """
    Variable Density Sampling Mask.

    Samples with probability inversely related to distance from center.
    Optimal for compressed sensing recovery.
    """

    def __init__(
        self,
        acceleration: int = 4,
        center_fraction: float = 0.1,
        power: float = 2.0  # Higher = more concentration at center
    ):
        """
        Args:
            acceleration: Target acceleration factor
            center_fraction: Fraction of center to fully sample
            power: Power law exponent for density falloff
        """
        super().__init__()
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.power = power

    def forward(
        self,
        shape: Tuple[int, int],
        seed: Optional[int] = None
    ) -> SamplingMask:
        """
        Generate variable density mask.

        Args:
            shape: (height, width) of k-space
            seed: Random seed

        Returns:
            SamplingMask object
        """
        h, w = shape
        rng = np.random.default_rng(seed)

        # Create coordinate grids
        y = np.arange(h) - h // 2
        x = np.arange(w) - w // 2
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Normalized distance from center
        R = np.sqrt((Y / (h/2)) ** 2 + (X / (w/2)) ** 2)

        # Sampling probability: higher near center
        prob = (1 - np.minimum(R, 1)) ** self.power
        prob = prob / prob.sum()  # Normalize

        # Number of samples
        num_samples = int(h * w / self.acceleration)

        # Flatten and sample
        flat_prob = prob.flatten()
        indices = rng.choice(
            h * w,
            size=num_samples,
            replace=False,
            p=flat_prob
        )

        # Create mask
        mask = np.zeros((h, w), dtype=np.float32)
        mask.flat[indices] = 1

        # Ensure center is fully sampled
        center_h = int(h * self.center_fraction)
        center_w = int(w * self.center_fraction)
        h_start = (h - center_h) // 2
        w_start = (w - center_w) // 2
        mask[h_start:h_start+center_h, w_start:w_start+center_w] = 1

        # Compute actual acceleration
        actual_accel = (h * w) / mask.sum() if mask.sum() > 0 else float('inf')

        return SamplingMask(
            mask=mask,
            acceleration=actual_accel,
            center_fraction=self.center_fraction,
            pattern_type="variable_density"
        )


def create_mask_from_acceleration(
    shape: Tuple[int, int],
    acceleration: int,
    pattern: str = "random",
    center_fraction: float = 0.08,
    seed: Optional[int] = None
) -> SamplingMask:
    """
    Convenience function to create sampling mask.

    Args:
        shape: (height, width) of k-space
        acceleration: Target acceleration factor
        pattern: "random", "equispaced", "gaussian", "radial", "spiral", "variable_density"
        center_fraction: Fraction of center to sample
        seed: Random seed

    Returns:
        SamplingMask object
    """
    if pattern in ["random", "equispaced", "gaussian"]:
        generator = CartesianMask(acceleration, center_fraction, pattern)
    elif pattern == "radial":
        num_spokes = max(1, int(shape[0] / acceleration))
        generator = RadialMask(num_spokes)
    elif pattern == "spiral":
        num_interleaves = max(1, int(4 / acceleration))
        generator = SpiralMask(num_interleaves)
    elif pattern == "variable_density":
        generator = VariableDensityMask(acceleration, center_fraction)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return generator(shape, seed)


def create_mask_batch(
    batch_size: int,
    shape: Tuple[int, int],
    acceleration: int,
    pattern: str = "random",
    center_fraction: float = 0.08,
    same_mask: bool = False,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create a batch of sampling masks.

    Args:
        batch_size: Number of masks
        shape: (height, width) of k-space
        acceleration: Target acceleration factor
        pattern: Sampling pattern type
        center_fraction: Fraction of center to sample
        same_mask: If True, use same mask for all samples
        seed: Random seed

    Returns:
        Tensor of shape (batch_size, 1, height, width)
    """
    masks = []

    for i in range(batch_size):
        if same_mask and i > 0:
            masks.append(masks[0])
        else:
            mask_seed = seed + i if seed is not None else None
            mask_obj = create_mask_from_acceleration(
                shape, acceleration, pattern, center_fraction, mask_seed
            )
            masks.append(mask_obj.to_tensor())

    return torch.stack(masks, dim=0).unsqueeze(1)
