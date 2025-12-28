"""
Implicit Neural Representation (INR) Training Utilities

Provides tools for fitting INRs to MRI images and sampling them.

WORKFLOW:
========
1. Given an MRI image, create coordinate grid
2. Train INR to map coordinates → intensities
3. Use trained INR for super-resolution, interpolation, or as canonical representation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, Callable
from tqdm import tqdm

from .siren import SIREN, SIRENConfig, create_coordinate_grid, sample_image_from_siren


class CoordinateDataset(Dataset):
    """
    Dataset of (coordinate, intensity) pairs from an image.

    Used for training INR on a single image.
    """

    def __init__(
        self,
        image: torch.Tensor,
        normalize_intensity: bool = True
    ):
        """
        Args:
            image: Image tensor (H, W) or (1, H, W) or (B, 1, H, W)
            normalize_intensity: Normalize intensity to [-1, 1]
        """
        # Handle different input shapes
        if image.dim() == 4:
            image = image[0, 0]  # (B, 1, H, W) → (H, W)
        elif image.dim() == 3:
            image = image[0]  # (1, H, W) → (H, W)

        self.H, self.W = image.shape
        self.image = image

        # Create coordinate grid
        self.coords = create_coordinate_grid(self.H, self.W, normalized=True)  # (H, W, 2)

        # Flatten
        self.coords_flat = self.coords.reshape(-1, 2)  # (H*W, 2)
        self.intensities_flat = image.reshape(-1, 1)  # (H*W, 1)

        # Normalize intensities
        if normalize_intensity:
            self.intensity_min = self.intensities_flat.min()
            self.intensity_max = self.intensities_flat.max()
            self.intensities_flat = 2 * (self.intensities_flat - self.intensity_min) / (
                self.intensity_max - self.intensity_min + 1e-8
            ) - 1
        else:
            self.intensity_min = 0
            self.intensity_max = 1

    def __len__(self) -> int:
        return self.H * self.W

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords_flat[idx], self.intensities_flat[idx]

    def denormalize_intensity(self, intensity: torch.Tensor) -> torch.Tensor:
        """Convert normalized intensity back to original range."""
        return (intensity + 1) / 2 * (self.intensity_max - self.intensity_min) + self.intensity_min


class INRTrainer:
    """
    Trainer for fitting INR to images.

    Handles training loop, sampling, and super-resolution.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[SIRENConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: INR model (creates SIREN if not provided)
            config: SIREN config (used if model not provided)
            device: Device to use
        """
        self.device = device

        if model is None:
            config = config or SIRENConfig()
            model = SIREN(config)

        self.model = model.to(device)
        self.intensity_min = 0
        self.intensity_max = 1

    def fit(
        self,
        image: torch.Tensor,
        num_steps: int = 1000,
        batch_size: int = 4096,
        lr: float = 1e-4,
        show_progress: bool = True,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Fit INR to an image.

        Args:
            image: Image tensor (H, W) or (1, H, W)
            num_steps: Training steps
            batch_size: Batch size for coordinate sampling
            lr: Learning rate
            show_progress: Show progress bar
            callback: Optional callback(step, loss, model)

        Returns:
            Dict with training history
        """
        # Create dataset
        dataset = CoordinateDataset(image.to(self.device))
        self.intensity_min = dataset.intensity_min.item()
        self.intensity_max = dataset.intensity_max.item()

        # Use all coordinates in random order
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training history
        history = {'loss': []}

        # Training loop
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Fitting INR")

        step = 0
        for epoch in iterator:
            for coords, targets in dataloader:
                coords = coords.to(self.device)
                targets = targets.to(self.device)

                # Forward
                pred = self.model(coords)
                loss = torch.mean((pred - targets) ** 2)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                history['loss'].append(loss.item())

                if callback is not None:
                    callback(step, loss.item(), self.model)

                step += 1

                if step >= num_steps:
                    break

            if step >= num_steps:
                break

        return history

    @torch.no_grad()
    def sample(
        self,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Sample image from trained INR.

        Args:
            height: Output height
            width: Output width

        Returns:
            Image tensor (H, W)
        """
        self.model.eval()
        coords = create_coordinate_grid(height, width, self.device)
        intensities = self.model(coords)

        # Denormalize
        intensities = (intensities + 1) / 2 * (self.intensity_max - self.intensity_min) + self.intensity_min

        return intensities.squeeze(-1)

    @torch.no_grad()
    def super_resolve(
        self,
        scale_factor: float = 2.0,
        original_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Generate super-resolved image.

        Args:
            scale_factor: Upsampling factor
            original_shape: Original (H, W) if known

        Returns:
            Super-resolved image
        """
        if original_shape is None:
            raise ValueError("original_shape required for super-resolution")

        H, W = original_shape
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)

        return self.sample(new_H, new_W)


def fit_inr_to_image(
    image: torch.Tensor,
    num_steps: int = 1000,
    hidden_features: int = 256,
    hidden_layers: int = 5,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[SIREN, Dict]:
    """
    Convenience function to fit INR to an image.

    Args:
        image: Image tensor
        num_steps: Training steps
        hidden_features: SIREN hidden dimension
        hidden_layers: SIREN depth
        lr: Learning rate
        device: Device

    Returns:
        Trained SIREN model, training history
    """
    config = SIRENConfig(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers
    )

    trainer = INRTrainer(config=config, device=device)
    history = trainer.fit(image, num_steps=num_steps, lr=lr)

    return trainer.model, history


def sample_inr_at_coordinates(
    model: nn.Module,
    coords: torch.Tensor,
    intensity_range: Optional[Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Sample INR at arbitrary coordinates.

    Args:
        model: Trained INR
        coords: Coordinates tensor (..., 2) in [-1, 1]
        intensity_range: (min, max) for denormalization

    Returns:
        Intensity values
    """
    model.eval()
    with torch.no_grad():
        intensities = model(coords)

    if intensity_range is not None:
        min_val, max_val = intensity_range
        intensities = (intensities + 1) / 2 * (max_val - min_val) + min_val

    return intensities


class INRComparator:
    """
    Compare images using their INR representations.

    Useful for the Guardian auditor - compare black-box output
    against physics-guided reference in continuous space.
    """

    def __init__(
        self,
        inr_config: Optional[SIRENConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config = inr_config or SIRENConfig()
        self.device = device

    def fit_both(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        num_steps: int = 500
    ) -> Tuple[SIREN, SIREN]:
        """
        Fit INRs to both images.

        Args:
            image1: First image
            image2: Second image
            num_steps: Training steps

        Returns:
            Two trained SIREN models
        """
        trainer1 = INRTrainer(config=self.config, device=self.device)
        trainer2 = INRTrainer(config=self.config, device=self.device)

        trainer1.fit(image1, num_steps=num_steps, show_progress=False)
        trainer2.fit(image2, num_steps=num_steps, show_progress=False)

        return trainer1.model, trainer2.model

    @torch.no_grad()
    def compute_discrepancy_map(
        self,
        inr1: SIREN,
        inr2: SIREN,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Compute discrepancy between two INR representations.

        Args:
            inr1: First INR (e.g., Guardian)
            inr2: Second INR (e.g., black-box)
            height: Output height
            width: Output width

        Returns:
            Discrepancy map (H, W)
        """
        coords = create_coordinate_grid(height, width, self.device)

        out1 = inr1(coords)
        out2 = inr2(coords)

        discrepancy = torch.abs(out1 - out2).squeeze(-1)

        return discrepancy

    @torch.no_grad()
    def compute_gradient_discrepancy(
        self,
        inr1: SIREN,
        inr2: SIREN,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Compare gradients (edge information) between INRs.

        More sensitive to structural differences.

        Returns:
            Gradient discrepancy map
        """
        coords = create_coordinate_grid(height, width, self.device).requires_grad_(True)

        # Get outputs and gradients
        out1, grad1 = inr1.forward_with_gradients(coords)
        out2, grad2 = inr2.forward_with_gradients(coords)

        # Gradient magnitude difference
        grad_mag1 = torch.sqrt(grad1[..., 0]**2 + grad1[..., 1]**2)
        grad_mag2 = torch.sqrt(grad2[..., 0]**2 + grad2[..., 1]**2)

        discrepancy = torch.abs(grad_mag1 - grad_mag2)

        return discrepancy
