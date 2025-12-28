"""
Fourier Feature Mapping for Neural Networks

Standard neural networks have "spectral bias" - they learn
low-frequency patterns easily but struggle with high-frequency details.

FOURIER FEATURES solve this by mapping low-dimensional inputs
to a higher-dimensional space using sinusoidal functions.

INTUITION:
=========
Imagine trying to draw a complex pattern with only one crayon.
Fourier features give you many crayons of different "frequencies",
making it easier to draw fine details.

Input: (x, y) → simple 2D coordinate
After mapping: (sin(2πf₁x), cos(2πf₁x), sin(2πf₂x), ...) → rich representation

This is CRITICAL for learning high-frequency MRI details.

REFERENCE:
Tancik et al., "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains", NeurIPS 2020
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class FourierFeatureMapping(nn.Module):
    """
    Deterministic Fourier feature mapping.

    Maps inputs through a fixed set of sinusoidal functions
    at different frequencies.

    Output: [sin(2π B x), cos(2π B x)]
    Where B is a matrix of frequencies.
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 64,
        max_frequency: float = 10.0,
        log_scale: bool = True
    ):
        """
        Args:
            in_features: Input dimension (2 for 2D coordinates)
            num_frequencies: Number of frequency bands
            max_frequency: Maximum frequency
            log_scale: Use log-spaced frequencies
        """
        super().__init__()

        if log_scale:
            # Log-spaced frequencies (more at low end)
            frequencies = 2.0 ** torch.linspace(0, np.log2(max_frequency), num_frequencies)
        else:
            # Linear-spaced frequencies
            frequencies = torch.linspace(1.0, max_frequency, num_frequencies)

        # Create frequency matrix (num_frequencies, in_features)
        # Each row is a frequency vector
        freq_matrix = torch.zeros(num_frequencies, in_features)
        for i in range(num_frequencies):
            # Alternate between x and y directions
            freq_matrix[i, i % in_features] = frequencies[i]

        self.register_buffer("freq_matrix", freq_matrix)
        self.out_features = num_frequencies * 2  # Sin and cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map coordinates to Fourier features.

        Args:
            x: Input coordinates (..., in_features)

        Returns:
            Fourier features (..., num_frequencies * 2)
        """
        # x @ freq_matrix.T gives us the dot products with frequency vectors
        proj = 2 * np.pi * torch.matmul(x, self.freq_matrix.T)

        # Concatenate sin and cos
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class GaussianFourierFeatures(nn.Module):
    """
    Random Fourier Features with Gaussian initialization.

    Uses random frequency vectors sampled from a Gaussian distribution.
    The scale parameter controls the "bandwidth" of frequencies.

    Higher scale → higher frequencies → more detail, but harder to train
    Lower scale → lower frequencies → smoother, easier to train
    """

    def __init__(
        self,
        in_features: int,
        mapping_size: int = 256,
        scale: float = 10.0,
        learnable: bool = False
    ):
        """
        Args:
            in_features: Input dimension
            mapping_size: Number of random features
            scale: Standard deviation of Gaussian (controls frequency range)
            learnable: If True, frequencies are learnable parameters
        """
        super().__init__()

        self.mapping_size = mapping_size
        self.scale = scale
        self.out_features = mapping_size * 2

        # Random frequency matrix
        B = torch.randn(mapping_size, in_features) * scale

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random Fourier feature mapping.

        Args:
            x: Input (..., in_features)

        Returns:
            Fourier features (..., mapping_size * 2)
        """
        proj = 2 * np.pi * torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding (NeRF-style).

    The original positional encoding from NeRF.
    Uses fixed frequencies in powers of 2.

    γ(x) = [sin(2⁰πx), cos(2⁰πx), sin(2¹πx), cos(2¹πx), ..., sin(2^(L-1)πx), cos(2^(L-1)πx)]
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 10,
        include_input: bool = True
    ):
        """
        Args:
            in_features: Input dimension
            num_frequencies: Number of frequency octaves
            include_input: Also output the raw input
        """
        super().__init__()

        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.in_features = in_features

        # Frequencies: 2^0, 2^1, ..., 2^(L-1)
        frequencies = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer("frequencies", frequencies)

        # Output dimension
        self.out_features = in_features * num_frequencies * 2
        if include_input:
            self.out_features += in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.

        Args:
            x: Input (..., in_features)

        Returns:
            Encoded (..., out_features)
        """
        outputs = []

        if self.include_input:
            outputs.append(x)

        for freq in self.frequencies:
            outputs.append(torch.sin(np.pi * freq * x))
            outputs.append(torch.cos(np.pi * freq * x))

        return torch.cat(outputs, dim=-1)


class HybridFourierFeatures(nn.Module):
    """
    Hybrid Fourier Features combining deterministic and random components.

    Uses both:
    1. Low deterministic frequencies (stable, predictable)
    2. High random frequencies (flexible, detailed)

    Good for MRI where we want both smooth anatomy and fine details.
    """

    def __init__(
        self,
        in_features: int,
        num_deterministic: int = 32,
        num_random: int = 128,
        max_det_freq: float = 5.0,
        random_scale: float = 20.0
    ):
        """
        Args:
            in_features: Input dimension
            num_deterministic: Number of deterministic frequencies
            num_random: Number of random frequencies
            max_det_freq: Maximum deterministic frequency
            random_scale: Scale for random frequencies
        """
        super().__init__()

        self.det_features = FourierFeatureMapping(
            in_features, num_deterministic, max_det_freq
        )

        self.rand_features = GaussianFourierFeatures(
            in_features, num_random, random_scale
        )

        self.out_features = self.det_features.out_features + self.rand_features.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine deterministic and random features."""
        det = self.det_features(x)
        rand = self.rand_features(x)
        return torch.cat([det, rand], dim=-1)


class AdaptiveFourierFeatures(nn.Module):
    """
    Adaptive Fourier Features with learnable weights.

    Learns to weight different frequency components
    based on what's important for the specific image.
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 128,
        max_frequency: float = 32.0
    ):
        """
        Args:
            in_features: Input dimension
            num_frequencies: Total frequencies
            max_frequency: Maximum frequency
        """
        super().__init__()

        self.num_frequencies = num_frequencies

        # Learnable frequencies
        init_freqs = torch.rand(num_frequencies, in_features) * max_frequency
        self.frequencies = nn.Parameter(init_freqs)

        # Learnable phase shifts
        self.phases = nn.Parameter(torch.zeros(num_frequencies))

        # Learnable weights for each frequency
        self.weights = nn.Parameter(torch.ones(num_frequencies * 2))

        self.out_features = num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive Fourier features."""
        # Project input through learnable frequencies
        proj = 2 * np.pi * torch.matmul(x, self.frequencies.T)

        # Add phase shifts
        proj = proj + self.phases

        # Compute sin and cos
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        # Apply learnable weights (softmax for stability)
        weights = torch.softmax(self.weights, dim=0)
        features = features * weights

        return features


def get_fourier_features(
    feature_type: str = "gaussian",
    in_features: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function for Fourier feature modules.

    Args:
        feature_type: Type of features
            - "gaussian": Random Gaussian features
            - "deterministic": Fixed frequency grid
            - "positional": NeRF-style positional encoding
            - "hybrid": Combination of deterministic + random
            - "adaptive": Learnable frequencies
        in_features: Input dimension
        **kwargs: Additional arguments for specific types

    Returns:
        Fourier feature module
    """
    if feature_type == "gaussian":
        return GaussianFourierFeatures(
            in_features,
            kwargs.get("mapping_size", 256),
            kwargs.get("scale", 10.0)
        )
    elif feature_type == "deterministic":
        return FourierFeatureMapping(
            in_features,
            kwargs.get("num_frequencies", 64),
            kwargs.get("max_frequency", 10.0)
        )
    elif feature_type == "positional":
        return PositionalEncoding(
            in_features,
            kwargs.get("num_frequencies", 10),
            kwargs.get("include_input", True)
        )
    elif feature_type == "hybrid":
        return HybridFourierFeatures(
            in_features,
            kwargs.get("num_deterministic", 32),
            kwargs.get("num_random", 128)
        )
    elif feature_type == "adaptive":
        return AdaptiveFourierFeatures(
            in_features,
            kwargs.get("num_frequencies", 128)
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
