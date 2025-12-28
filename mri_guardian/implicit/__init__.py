"""
Implicit Neural Representations for MRI

This module implements coordinate-based neural networks (INRs)
for continuous MRI representation.
"""

from .siren import SIREN, SineLayer, SIRENConfig
from .fourier_features import FourierFeatureMapping, GaussianFourierFeatures
from .inr_trainer import INRTrainer, fit_inr_to_image, sample_inr_at_coordinates

__all__ = [
    "SIREN",
    "SineLayer",
    "SIRENConfig",
    "FourierFeatureMapping",
    "GaussianFourierFeatures",
    "INRTrainer",
    "fit_inr_to_image",
    "sample_inr_at_coordinates",
]
