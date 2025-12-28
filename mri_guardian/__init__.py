"""
MRI-GUARDIAN: Physics-Guided Generative MRI Reconstruction and Hallucination Auditor

A deep learning framework for trustworthy MRI reconstruction with built-in
hallucination detection capabilities.

Main Components:
- data: Data loading and preprocessing for fastMRI
- physics: MRI physics operations and data consistency
- models: Neural network architectures (UNet, Guardian, Diffusion)
- implicit: Implicit neural representations (SIREN)
- auditor: Hallucination detection system
- metrics: Evaluation metrics (PSNR, SSIM, AUC, etc.)
- visualization: Plotting and comparison utilities
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import data
from . import physics
from . import models
from . import implicit
from . import auditor
from . import metrics
from . import visualization

__all__ = [
    "data",
    "physics",
    "models",
    "implicit",
    "auditor",
    "metrics",
    "visualization",
]
