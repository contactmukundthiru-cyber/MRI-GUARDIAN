"""
Neural Network Models for MRI-GUARDIAN

This module contains all neural network architectures:
- UNet: Standard image-domain baseline
- Guardian: Physics-guided dual-domain model
- DualDomain: Combined k-space and image processing
- Diffusion: Score-based refinement model
- BlackBox: Model for testing hallucination detection
- BiologicalPriors: Disease-aware biological plausibility constraints
"""

from .unet import UNet, UNetEncoder, UNetDecoder
from .guardian import GuardianModel, GuardianConfig
from .dual_domain import DualDomainNet, KSpaceNet, ImageDomainNet
from .diffusion import ScoreNetwork, DiffusionSampler, GaussianDiffusion
from .blackbox import BlackBoxModel, HallucinatingModel
from .biological_priors import (
    BiologicalPriorLoss,
    BiologicalPriorConfig,
    BiologicalPlausibilityScore,
    DiseaseAwarePrior,
    LesionPersistencePrior,
    TissueContinuityPrior,
    AnatomicalBoundaryPrior,
    integrate_biological_prior_into_guardian
)
from .calibrated_uncertainty import (
    CalibratedUncertaintyEstimator,
    CalibratedUncertaintyResult,
    MultiSampleReconstructor,
    UncertaintyCalibrator,
    create_reliability_diagram,
    generate_uncertainty_report,
)

__all__ = [
    # UNet
    "UNet",
    "UNetEncoder",
    "UNetDecoder",
    # Guardian
    "GuardianModel",
    "GuardianConfig",
    # Dual Domain
    "DualDomainNet",
    "KSpaceNet",
    "ImageDomainNet",
    # Diffusion
    "ScoreNetwork",
    "DiffusionSampler",
    "GaussianDiffusion",
    # BlackBox
    "BlackBoxModel",
    "HallucinatingModel",
    # Biological Priors (Novel Bioengineering Contribution)
    "BiologicalPriorLoss",
    "BiologicalPriorConfig",
    "BiologicalPlausibilityScore",
    "DiseaseAwarePrior",
    "LesionPersistencePrior",
    "TissueContinuityPrior",
    "AnatomicalBoundaryPrior",
    "integrate_biological_prior_into_guardian",
    # Calibrated Uncertainty (addresses overconfidence)
    "CalibratedUncertaintyEstimator",
    "CalibratedUncertaintyResult",
    "MultiSampleReconstructor",
    "UncertaintyCalibrator",
    "create_reliability_diagram",
    "generate_uncertainty_report",
]
