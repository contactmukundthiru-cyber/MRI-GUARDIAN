"""
Physics Module for MRI-GUARDIAN

Contains MRI physics operations and data consistency layers.
These enforce physical constraints during reconstruction.

PHD-LEVEL FEATURES:
- Universal MRI Physics: Supports ALL contrasts (T1/T2/FLAIR/DWI/SWI),
  field strengths (1.5T/3T/7T), and vendors (Siemens/GE/Philips)
- Multi-Sequence Causal Model: Enforces clinical logic across sequences
  to detect biologically impossible hallucinations
"""

from .mri_physics import (
    MRIPhysics,
    forward_model,
    adjoint_model,
    compute_sense_model,
)
from .data_consistency import (
    DataConsistencyLayer,
    SoftDataConsistency,
    HardDataConsistency,
    GradientDataConsistency,
)
from .sampling import (
    SamplingMask,
    CartesianMask,
    RadialMask,
    SpiralMask,
    create_mask_from_acceleration,
)
from .universal_mri_physics import (
    MRIContrast,
    FieldStrength,
    Vendor,
    Anatomy,
    MRIScanConfig,
    TissueProperties,
    TISSUE_DB,
    ContrastModel,
    UniversalMRIPhysicsModel,
    ModalitySpecificPhysics,
    DomainAdaptation,
    UniversalAuditResult,
    UniversalMRIAuditor,
)
from .multi_sequence_causal import (
    PathologyType,
    MRISequence,
    ContrastAppearance,
    ClinicalRule,
    CLINICAL_RULES,
    CrossContrastViolation,
    CausalAuditResult,
    ClinicalLogicEngine,
    MultiSequenceAnalyzer,
    MultiSequenceCausalAuditor,
    check_impossible_patterns,
    generate_causal_audit_report,
)

__all__ = [
    # MRI Physics
    "MRIPhysics",
    "forward_model",
    "adjoint_model",
    "compute_sense_model",
    # Data Consistency
    "DataConsistencyLayer",
    "SoftDataConsistency",
    "HardDataConsistency",
    "GradientDataConsistency",
    # Sampling
    "SamplingMask",
    "CartesianMask",
    "RadialMask",
    "SpiralMask",
    "create_mask_from_acceleration",
    # Universal MRI Physics (PHD-LEVEL: All Modalities)
    "MRIContrast",
    "FieldStrength",
    "Vendor",
    "Anatomy",
    "MRIScanConfig",
    "TissueProperties",
    "TISSUE_DB",
    "ContrastModel",
    "UniversalMRIPhysicsModel",
    "ModalitySpecificPhysics",
    "DomainAdaptation",
    "UniversalAuditResult",
    "UniversalMRIAuditor",
    # Multi-Sequence Causal Model (PHD-LEVEL: Clinical Logic)
    "PathologyType",
    "MRISequence",
    "ContrastAppearance",
    "ClinicalRule",
    "CLINICAL_RULES",
    "CrossContrastViolation",
    "CausalAuditResult",
    "ClinicalLogicEngine",
    "MultiSequenceAnalyzer",
    "MultiSequenceCausalAuditor",
    "check_impossible_patterns",
    "generate_causal_audit_report",
]
