"""
Auditor Module for MRI-GUARDIAN

The auditor is responsible for detecting hallucinations in black-box
MRI reconstructions by comparing them against physics-guided references.

Includes:
- Lesion Integrity Marker (LIM) - quantifies lesion preservation
- K-Space Analysis - separates physics violations from hallucinations
- Dirty Test Generator - stress tests with realistic corruptions
- Failure Analysis - documents limitations honestly
- Adversarial Generator - creates realistic AI failures
"""

from .detector import HallucinationDetector, DetectionResult
from .discrepancy import (
    compute_intensity_discrepancy,
    compute_structural_discrepancy,
    compute_frequency_discrepancy,
    combine_discrepancy_maps,
)
from .uncertainty import (
    compute_reconstruction_uncertainty,
    monte_carlo_uncertainty,
    ensemble_uncertainty,
)
from .hallucination import (
    HallucinationInjector,
    HallucinationConfig,
    generate_synthetic_lesions,
    generate_texture_artifacts,
)
from .lesion_integrity_marker import (
    LesionIntegrityMarker,
    LesionFingerprint,
    LesionFingerprintExtractor,
    LIMResult,
    LIMAggregator,
    AuditorLIMCorrelator,
    create_lim_visualization,
)
from .kspace_analysis import (
    analyze_kspace_residuals,
    KSpaceAnalysisResult,
    create_violation_overlay,
    compute_kspace_consistency_detailed,
)
from .dirty_test_generator import (
    DirtyTestGenerator,
    CorruptionDetector,
    CorruptionType,
    CorruptionResult,
)
from .failure_analysis import (
    FailureModeAnalyzer,
    FailureCase,
    FailureType,
    create_failure_examples,
)
from .adversarial_generator import (
    AdversarialHallucinationGenerator,
    RealFailureCollector,
    create_hallucination_test_suite,
)
from .z_consistency import (
    ZConsistencyChecker,
    ZConsistencyResult,
    check_z_consistency_batch,
    generate_z_consistency_report,
)
from .counterfactual import (
    HypothesisTester,
    HypothesisTestResult,
    AutomaticHypothesisTester,
    generate_hypothesis_report,
)
from .spectral_fingerprint import (
    SpectralFingerprintDetector,
    SpectralFingerprintResult,
    RadialPowerSpectrum,
    generate_spectral_report,
)
from .clinical_resampling import (
    ClinicalResamplingAdvisor,
    ResamplingRecommendation,
    simulate_resampling_benefit,
    generate_resampling_report,
)
from .longitudinal_audit import (
    LongitudinalAuditor,
    LongitudinalAuditResult,
    LesionProgression,
    LesionState,
    ProgressionType,
    LesionTracker,
    ProgressionPhysics,
    generate_longitudinal_report,
)

__all__ = [
    # Main detector
    "HallucinationDetector",
    "DetectionResult",
    # Discrepancy computation
    "compute_intensity_discrepancy",
    "compute_structural_discrepancy",
    "compute_frequency_discrepancy",
    "combine_discrepancy_maps",
    # Uncertainty estimation
    "compute_reconstruction_uncertainty",
    "monte_carlo_uncertainty",
    "ensemble_uncertainty",
    # Hallucination generation
    "HallucinationInjector",
    "HallucinationConfig",
    "generate_synthetic_lesions",
    "generate_texture_artifacts",
    # Lesion Integrity Marker (LIM)
    "LesionIntegrityMarker",
    "LesionFingerprint",
    "LesionFingerprintExtractor",
    "LIMResult",
    "LIMAggregator",
    "AuditorLIMCorrelator",
    "create_lim_visualization",
    # K-Space Analysis (addresses Black Box Hypocrisy)
    "analyze_kspace_residuals",
    "KSpaceAnalysisResult",
    "create_violation_overlay",
    "compute_kspace_consistency_detailed",
    # Dirty Test Generator (stress testing)
    "DirtyTestGenerator",
    "CorruptionDetector",
    "CorruptionType",
    "CorruptionResult",
    # Failure Analysis (scientific honesty)
    "FailureModeAnalyzer",
    "FailureCase",
    "FailureType",
    "create_failure_examples",
    # Adversarial Testing (real AI failures)
    "AdversarialHallucinationGenerator",
    "RealFailureCollector",
    "create_hallucination_test_suite",
    # Z-Consistency (3D hallucination detection)
    "ZConsistencyChecker",
    "ZConsistencyResult",
    "check_z_consistency_batch",
    "generate_z_consistency_report",
    # Counterfactual Hypothesis Testing (THE SCIENCE-HEAVY FEATURE)
    "HypothesisTester",
    "HypothesisTestResult",
    "AutomaticHypothesisTester",
    "generate_hypothesis_report",
    # Spectral Fingerprint Forensics (THE FORENSICS FEATURE)
    "SpectralFingerprintDetector",
    "SpectralFingerprintResult",
    "RadialPowerSpectrum",
    "generate_spectral_report",
    # Clinical Re-Sampling (THE WORKFLOW FEATURE)
    "ClinicalResamplingAdvisor",
    "ResamplingRecommendation",
    "simulate_resampling_benefit",
    "generate_resampling_report",
    # Longitudinal Audit (PHD-LEVEL: Disease Progression Tracking)
    "LongitudinalAuditor",
    "LongitudinalAuditResult",
    "LesionProgression",
    "LesionState",
    "ProgressionType",
    "LesionTracker",
    "ProgressionPhysics",
    "generate_longitudinal_report",
]
