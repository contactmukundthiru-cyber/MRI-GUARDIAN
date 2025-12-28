"""
MRI-GUARDIAN Safety Framework

A comprehensive AI safety pipeline for MRI reconstruction that evaluates:
- Hallucinations
- Uncertainty
- Distribution shift
- Physics violations
- Lesion integrity
- Patient-specific bias
- Clinical risk scoring

This framework goes beyond hallucination detection to provide
a complete safety assessment for AI-reconstructed MRI images.
"""

from .uncertainty import (
    UncertaintyEstimator,
    MCDropoutEstimator,
    EnsembleEstimator,
    EvidentialEstimator,
    TTAUncertaintyEstimator,
)
from .distribution_shift import (
    DistributionShiftDetector,
    ScannerFingerprintDetector,
    AnatomyOutlierDetector,
)
from .physics_violation import (
    PhysicsViolationDetector,
    KSpaceConsistencyChecker,
    PhaseCoherenceChecker,
    GradientPhysicsChecker,
)
from .lesion_integrity import (
    LesionIntegrityVerifier,
    SubtleLesionGenerator,
    LesionPreservationMetrics,
)
from .bias_detection import (
    BiasDetector,
    SubgroupAnalyzer,
    FairnessMetrics,
)
from .clinical_risk import (
    ClinicalRiskScorer,
    RiskLevel,
    ClinicalConfidenceMap,
)
from .clinical_validation import (
    ClinicalValidationCalculator,
    ClinicalValidationBenchmark,
    ValidationReport,
    ValidationResult,
    ValidationMetrics,
    ResidualErrorAnalyzer,
    ArtifactSeverityScorer,
    UncertaintyCalibrationAnalyzer,
    LesionDetectabilityAnalyzer,
)
from .multi_signal import (
    MultiSignalConsistencyChecker,
    AnomalyFusion,
)
from .virtual_clinical_trial import (
    VirtualClinicalTrial,
    VCTBattery,
    VCTBatteryResult,
    VCTTestResult,
    TestStatus,
    RegulatoryStandard,
    LesionSafetyBattery,
    AcquisitionStressTest,
    BiasGeneralizationPanel,
    AuditorPerformanceEvaluation,
    run_virtual_clinical_trial,
)

__all__ = [
    # Uncertainty
    "UncertaintyEstimator",
    "MCDropoutEstimator",
    "EnsembleEstimator",
    "EvidentialEstimator",
    "TTAUncertaintyEstimator",
    # Distribution shift
    "DistributionShiftDetector",
    "ScannerFingerprintDetector",
    "AnatomyOutlierDetector",
    # Physics violation
    "PhysicsViolationDetector",
    "KSpaceConsistencyChecker",
    "PhaseCoherenceChecker",
    "GradientPhysicsChecker",
    # Lesion integrity
    "LesionIntegrityVerifier",
    "SubtleLesionGenerator",
    "LesionPreservationMetrics",
    # Bias detection
    "BiasDetector",
    "SubgroupAnalyzer",
    "FairnessMetrics",
    # Clinical risk
    "ClinicalRiskScorer",
    "RiskLevel",
    "ClinicalConfidenceMap",
    # Clinical Validation (Research, NOT Regulatory)
    "ClinicalValidationCalculator",
    "ClinicalValidationBenchmark",
    "ValidationReport",
    "ValidationResult",
    "ValidationMetrics",
    "ResidualErrorAnalyzer",
    "ArtifactSeverityScorer",
    "UncertaintyCalibrationAnalyzer",
    "LesionDetectabilityAnalyzer",
    # Multi-signal
    "MultiSignalConsistencyChecker",
    "AnomalyFusion",
    # Virtual Clinical Trial (Novel Regulatory Framework)
    "VirtualClinicalTrial",
    "VCTBattery",
    "VCTBatteryResult",
    "VCTTestResult",
    "TestStatus",
    "RegulatoryStandard",
    "LesionSafetyBattery",
    "AcquisitionStressTest",
    "BiasGeneralizationPanel",
    "AuditorPerformanceEvaluation",
    "run_virtual_clinical_trial",
]
