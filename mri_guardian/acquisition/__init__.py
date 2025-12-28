"""
Acquisition Control Module for MRI-GUARDIAN

This module provides adaptive acquisition control - using the
auditor to guide what k-space lines should be acquired.

The ultimate goal: Real-time scanner control for optimal sampling.

Current implementation: Simulation-based proof of concept showing
that adaptive acquisition achieves same quality with less data.
"""

from .adaptive_controller import (
    AdaptiveAcquisitionController,
    AdaptiveAcquisitionResult,
    AcquisitionGuidance,
    AcquisitionDecision,
    UncertaintyEstimator,
    LesionSafetyMonitor,
    PhysicsViolationDetector,
    generate_adaptive_acquisition_report,
)

__all__ = [
    "AdaptiveAcquisitionController",
    "AdaptiveAcquisitionResult",
    "AcquisitionGuidance",
    "AcquisitionDecision",
    "UncertaintyEstimator",
    "LesionSafetyMonitor",
    "PhysicsViolationDetector",
    "generate_adaptive_acquisition_report",
]
