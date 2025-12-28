"""
Theoretical Framework Module for MRI-GUARDIAN

This module contains mathematical theory and formal definitions
that don't require training - pure physics and mathematics.

Includes:
- Lesion Integrity Theory (formal bounds on distortion)
- Frequency Band Analysis (k-space encoding of lesions)
- Auditor Response Theory (when to flag vs accept)
"""

from .lesion_integrity_theory import (
    LesionIntegrityTheory,
    LesionIntegrityBounds,
    FrequencyBandAnalysis,
    AuditorResponseTheory,
    generate_theory_report,
)

__all__ = [
    "LesionIntegrityTheory",
    "LesionIntegrityBounds",
    "FrequencyBandAnalysis",
    "AuditorResponseTheory",
    "generate_theory_report",
]
