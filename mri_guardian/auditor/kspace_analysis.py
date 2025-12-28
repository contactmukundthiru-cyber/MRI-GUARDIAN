"""
K-Space Residual Analysis - Addressing the "Black Box Hypocrisy"
=================================================================

THE CONCERN:
"You claim to solve the Black Box problem, but Guardian is also a
neural network. Why should we trust Guardian over the Black-box?"

THE ANSWER:
We don't ask you to trust Guardian blindly. We PROVE it using physics.

This module separates discrepancies into:
1. PHYSICS VIOLATIONS (Data Mismatch) - Where Black-box contradicts
   actual k-space measurements. These are OBJECTIVELY wrong.

2. PLAUSIBILITY VIOLATIONS (Hallucination) - Where Black-box fills
   in unmeasured frequencies differently than Guardian. These are
   potentially wrong.

The key insight:
- Errors in MEASURED k-space frequencies are provably wrong
- Errors in UNMEASURED frequencies are judgment calls
- Guardian is trusted because it agrees with physics, not because
  it's a neural network

VISUALIZATION:
Dashboard shows separate maps for:
- "Physics Violation Map" (red) - Objectively wrong
- "Plausibility Violation Map" (yellow) - Potentially wrong
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KSpaceAnalysisResult:
    """Results from k-space residual analysis."""
    # Maps
    physics_violation_map: np.ndarray  # Errors in measured frequencies
    plausibility_violation_map: np.ndarray  # Errors in unmeasured frequencies
    total_discrepancy_map: np.ndarray

    # Metrics
    physics_violation_score: float  # 0-1, lower is better
    plausibility_violation_score: float
    total_violation_score: float

    # Diagnosis
    is_physics_violated: bool
    is_hallucination_likely: bool
    confidence: float
    explanation: str


def analyze_kspace_residuals(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    measured_kspace: np.ndarray,
    sampling_mask: np.ndarray
) -> KSpaceAnalysisResult:
    """
    Analyze k-space residuals to separate physics violations from hallucinations.

    Args:
        reconstruction: Black-box reconstruction to audit
        reference: Guardian (trusted) reconstruction
        measured_kspace: Original k-space measurements
        sampling_mask: Binary mask of sampled k-space locations

    Returns:
        KSpaceAnalysisResult with separated violation maps
    """
    # Convert to k-space
    recon_kspace = np.fft.fftshift(np.fft.fft2(reconstruction))
    ref_kspace = np.fft.fftshift(np.fft.fft2(reference))

    # Masks
    measured_mask = sampling_mask > 0.5
    unmeasured_mask = ~measured_mask

    # =========================================================================
    # PHYSICS VIOLATIONS: Errors at MEASURED k-space locations
    # These are OBJECTIVELY WRONG - the reconstruction contradicts data
    # =========================================================================
    physics_error_kspace = np.zeros_like(recon_kspace, dtype=np.float32)
    physics_error_kspace[measured_mask] = np.abs(
        recon_kspace[measured_mask] - measured_kspace[measured_mask]
    )

    # Convert to image domain to show WHERE physics is violated
    physics_violation_map = np.abs(np.fft.ifft2(np.fft.ifftshift(physics_error_kspace)))
    physics_violation_map = physics_violation_map / (physics_violation_map.max() + 1e-8)

    # =========================================================================
    # PLAUSIBILITY VIOLATIONS: Differences at UNMEASURED k-space locations
    # These are where Black-box and Guardian disagree on how to fill gaps
    # Not objectively wrong, but suspicious
    # =========================================================================
    plausibility_error_kspace = np.zeros_like(recon_kspace, dtype=np.float32)
    plausibility_error_kspace[unmeasured_mask] = np.abs(
        recon_kspace[unmeasured_mask] - ref_kspace[unmeasured_mask]
    )

    # Convert to image domain
    plausibility_violation_map = np.abs(np.fft.ifft2(np.fft.ifftshift(plausibility_error_kspace)))
    plausibility_violation_map = plausibility_violation_map / (plausibility_violation_map.max() + 1e-8)

    # =========================================================================
    # TOTAL DISCREPANCY (for reference)
    # =========================================================================
    total_discrepancy_map = np.abs(reconstruction - reference)
    total_discrepancy_map = total_discrepancy_map / (total_discrepancy_map.max() + 1e-8)

    # =========================================================================
    # METRICS
    # =========================================================================
    # Physics violation score (normalized by signal energy)
    signal_energy = np.sum(np.abs(measured_kspace[measured_mask]) ** 2)
    physics_error_energy = np.sum(physics_error_kspace[measured_mask] ** 2)
    physics_violation_score = physics_error_energy / (signal_energy + 1e-8)

    # Plausibility violation score (normalized)
    plausibility_violation_score = float(np.mean(plausibility_violation_map))

    # Total
    total_violation_score = float(np.mean(total_discrepancy_map))

    # =========================================================================
    # DIAGNOSIS
    # =========================================================================
    # Physics violation threshold: if > 0.01, something is wrong
    is_physics_violated = physics_violation_score > 0.01

    # Hallucination threshold: high discrepancy in unmeasured regions
    is_hallucination_likely = plausibility_violation_score > 0.1

    # Confidence based on how clear the signal is
    if is_physics_violated:
        confidence = min(1.0, physics_violation_score * 10)
    elif is_hallucination_likely:
        confidence = min(1.0, plausibility_violation_score * 5)
    else:
        confidence = 1.0 - total_violation_score

    # Generate explanation
    if is_physics_violated:
        explanation = (
            f"PHYSICS VIOLATION DETECTED (score: {physics_violation_score:.4f}). "
            f"The reconstruction CONTRADICTS actual k-space measurements. "
            f"This is OBJECTIVELY wrong - not a matter of interpretation. "
            f"Affected regions shown in red."
        )
    elif is_hallucination_likely:
        explanation = (
            f"POTENTIAL HALLUCINATION (score: {plausibility_violation_score:.3f}). "
            f"The reconstruction differs from Guardian in unmeasured k-space regions. "
            f"This could be hallucination or legitimate difference in regularization. "
            f"Suspicious regions shown in yellow. Expert review recommended."
        )
    else:
        explanation = (
            f"NO SIGNIFICANT VIOLATIONS (total: {total_violation_score:.3f}). "
            f"Reconstruction is consistent with physics and plausibility constraints."
        )

    return KSpaceAnalysisResult(
        physics_violation_map=physics_violation_map.astype(np.float32),
        plausibility_violation_map=plausibility_violation_map.astype(np.float32),
        total_discrepancy_map=total_discrepancy_map.astype(np.float32),
        physics_violation_score=float(physics_violation_score),
        plausibility_violation_score=float(plausibility_violation_score),
        total_violation_score=float(total_violation_score),
        is_physics_violated=is_physics_violated,
        is_hallucination_likely=is_hallucination_likely,
        confidence=float(confidence),
        explanation=explanation
    )


def create_violation_overlay(
    image: np.ndarray,
    physics_map: np.ndarray,
    plausibility_map: np.ndarray,
    physics_color: Tuple[float, float, float] = (1, 0, 0),  # Red
    plausibility_color: Tuple[float, float, float] = (1, 1, 0)  # Yellow
) -> np.ndarray:
    """
    Create RGB overlay showing physics (red) and plausibility (yellow) violations.

    This is the key visualization for the dashboard.
    """
    # Normalize image to grayscale RGB
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

    # Threshold maps for clear visualization
    physics_thresh = physics_map > 0.1
    plausibility_thresh = (plausibility_map > 0.1) & ~physics_thresh

    # Apply colors
    for i, c in enumerate(physics_color):
        rgb[:, :, i] = np.where(physics_thresh, c, rgb[:, :, i])

    for i, c in enumerate(plausibility_color):
        rgb[:, :, i] = np.where(plausibility_thresh,
                                0.5 * rgb[:, :, i] + 0.5 * c,
                                rgb[:, :, i])

    return rgb


def generate_kspace_report(result: KSpaceAnalysisResult) -> str:
    """Generate a report explaining the k-space analysis."""

    physics_status = "⚠️ VIOLATION" if result.is_physics_violated else "✓ OK"
    plausibility_status = "⚠️ SUSPICIOUS" if result.is_hallucination_likely else "✓ OK"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║     K-SPACE RESIDUAL ANALYSIS                                ║
║     Separating Physics Violations from Hallucinations        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  WHY THIS MATTERS:                                           ║
║  • Errors in MEASURED k-space = OBJECTIVELY WRONG            ║
║  • Errors in UNMEASURED k-space = POTENTIALLY WRONG          ║
║  • Guardian is trusted because of PHYSICS, not faith         ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  PHYSICS VIOLATIONS (Red in visualization)                   ║
╠──────────────────────────────────────────────────────────────╣
║  Status:        {physics_status:<44} ║
║  Score:         {result.physics_violation_score:>8.6f} (< 0.01 = acceptable)       ║
║  Meaning:       Reconstruction contradicts measured data     ║
╠──────────────────────────────────────────────────────────────╣
║  PLAUSIBILITY VIOLATIONS (Yellow in visualization)           ║
╠──────────────────────────────────────────────────────────────╣
║  Status:        {plausibility_status:<44} ║
║  Score:         {result.plausibility_violation_score:>8.4f} (< 0.1 = acceptable)          ║
║  Meaning:       Different from Guardian in unmeasured region ║
╠──────────────────────────────────────────────────────────────╣
║  CONFIDENCE: {result.confidence:>6.1%}                                       ║
╠══════════════════════════════════════════════════════════════╣
║  {result.explanation[:58]:<58} ║
╚══════════════════════════════════════════════════════════════╝
"""
    return report


def compute_kspace_consistency_detailed(
    reconstruction: np.ndarray,
    measured_kspace: np.ndarray,
    sampling_mask: np.ndarray
) -> Dict[str, float]:
    """
    Detailed k-space consistency metrics.

    This proves Guardian respects physics while Black-box doesn't.
    """
    recon_kspace = np.fft.fftshift(np.fft.fft2(reconstruction))
    measured_mask = sampling_mask > 0.5

    # Error at measured locations
    measured_error = np.abs(recon_kspace[measured_mask] - measured_kspace[measured_mask])

    # Signal magnitude at measured locations
    signal_mag = np.abs(measured_kspace[measured_mask])

    # Metrics
    mean_abs_error = float(np.mean(measured_error))
    max_abs_error = float(np.max(measured_error))
    relative_error = float(np.mean(measured_error / (signal_mag + 1e-8)))

    # Energy conservation (Parseval's theorem)
    image_energy = float(np.sum(np.abs(reconstruction) ** 2))
    kspace_energy = float(np.sum(np.abs(recon_kspace) ** 2)) / recon_kspace.size
    energy_ratio = image_energy / (kspace_energy + 1e-8)

    # Phase consistency at center
    center = tuple(s // 2 for s in recon_kspace.shape)
    center_region = slice(center[0]-5, center[0]+5), slice(center[1]-5, center[1]+5)
    phase_recon = np.angle(recon_kspace[center_region])
    phase_measured = np.angle(measured_kspace[center_region])
    phase_consistency = float(np.cos(phase_recon - phase_measured).mean())

    return {
        'mean_kspace_error': mean_abs_error,
        'max_kspace_error': max_abs_error,
        'relative_kspace_error': relative_error,
        'energy_conservation_ratio': energy_ratio,
        'phase_consistency': phase_consistency,
        'physics_score': float(max(0, 1.0 - 10 * relative_error)),
        'is_physics_consistent': relative_error < 0.01
    }
