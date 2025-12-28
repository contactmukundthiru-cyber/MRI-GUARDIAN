"""
Longitudinal MRI Safety Audit - Disease Progression Tracking
=============================================================

PHD-LEVEL CONTRIBUTION (Achievable via Simulation):
Track lesion geometry evolution across time points.
Detect unnatural progression patterns that indicate AI hallucination.

CLINICAL IMPORTANCE:
====================
Doctors care deeply about:
- MS lesion progression
- Tumor growth/shrinkage
- Stroke evolution
- Cartilage thinning
- New microbleeds appearing

AI reconstructions can:
- Hide real progression
- Exaggerate growth
- Fake shrinkage
- Distort volumes

THIS MODULE:
============
✓ Compare MRI scans over time
✓ Track lesion geometry evolution
✓ Check consistency across longitudinal scans
✓ Automatically detect unnatural progression patterns
✓ Flag hallucinations that mimic real progression

NO TRAINING REQUIRED - Uses physics-based consistency checks.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto


class ProgressionType(Enum):
    """Types of lesion progression."""
    STABLE = auto()
    GROWING = auto()
    SHRINKING = auto()
    NEW_LESION = auto()
    RESOLVED = auto()
    SUSPICIOUS = auto()  # Unnatural pattern


@dataclass
class LesionState:
    """State of a lesion at one time point."""
    time_point: int
    centroid: Tuple[float, float]
    area: float
    diameter: float
    mean_intensity: float
    max_intensity: float
    contrast: float
    boundary_sharpness: float
    mask: Optional[np.ndarray] = None


@dataclass
class LesionProgression:
    """Progression of a single lesion across time."""
    lesion_id: int
    states: List[LesionState]
    progression_type: ProgressionType

    # Derived metrics
    area_change_rate: float  # % per time point
    volume_doubling_time: Optional[float]  # Time points
    centroid_drift: float  # pixels moved
    contrast_change: float  # Intensity change

    # Flags
    is_natural: bool
    suspicion_score: float
    explanation: str


@dataclass
class LongitudinalAuditResult:
    """Result from longitudinal audit."""
    num_time_points: int
    lesion_progressions: List[LesionProgression]

    # Summary metrics
    num_stable: int
    num_growing: int
    num_shrinking: int
    num_new: int
    num_resolved: int
    num_suspicious: int

    # Overall assessment
    progression_consistent: bool
    hallucination_risk: float
    clinical_alert: bool

    explanation: str


class ProgressionPhysics:
    """
    Physics-based constraints on natural disease progression.

    KEY INSIGHT:
    Real disease progression has physical limits.
    A tumor can't double overnight. Lesions don't teleport.
    These constraints help detect AI hallucinations.
    """

    # Biological constraints (from literature)
    MAX_TUMOR_DOUBLING_RATE = 0.1  # 10% volume increase per month (fast tumor)
    MAX_MS_LESION_GROWTH = 0.2    # 20% area increase per scan interval
    MAX_STROKE_EVOLUTION = 0.5   # 50% change in acute phase
    MAX_CENTROID_DRIFT = 3.0     # pixels between scans
    MIN_CONTRAST_STABILITY = 0.8  # Contrast shouldn't drop more than 20%

    @staticmethod
    def get_progression_limits(lesion_type: str) -> Dict[str, float]:
        """Get physics-based limits for progression."""
        limits = {
            'tumor': {
                'max_growth_rate': 0.15,  # 15% per scan
                'max_shrink_rate': 0.30,  # 30% (with treatment)
                'max_drift': 2.0,
                'min_contrast_ratio': 0.7,
            },
            'ms_lesion': {
                'max_growth_rate': 0.25,
                'max_shrink_rate': 0.20,
                'max_drift': 1.5,
                'min_contrast_ratio': 0.8,
            },
            'stroke': {
                'max_growth_rate': 0.50,  # Acute phase
                'max_shrink_rate': 0.40,
                'max_drift': 3.0,
                'min_contrast_ratio': 0.5,
            },
            'default': {
                'max_growth_rate': 0.20,
                'max_shrink_rate': 0.25,
                'max_drift': 2.5,
                'min_contrast_ratio': 0.7,
            }
        }
        return limits.get(lesion_type, limits['default'])


class LesionTracker:
    """
    Track lesions across time points using spatial correspondence.
    """

    def __init__(self, max_distance: float = 20.0):
        self.max_distance = max_distance

    def track_lesions(
        self,
        time_series_masks: List[np.ndarray],  # List of binary lesion masks
        time_series_images: List[np.ndarray]  # Corresponding images
    ) -> List[List[LesionState]]:
        """
        Track lesions across time series.

        Returns:
            List of lesion tracks, each containing states at each time point
        """
        all_tracks = []

        # Extract lesions at each time point
        time_point_lesions = []
        for t, (mask, image) in enumerate(zip(time_series_masks, time_series_images)):
            lesions = self._extract_lesion_states(mask, image, t)
            time_point_lesions.append(lesions)

        if not time_point_lesions or not time_point_lesions[0]:
            return []

        # Initialize tracks from first time point
        for lesion in time_point_lesions[0]:
            all_tracks.append([lesion])

        # Match lesions across subsequent time points
        for t in range(1, len(time_point_lesions)):
            current_lesions = time_point_lesions[t]
            matched = set()

            # Match each track to closest lesion
            for track in all_tracks:
                last_state = track[-1]
                best_match = None
                best_distance = self.max_distance

                for i, lesion in enumerate(current_lesions):
                    if i in matched:
                        continue
                    dist = np.sqrt(
                        (last_state.centroid[0] - lesion.centroid[0])**2 +
                        (last_state.centroid[1] - lesion.centroid[1])**2
                    )
                    if dist < best_distance:
                        best_distance = dist
                        best_match = i

                if best_match is not None:
                    track.append(current_lesions[best_match])
                    matched.add(best_match)
                else:
                    # Lesion disappeared - create placeholder
                    track.append(None)

            # New lesions (unmatched)
            for i, lesion in enumerate(current_lesions):
                if i not in matched:
                    # Start new track with None for previous time points
                    new_track = [None] * t + [lesion]
                    all_tracks.append(new_track)

        return all_tracks

    def _extract_lesion_states(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        time_point: int
    ) -> List[LesionState]:
        """Extract lesion states from a single time point."""
        labeled, n_lesions = ndimage.label(mask > 0.5)
        lesions = []

        for i in range(1, n_lesions + 1):
            lesion_mask = labeled == i

            if lesion_mask.sum() < 5:  # Skip tiny regions
                continue

            # Compute properties
            coords = np.where(lesion_mask)
            centroid = (coords[0].mean(), coords[1].mean())
            area = float(lesion_mask.sum())
            diameter = 2 * np.sqrt(area / np.pi)

            # Intensity properties
            lesion_intensity = image[lesion_mask]
            background_intensity = image[~mask.astype(bool)].mean() if (~mask.astype(bool)).sum() > 0 else 0

            mean_intensity = float(lesion_intensity.mean())
            max_intensity = float(lesion_intensity.max())
            contrast = (mean_intensity - background_intensity) / (background_intensity + 1e-8)

            # Boundary sharpness (gradient at edge)
            dilated = ndimage.binary_dilation(lesion_mask)
            boundary = dilated & ~lesion_mask
            if boundary.sum() > 0:
                boundary_sharpness = float(np.abs(image[boundary] - mean_intensity).mean())
            else:
                boundary_sharpness = 0.0

            lesions.append(LesionState(
                time_point=time_point,
                centroid=centroid,
                area=area,
                diameter=diameter,
                mean_intensity=mean_intensity,
                max_intensity=max_intensity,
                contrast=contrast,
                boundary_sharpness=boundary_sharpness,
                mask=lesion_mask
            ))

        return lesions


class LongitudinalAuditor:
    """
    Audit lesion progression across longitudinal MRI scans.

    Detects:
    - Unnatural growth/shrinkage rates
    - Lesion "teleportation" (centroid drift)
    - Contrast inconsistencies
    - Sudden appearance/disappearance

    These patterns suggest AI hallucination rather than real disease.
    """

    def __init__(self, lesion_type: str = 'default'):
        self.lesion_type = lesion_type
        self.tracker = LesionTracker()
        self.limits = ProgressionPhysics.get_progression_limits(lesion_type)

    def audit(
        self,
        time_series_masks: List[np.ndarray],
        time_series_images: List[np.ndarray],
        scan_interval_days: float = 90.0  # Days between scans
    ) -> LongitudinalAuditResult:
        """
        Audit longitudinal progression for hallucination detection.

        Args:
            time_series_masks: Lesion masks at each time point
            time_series_images: Images at each time point
            scan_interval_days: Time between scans

        Returns:
            LongitudinalAuditResult with comprehensive analysis
        """
        num_time_points = len(time_series_masks)

        # Track lesions across time
        tracks = self.tracker.track_lesions(time_series_masks, time_series_images)

        # Analyze each track
        progressions = []
        for lesion_id, track in enumerate(tracks):
            progression = self._analyze_track(lesion_id, track, scan_interval_days)
            progressions.append(progression)

        # Count progression types
        type_counts = {t: 0 for t in ProgressionType}
        for p in progressions:
            type_counts[p.progression_type] += 1

        # Overall assessment
        num_suspicious = type_counts[ProgressionType.SUSPICIOUS]
        total_lesions = len(progressions)
        hallucination_risk = num_suspicious / max(1, total_lesions)

        progression_consistent = hallucination_risk < 0.2
        clinical_alert = num_suspicious > 0 or hallucination_risk > 0.1

        # Generate explanation
        explanation = self._generate_explanation(
            progressions, type_counts, hallucination_risk
        )

        return LongitudinalAuditResult(
            num_time_points=num_time_points,
            lesion_progressions=progressions,
            num_stable=type_counts[ProgressionType.STABLE],
            num_growing=type_counts[ProgressionType.GROWING],
            num_shrinking=type_counts[ProgressionType.SHRINKING],
            num_new=type_counts[ProgressionType.NEW_LESION],
            num_resolved=type_counts[ProgressionType.RESOLVED],
            num_suspicious=num_suspicious,
            progression_consistent=progression_consistent,
            hallucination_risk=hallucination_risk,
            clinical_alert=clinical_alert,
            explanation=explanation
        )

    def _analyze_track(
        self,
        lesion_id: int,
        track: List[Optional[LesionState]],
        scan_interval_days: float
    ) -> LesionProgression:
        """Analyze a single lesion track for natural progression."""
        # Filter out None (missing time points)
        valid_states = [s for s in track if s is not None]

        if len(valid_states) == 0:
            return self._create_empty_progression(lesion_id)

        if len(valid_states) == 1:
            # Single observation - check if it's a new lesion
            first_idx = next(i for i, s in enumerate(track) if s is not None)
            if first_idx > 0:
                return LesionProgression(
                    lesion_id=lesion_id,
                    states=valid_states,
                    progression_type=ProgressionType.NEW_LESION,
                    area_change_rate=0.0,
                    volume_doubling_time=None,
                    centroid_drift=0.0,
                    contrast_change=0.0,
                    is_natural=True,  # Can't assess from single point
                    suspicion_score=0.2,
                    explanation="New lesion appeared"
                )
            else:
                return LesionProgression(
                    lesion_id=lesion_id,
                    states=valid_states,
                    progression_type=ProgressionType.RESOLVED,
                    area_change_rate=0.0,
                    volume_doubling_time=None,
                    centroid_drift=0.0,
                    contrast_change=0.0,
                    is_natural=True,
                    suspicion_score=0.2,
                    explanation="Lesion resolved"
                )

        # Compute progression metrics
        first_state = valid_states[0]
        last_state = valid_states[-1]

        # Area change
        area_change = (last_state.area - first_state.area) / (first_state.area + 1e-8)
        num_intervals = len(valid_states) - 1
        area_change_rate = area_change / max(1, num_intervals)

        # Volume doubling time (if growing)
        if area_change > 0:
            # Assuming spherical, volume ~ area^1.5
            volume_ratio = (last_state.area / first_state.area) ** 1.5
            if volume_ratio > 1:
                doubling_time = np.log(2) / np.log(volume_ratio) * num_intervals
            else:
                doubling_time = None
        else:
            doubling_time = None

        # Centroid drift
        total_drift = 0
        for i in range(1, len(valid_states)):
            drift = np.sqrt(
                (valid_states[i].centroid[0] - valid_states[i-1].centroid[0])**2 +
                (valid_states[i].centroid[1] - valid_states[i-1].centroid[1])**2
            )
            total_drift += drift
        centroid_drift = total_drift / num_intervals

        # Contrast change
        contrast_change = last_state.contrast / (first_state.contrast + 1e-8)

        # Check against physics limits
        violations = []
        suspicion_score = 0.0

        # Growth rate check
        if abs(area_change_rate) > self.limits['max_growth_rate']:
            violations.append(f"Abnormal growth rate: {100*area_change_rate:.0f}%")
            suspicion_score += 0.3

        # Drift check
        if centroid_drift > self.limits['max_drift']:
            violations.append(f"Abnormal drift: {centroid_drift:.1f}px")
            suspicion_score += 0.3

        # Contrast check
        if contrast_change < self.limits['min_contrast_ratio']:
            violations.append(f"Abnormal contrast drop: {100*contrast_change:.0f}%")
            suspicion_score += 0.2

        # Non-monotonic progression check
        areas = [s.area for s in valid_states]
        if len(areas) > 2:
            diffs = np.diff(areas)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes > 1:
                violations.append(f"Non-monotonic progression ({sign_changes} reversals)")
                suspicion_score += 0.2

        # Determine progression type
        if suspicion_score > 0.5:
            progression_type = ProgressionType.SUSPICIOUS
        elif abs(area_change) < 0.1:
            progression_type = ProgressionType.STABLE
        elif area_change > 0:
            progression_type = ProgressionType.GROWING
        else:
            progression_type = ProgressionType.SHRINKING

        is_natural = len(violations) == 0

        explanation = (
            f"{'Natural' if is_natural else 'SUSPICIOUS'} progression. "
            f"Area change: {100*area_change:+.0f}%, Drift: {centroid_drift:.1f}px. "
            + (f"Violations: {'; '.join(violations)}" if violations else "")
        )

        return LesionProgression(
            lesion_id=lesion_id,
            states=valid_states,
            progression_type=progression_type,
            area_change_rate=area_change_rate,
            volume_doubling_time=doubling_time,
            centroid_drift=centroid_drift,
            contrast_change=contrast_change,
            is_natural=is_natural,
            suspicion_score=min(1.0, suspicion_score),
            explanation=explanation
        )

    def _create_empty_progression(self, lesion_id: int) -> LesionProgression:
        """Create empty progression for missing lesion."""
        return LesionProgression(
            lesion_id=lesion_id,
            states=[],
            progression_type=ProgressionType.RESOLVED,
            area_change_rate=0.0,
            volume_doubling_time=None,
            centroid_drift=0.0,
            contrast_change=0.0,
            is_natural=True,
            suspicion_score=0.0,
            explanation="Lesion not found in any time point"
        )

    def _generate_explanation(
        self,
        progressions: List[LesionProgression],
        type_counts: Dict,
        risk: float
    ) -> str:
        """Generate overall explanation."""
        suspicious = [p for p in progressions if p.progression_type == ProgressionType.SUSPICIOUS]

        if not suspicious:
            return (
                f"All {len(progressions)} tracked lesions show NATURAL progression patterns. "
                f"No hallucination detected. "
                f"({type_counts[ProgressionType.STABLE]} stable, "
                f"{type_counts[ProgressionType.GROWING]} growing, "
                f"{type_counts[ProgressionType.SHRINKING]} shrinking)"
            )
        else:
            return (
                f"WARNING: {len(suspicious)} lesions show SUSPICIOUS progression. "
                f"Possible AI hallucination affecting disease tracking. "
                f"Hallucination risk: {100*risk:.0f}%. "
                f"Review recommended."
            )


def generate_longitudinal_report(result: LongitudinalAuditResult) -> str:
    """Generate longitudinal audit report."""
    status = "CONSISTENT" if result.progression_consistent else "SUSPICIOUS"
    icon = "✓" if result.progression_consistent else "⚠️"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║       LONGITUDINAL MRI SAFETY AUDIT                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CLINICAL IMPORTANCE:                                        ║
║  Real disease progression has physical limits.               ║
║  AI hallucinations can fake/hide progression.                ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  TIME SERIES SUMMARY                                          ║
╠──────────────────────────────────────────────────────────────╣
║  Time points analyzed:    {result.num_time_points:>8}                        ║
║  Lesions tracked:         {len(result.lesion_progressions):>8}                        ║
╠──────────────────────────────────────────────────────────────╣
║  PROGRESSION BREAKDOWN                                        ║
╠──────────────────────────────────────────────────────────────╣
║  Stable:                  {result.num_stable:>8}                        ║
║  Growing:                 {result.num_growing:>8}                        ║
║  Shrinking:               {result.num_shrinking:>8}                        ║
║  New lesions:             {result.num_new:>8}                        ║
║  Resolved:                {result.num_resolved:>8}                        ║
║  ⚠️  SUSPICIOUS:          {result.num_suspicious:>8}                        ║
╠══════════════════════════════════════════════════════════════╣
║  ASSESSMENT: {icon} {status:<45} ║
║  Hallucination Risk: {100*result.hallucination_risk:>5.1f}%                                ║
║  Clinical Alert: {"YES" if result.clinical_alert else "No":<10}                               ║
╠══════════════════════════════════════════════════════════════╣
║  EXPLANATION                                                  ║
╠──────────────────────────────────────────────────────────────╣
"""
    # Wrap explanation
    exp_lines = [result.explanation[i:i+58] for i in range(0, len(result.explanation), 58)]
    for line in exp_lines:
        report += f"║  {line:<58} ║\n"

    report += "╚══════════════════════════════════════════════════════════════╝"

    return report
