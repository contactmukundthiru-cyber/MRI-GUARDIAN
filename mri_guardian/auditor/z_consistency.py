"""
Z-Consistency Check for 3D Hallucination Detection
===================================================

THE PROBLEM:
Current MRI reconstruction and auditing is often done slice-by-slice (2D).
This is BLIND to 3D anatomical consistency.

Example failure mode:
- Slice 45: AI "sees" a tumor (hallucination)
- Slice 46: No tumor
- Slice 47: AI "sees" a tumor again

This violates 3D anatomy: real tumors don't appear/disappear between slices!

THE SOLUTION:
Z-Consistency Check examines:
1. ANATOMICAL CONTINUITY: Structures should be continuous in Z
2. INTENSITY CORRELATION: Adjacent slices should have correlated intensities
3. PATHOLOGY PERSISTENCE: Lesions should persist across multiple slices
4. EDGE ALIGNMENT: Boundaries should be smooth across Z

When Z-inconsistency is detected, it's a strong signal for hallucination.
"""

import numpy as np
import torch
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ZConsistencyResult:
    """Results from Z-consistency analysis."""
    # Overall scores
    z_consistency_score: float  # 0-1, higher = more consistent (better)
    hallucination_risk_score: float  # 0-1, higher = more suspicious

    # Detailed scores
    intensity_continuity: float  # Intensity consistency across Z
    structural_continuity: float  # Edge/structure consistency
    pathology_persistence: float  # Lesion consistency (if detected)
    anatomical_plausibility: float  # Overall anatomical sense

    # Problem areas
    suspicious_slices: List[int]  # Slice indices with anomalies
    discontinuity_locations: List[Dict]  # Location and type of discontinuities

    # Diagnosis
    has_z_violation: bool
    violation_type: str  # "none", "appearing_structure", "disappearing_structure", "intensity_jump"
    explanation: str


class ZConsistencyChecker:
    """
    Check 3D consistency of reconstructions.

    Tumors/lesions don't appear and disappear between slices.
    This check catches such anatomically implausible hallucinations.
    """

    def __init__(
        self,
        intensity_threshold: float = 0.15,  # Max acceptable intensity jump
        structure_threshold: float = 0.3,  # Max acceptable structural change
        min_persistence: int = 3,  # Minimum slices for valid pathology
    ):
        self.intensity_threshold = intensity_threshold
        self.structure_threshold = structure_threshold
        self.min_persistence = min_persistence

    def check_volume(
        self,
        volume: np.ndarray,  # (Z, H, W) or (Z, C, H, W)
        lesion_masks: Optional[np.ndarray] = None,  # (Z, H, W) if available
    ) -> ZConsistencyResult:
        """
        Check Z-consistency of a 3D volume.

        Args:
            volume: 3D volume (Z, H, W) or 4D with channels
            lesion_masks: Optional lesion masks per slice

        Returns:
            ZConsistencyResult with detailed analysis
        """
        # Handle different input formats
        if volume.ndim == 4:
            volume = volume.squeeze(1)  # Remove channel dim
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()

        Z, H, W = volume.shape

        # 1. Intensity continuity analysis
        intensity_results = self._analyze_intensity_continuity(volume)

        # 2. Structural continuity analysis
        structure_results = self._analyze_structural_continuity(volume)

        # 3. Pathology persistence (if lesion masks provided)
        if lesion_masks is not None:
            if isinstance(lesion_masks, torch.Tensor):
                lesion_masks = lesion_masks.cpu().numpy()
            pathology_results = self._analyze_pathology_persistence(lesion_masks)
        else:
            # Try to detect bright regions that might be pathology
            pathology_results = self._detect_potential_pathology(volume)

        # 4. Anatomical plausibility
        anatomy_score = self._compute_anatomical_plausibility(
            volume, intensity_results, structure_results
        )

        # Combine scores
        intensity_continuity = intensity_results['score']
        structural_continuity = structure_results['score']
        pathology_persistence = pathology_results['score']

        # Overall Z-consistency score
        z_consistency_score = (
            0.3 * intensity_continuity +
            0.3 * structural_continuity +
            0.25 * pathology_persistence +
            0.15 * anatomy_score
        )

        # Hallucination risk (inverse of consistency)
        hallucination_risk = 1.0 - z_consistency_score

        # Identify suspicious slices
        suspicious_slices = list(set(
            intensity_results['problem_slices'] +
            structure_results['problem_slices'] +
            pathology_results.get('problem_slices', [])
        ))
        suspicious_slices.sort()

        # Collect discontinuity locations
        discontinuities = []
        for loc in intensity_results.get('discontinuities', []):
            discontinuities.append({'type': 'intensity_jump', **loc})
        for loc in structure_results.get('discontinuities', []):
            discontinuities.append({'type': 'structural_change', **loc})
        for loc in pathology_results.get('discontinuities', []):
            discontinuities.append({'type': 'pathology_inconsistency', **loc})

        # Determine violation type
        has_violation = z_consistency_score < 0.7
        if not has_violation:
            violation_type = "none"
        elif pathology_results.get('appearing_structures', False):
            violation_type = "appearing_structure"
        elif pathology_results.get('disappearing_structures', False):
            violation_type = "disappearing_structure"
        elif intensity_results['max_jump'] > self.intensity_threshold:
            violation_type = "intensity_jump"
        else:
            violation_type = "structural_inconsistency"

        # Generate explanation
        explanation = self._generate_explanation(
            z_consistency_score, suspicious_slices, violation_type,
            intensity_results, structure_results, pathology_results
        )

        return ZConsistencyResult(
            z_consistency_score=float(z_consistency_score),
            hallucination_risk_score=float(hallucination_risk),
            intensity_continuity=float(intensity_continuity),
            structural_continuity=float(structural_continuity),
            pathology_persistence=float(pathology_persistence),
            anatomical_plausibility=float(anatomy_score),
            suspicious_slices=suspicious_slices,
            discontinuity_locations=discontinuities,
            has_z_violation=has_violation,
            violation_type=violation_type,
            explanation=explanation
        )

    def _analyze_intensity_continuity(self, volume: np.ndarray) -> Dict:
        """Analyze intensity consistency between adjacent slices."""
        Z = volume.shape[0]
        slice_means = [volume[z].mean() for z in range(Z)]
        slice_stds = [volume[z].std() for z in range(Z)]

        # Compute inter-slice intensity jumps
        jumps = []
        for z in range(1, Z):
            # Normalized intensity difference
            jump = abs(slice_means[z] - slice_means[z-1]) / (np.mean(slice_means) + 1e-8)
            jumps.append(jump)

        # Identify problem slices (large jumps)
        problem_slices = []
        discontinuities = []
        for z, jump in enumerate(jumps):
            if jump > self.intensity_threshold:
                problem_slices.append(z)
                problem_slices.append(z + 1)
                discontinuities.append({
                    'slice': z,
                    'magnitude': float(jump),
                    'description': f"Intensity jump of {jump:.2%} between slices {z} and {z+1}"
                })

        # Score: penalize large jumps
        max_jump = max(jumps) if jumps else 0
        mean_jump = np.mean(jumps) if jumps else 0
        score = max(0, 1.0 - max_jump / self.intensity_threshold)

        return {
            'score': score,
            'max_jump': max_jump,
            'mean_jump': mean_jump,
            'jumps': jumps,
            'problem_slices': list(set(problem_slices)),
            'discontinuities': discontinuities
        }

    def _analyze_structural_continuity(self, volume: np.ndarray) -> Dict:
        """Analyze structural consistency using edge detection."""
        Z = volume.shape[0]

        # Compute edge maps for each slice
        edge_maps = []
        for z in range(Z):
            edges = np.sqrt(
                ndimage.sobel(volume[z], axis=0)**2 +
                ndimage.sobel(volume[z], axis=1)**2
            )
            edge_maps.append(edges)

        # Compare edge maps between adjacent slices
        edge_diffs = []
        for z in range(1, Z):
            # Normalized edge difference
            diff = np.abs(edge_maps[z] - edge_maps[z-1]).mean()
            norm = (edge_maps[z].mean() + edge_maps[z-1].mean()) / 2 + 1e-8
            edge_diffs.append(diff / norm)

        # Identify problem slices
        problem_slices = []
        discontinuities = []
        for z, diff in enumerate(edge_diffs):
            if diff > self.structure_threshold:
                problem_slices.append(z)
                problem_slices.append(z + 1)
                discontinuities.append({
                    'slice': z,
                    'magnitude': float(diff),
                    'description': f"Structural change of {diff:.2f} between slices {z} and {z+1}"
                })

        # Score
        max_diff = max(edge_diffs) if edge_diffs else 0
        score = max(0, 1.0 - max_diff / self.structure_threshold)

        return {
            'score': score,
            'max_diff': max_diff,
            'edge_diffs': edge_diffs,
            'problem_slices': list(set(problem_slices)),
            'discontinuities': discontinuities
        }

    def _analyze_pathology_persistence(self, lesion_masks: np.ndarray) -> Dict:
        """Analyze persistence of detected pathology across slices."""
        Z = lesion_masks.shape[0]

        # Find slices with lesions
        slices_with_lesions = []
        for z in range(Z):
            if lesion_masks[z].sum() > 0:
                slices_with_lesions.append(z)

        if not slices_with_lesions:
            return {
                'score': 1.0,  # No pathology = consistent
                'problem_slices': [],
                'discontinuities': [],
                'appearing_structures': False,
                'disappearing_structures': False
            }

        # Check for gaps in lesion presence
        problem_slices = []
        discontinuities = []
        appearing_structures = False
        disappearing_structures = False

        # Group consecutive slices with lesions
        groups = []
        current_group = [slices_with_lesions[0]]
        for z in slices_with_lesions[1:]:
            if z == current_group[-1] + 1:
                current_group.append(z)
            else:
                groups.append(current_group)
                current_group = [z]
        groups.append(current_group)

        # Check each group
        for group in groups:
            if len(group) < self.min_persistence:
                # Suspicious: lesion appears for too few slices
                problem_slices.extend(group)
                if group[0] > 0 and lesion_masks[group[0] - 1].sum() == 0:
                    appearing_structures = True
                if group[-1] < Z - 1 and lesion_masks[group[-1] + 1].sum() == 0:
                    disappearing_structures = True
                discontinuities.append({
                    'slice': group[0],
                    'magnitude': len(group),
                    'description': f"Structure present only in {len(group)} consecutive slices ({group[0]}-{group[-1]})"
                })

        # Check for gaps between groups
        for i in range(1, len(groups)):
            gap_start = groups[i-1][-1] + 1
            gap_end = groups[i][0] - 1
            gap_size = gap_end - gap_start + 1

            if gap_size < 3:  # Small gap is suspicious
                problem_slices.extend(range(gap_start, gap_end + 1))
                discontinuities.append({
                    'slice': gap_start,
                    'magnitude': gap_size,
                    'description': f"Gap of {gap_size} slices ({gap_start}-{gap_end}) between lesions"
                })

        # Score based on persistence
        total_lesion_slices = len(slices_with_lesions)
        if total_lesion_slices == 0:
            score = 1.0
        else:
            # Penalize short runs and gaps
            longest_run = max(len(g) for g in groups)
            score = min(1.0, longest_run / (self.min_persistence * 2))
            if problem_slices:
                score *= 0.5  # Penalize discontinuities

        return {
            'score': score,
            'problem_slices': list(set(problem_slices)),
            'discontinuities': discontinuities,
            'appearing_structures': appearing_structures,
            'disappearing_structures': disappearing_structures,
            'groups': groups
        }

    def _detect_potential_pathology(self, volume: np.ndarray) -> Dict:
        """Detect potential pathology without ground truth masks."""
        Z = volume.shape[0]

        # Threshold to find bright regions (potential lesions)
        threshold = np.percentile(volume, 95)
        bright_regions = volume > threshold

        # Analyze persistence of bright regions
        return self._analyze_pathology_persistence(bright_regions)

    def _compute_anatomical_plausibility(
        self,
        volume: np.ndarray,
        intensity_results: Dict,
        structure_results: Dict
    ) -> float:
        """Compute overall anatomical plausibility score."""
        # Check for physiologically implausible patterns

        # 1. Volume should have smooth intensity gradient
        z_profile = [volume[z].mean() for z in range(volume.shape[0])]
        z_smoothness = 1.0 - np.std(np.diff(z_profile)) / (np.mean(z_profile) + 1e-8)

        # 2. Structure should be similar across adjacent slices
        structure_smoothness = 1.0 - np.std(structure_results['edge_diffs'])

        # 3. No sudden appearances/disappearances
        jump_penalty = np.sum(np.array(intensity_results['jumps']) > self.intensity_threshold)
        structure_penalty = np.sum(np.array(structure_results['edge_diffs']) > self.structure_threshold)

        total_slices = volume.shape[0]
        continuity_score = 1.0 - (jump_penalty + structure_penalty) / (2 * total_slices)

        return max(0, min(1, (z_smoothness + structure_smoothness + continuity_score) / 3))

    def _generate_explanation(
        self,
        score: float,
        suspicious_slices: List[int],
        violation_type: str,
        intensity_results: Dict,
        structure_results: Dict,
        pathology_results: Dict
    ) -> str:
        """Generate human-readable explanation of Z-consistency analysis."""
        if score >= 0.9:
            return (
                f"EXCELLENT Z-CONSISTENCY (score: {score:.2f}). "
                f"Volume shows smooth anatomical transitions across all slices. "
                f"No suspicious discontinuities detected."
            )
        elif score >= 0.7:
            return (
                f"ACCEPTABLE Z-CONSISTENCY (score: {score:.2f}). "
                f"Minor variations detected but within normal range. "
                f"Max intensity jump: {intensity_results['max_jump']:.2%}, "
                f"Max structural change: {structure_results['max_diff']:.2f}."
            )
        else:
            issues = []
            if violation_type == "appearing_structure":
                issues.append("structures appearing suddenly (possible hallucination)")
            elif violation_type == "disappearing_structure":
                issues.append("structures disappearing suddenly (possible hallucination)")
            elif violation_type == "intensity_jump":
                issues.append(f"large intensity jumps ({intensity_results['max_jump']:.2%})")
            else:
                issues.append("structural inconsistencies across slices")

            return (
                f"Z-CONSISTENCY VIOLATION DETECTED (score: {score:.2f}). "
                f"Issues: {', '.join(issues)}. "
                f"Suspicious slices: {suspicious_slices[:5]}{'...' if len(suspicious_slices) > 5 else ''}. "
                f"This pattern is anatomically implausible and suggests possible hallucination."
            )


def check_z_consistency_batch(
    volumes: List[np.ndarray],
    lesion_masks: Optional[List[np.ndarray]] = None
) -> List[ZConsistencyResult]:
    """
    Check Z-consistency for a batch of volumes.

    Args:
        volumes: List of 3D volumes
        lesion_masks: Optional list of lesion masks

    Returns:
        List of ZConsistencyResult objects
    """
    checker = ZConsistencyChecker()
    results = []

    for i, volume in enumerate(volumes):
        mask = lesion_masks[i] if lesion_masks is not None else None
        result = checker.check_volume(volume, mask)
        results.append(result)

    return results


def generate_z_consistency_report(results: List[ZConsistencyResult]) -> str:
    """Generate a summary report for Z-consistency analysis."""
    n_volumes = len(results)
    n_violations = sum(1 for r in results if r.has_z_violation)
    mean_score = np.mean([r.z_consistency_score for r in results])
    mean_risk = np.mean([r.hallucination_risk_score for r in results])

    # Count violation types
    violation_types = {}
    for r in results:
        vtype = r.violation_type
        violation_types[vtype] = violation_types.get(vtype, 0) + 1

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           Z-CONSISTENCY ANALYSIS REPORT                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  WHY THIS MATTERS:                                           ║
║  Tumors don't appear and disappear between slices!           ║
║  Z-inconsistency = anatomically implausible = hallucination  ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  SUMMARY                                                      ║
╠──────────────────────────────────────────────────────────────╣
║  Volumes analyzed:         {n_volumes:>8}                              ║
║  Volumes with violations:  {n_violations:>8} ({100*n_violations/max(1,n_volumes):>5.1f}%)               ║
║  Mean Z-consistency score: {mean_score:>8.3f}                          ║
║  Mean hallucination risk:  {mean_risk:>8.3f}                          ║
╠──────────────────────────────────────────────────────────────╣
║  VIOLATION TYPES                                              ║
╠──────────────────────────────────────────────────────────────╣
"""
    for vtype, count in sorted(violation_types.items()):
        report += f"║  {vtype:<25} {count:>6} ({100*count/n_volumes:>5.1f}%)           ║\n"

    report += """╠──────────────────────────────────────────────────────────────╣
║  RECOMMENDATION                                               ║
╠──────────────────────────────────────────────────────────────╣
"""
    if mean_score >= 0.85:
        report += "║  Z-consistency is EXCELLENT. Low hallucination risk.         ║\n"
    elif mean_score >= 0.7:
        report += "║  Z-consistency is ACCEPTABLE. Monitor edge cases.            ║\n"
    else:
        report += "║  Z-consistency is POOR. High hallucination risk!             ║\n"
        report += "║  Manual review of flagged volumes is REQUIRED.               ║\n"

    report += "╚══════════════════════════════════════════════════════════════╝"

    return report
