"""
Failure Mode Analysis - Scientific Honesty
==========================================

THE CONCERN:
"Students cherry-pick the best examples. If a judge asks for a failure
case and you can't show one, you look dishonest or unscientific."

THE SOLUTION:
Explicitly analyze and document failure modes.
Admitting failure shows scientific maturity.

This module:
1. Identifies cases where Guardian failed
2. Categorizes failure modes
3. Provides explanations for WHY failures occurred
4. Documents limitations honestly

CRITICAL FOR ISEF:
"Here, the Guardian flagged a motion artifact as a tumor.
This shows that patient movement is a confounding factor
we need to address in future work."
"""

import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FailureType(Enum):
    """Types of auditor failures."""
    FALSE_POSITIVE = "false_positive"  # Flagged healthy tissue as suspicious
    FALSE_NEGATIVE = "false_negative"  # Missed a real hallucination
    MISLOCALIZATION = "mislocalization"  # Found something, wrong location
    SEVERITY_ERROR = "severity_error"  # Wrong severity estimate
    ARTIFACT_CONFUSION = "artifact_confusion"  # Confused artifact with pathology


@dataclass
class FailureCase:
    """A documented failure case."""
    case_id: str
    failure_type: FailureType
    image: np.ndarray
    ground_truth: np.ndarray
    prediction: np.ndarray
    error_map: np.ndarray

    # Context
    acceleration_factor: float
    lesion_size_mm: float
    noise_level: float

    # Analysis
    root_cause: str
    contributing_factors: List[str]
    suggested_mitigation: str

    # Metrics
    severity: float  # How bad was the failure
    clinical_impact: str  # What would happen clinically


class FailureModeAnalyzer:
    """
    Analyzes and documents failure modes of the auditor system.

    This is essential for scientific credibility.
    """

    def __init__(self, suspicion_threshold: float = 0.5):
        self.threshold = suspicion_threshold
        self.failure_cases: List[FailureCase] = []
        self.failure_statistics: Dict[FailureType, int] = {ft: 0 for ft in FailureType}

    def analyze_predictions(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        images: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """
        Analyze predictions to find and document failures.

        Args:
            predictions: Auditor suspicion maps
            ground_truths: True hallucination maps
            images: Original images
            metadata: Optional metadata for each case

        Returns:
            Analysis results with documented failures
        """
        n_samples = len(predictions)
        failures = []

        for i in range(n_samples):
            pred = predictions[i]
            gt = ground_truths[i]
            img = images[i]
            meta = metadata[i] if metadata else {}

            # Check for failures
            failure = self._check_for_failure(pred, gt, img, i, meta)
            if failure:
                failures.append(failure)
                self.failure_cases.append(failure)
                self.failure_statistics[failure.failure_type] += 1

        # Compute statistics
        total_failures = len(failures)
        failure_rate = total_failures / n_samples if n_samples > 0 else 0

        # Categorize by type
        by_type = {}
        for ft in FailureType:
            cases = [f for f in failures if f.failure_type == ft]
            by_type[ft.value] = {
                'count': len(cases),
                'rate': len(cases) / n_samples if n_samples > 0 else 0,
                'examples': cases[:3]  # Keep top 3 examples
            }

        return {
            'total_samples': n_samples,
            'total_failures': total_failures,
            'failure_rate': failure_rate,
            'by_type': by_type,
            'worst_failures': self._get_worst_failures(failures, n=5),
            'common_causes': self._identify_common_causes(failures)
        }

    def _check_for_failure(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        image: np.ndarray,
        case_idx: int,
        metadata: Dict
    ) -> Optional[FailureCase]:
        """Check if this prediction is a failure."""

        pred_binary = prediction > self.threshold
        gt_binary = ground_truth > 0.1

        # Calculate overlap
        true_positive = np.sum(pred_binary & gt_binary)
        false_positive = np.sum(pred_binary & ~gt_binary)
        false_negative = np.sum(~pred_binary & gt_binary)

        # Check for different failure types
        failure_type = None
        root_cause = ""
        contributing_factors = []
        clinical_impact = ""

        if false_positive > false_negative * 3 and false_positive > 100:
            # Significant false positives
            failure_type = FailureType.FALSE_POSITIVE
            root_cause = "Auditor flagged normal tissue as suspicious"
            clinical_impact = "Would trigger unnecessary follow-up imaging"

            # Analyze what was falsely flagged
            fp_region = pred_binary & ~gt_binary
            if self._is_near_edge(fp_region, image):
                contributing_factors.append("Near image boundary (edge artifacts)")
            if self._has_high_gradient(image, fp_region):
                contributing_factors.append("High gradient region (could be anatomy)")
            if metadata.get('motion', False):
                contributing_factors.append("Motion artifacts present")

        elif false_negative > 50 and gt_binary.sum() > 0:
            # Missed real hallucination
            failure_type = FailureType.FALSE_NEGATIVE
            root_cause = "Auditor missed a real hallucination"
            clinical_impact = "Hallucinated pathology could lead to misdiagnosis"

            # Analyze why it was missed
            fn_region = ~pred_binary & gt_binary
            hallucination_intensity = ground_truth[gt_binary].mean()
            if hallucination_intensity < 0.2:
                contributing_factors.append(f"Subtle hallucination (intensity: {hallucination_intensity:.2f})")
            if metadata.get('acceleration', 4) > 6:
                contributing_factors.append(f"High acceleration ({metadata.get('acceleration', 4)}x)")

        elif true_positive > 0 and self._is_mislocalized(pred_binary, gt_binary):
            failure_type = FailureType.MISLOCALIZATION
            root_cause = "Detected something suspicious, but in wrong location"
            clinical_impact = "Could lead radiologist to examine wrong region"
            contributing_factors.append("Spatial uncertainty in detection")

        if failure_type is None:
            return None

        # Create failure case
        error_map = np.abs(prediction - ground_truth)
        severity = float(error_map.max())

        return FailureCase(
            case_id=f"failure_{case_idx}",
            failure_type=failure_type,
            image=image,
            ground_truth=ground_truth,
            prediction=prediction,
            error_map=error_map,
            acceleration_factor=metadata.get('acceleration', 4.0),
            lesion_size_mm=metadata.get('lesion_size', 0.0),
            noise_level=metadata.get('noise', 0.0),
            root_cause=root_cause,
            contributing_factors=contributing_factors,
            suggested_mitigation=self._suggest_mitigation(failure_type, contributing_factors),
            severity=severity,
            clinical_impact=clinical_impact
        )

    def _is_near_edge(self, mask: np.ndarray, image: np.ndarray, margin: int = 20) -> bool:
        """Check if mask region is near image edge."""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return False
        h, w = image.shape
        return (coords[0].min() < margin or coords[0].max() > h - margin or
                coords[1].min() < margin or coords[1].max() > w - margin)

    def _has_high_gradient(self, image: np.ndarray, mask: np.ndarray, threshold: float = 0.2) -> bool:
        """Check if masked region has high gradient."""
        edges = np.sqrt(ndimage.sobel(image, 0)**2 + ndimage.sobel(image, 1)**2)
        if mask.sum() == 0:
            return False
        return edges[mask].mean() > threshold

    def _is_mislocalized(self, pred: np.ndarray, gt: np.ndarray) -> bool:
        """Check if prediction is mislocalized."""
        if gt.sum() == 0 or pred.sum() == 0:
            return False

        gt_center = np.array(ndimage.center_of_mass(gt))
        pred_center = np.array(ndimage.center_of_mass(pred))

        distance = np.linalg.norm(gt_center - pred_center)
        return distance > 30  # More than 30 pixels off

    def _suggest_mitigation(self, failure_type: FailureType, factors: List[str]) -> str:
        """Suggest mitigation strategy for failure."""
        suggestions = {
            FailureType.FALSE_POSITIVE: [
                "Increase suspicion threshold",
                "Add anatomical priors to reduce edge false positives",
                "Implement motion artifact detection"
            ],
            FailureType.FALSE_NEGATIVE: [
                "Decrease suspicion threshold for sensitive applications",
                "Improve detection of subtle hallucinations",
                "Add multi-scale analysis"
            ],
            FailureType.MISLOCALIZATION: [
                "Improve spatial resolution of suspicion map",
                "Add attention mechanism for better localization"
            ],
            FailureType.ARTIFACT_CONFUSION: [
                "Train separate artifact detector",
                "Add motion/artifact classification head"
            ]
        }

        base_suggestions = suggestions.get(failure_type, ["Further investigation needed"])

        if "Motion artifacts present" in factors:
            base_suggestions.insert(0, "Implement motion detection pre-filter")
        if "High acceleration" in str(factors):
            base_suggestions.insert(0, "Consider acceleration-adaptive thresholds")

        return base_suggestions[0] if base_suggestions else "Further investigation needed"

    def _get_worst_failures(self, failures: List[FailureCase], n: int = 5) -> List[FailureCase]:
        """Get the N worst failures by severity."""
        return sorted(failures, key=lambda f: f.severity, reverse=True)[:n]

    def _identify_common_causes(self, failures: List[FailureCase]) -> List[Tuple[str, int]]:
        """Identify most common contributing factors."""
        factor_counts = {}
        for f in failures:
            for factor in f.contributing_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1

        return sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    def generate_failure_report(self) -> str:
        """Generate a comprehensive failure mode report."""

        total = len(self.failure_cases)
        if total == 0:
            return "No failures documented yet."

        report = """
╔══════════════════════════════════════════════════════════════╗
║     FAILURE MODE ANALYSIS - LIMITATIONS & FAILURES           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  "Admitting failure shows scientific maturity."              ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  SUMMARY                                                     ║
╠──────────────────────────────────────────────────────────────╣
"""
        report += f"║  Total failures documented: {total:<31} ║\n"

        for ft, count in self.failure_statistics.items():
            if count > 0:
                report += f"║    {ft.value:<30} {count:>3} cases       ║\n"

        report += """╠──────────────────────────────────────────────────────────────╣
║  NOTABLE FAILURE CASES                                       ║
╠──────────────────────────────────────────────────────────────╣
"""

        for case in self.failure_cases[:3]:
            report += f"║  Case: {case.case_id:<52} ║\n"
            report += f"║  Type: {case.failure_type.value:<52} ║\n"
            report += f"║  Cause: {case.root_cause[:50]:<51} ║\n"
            report += f"║  Impact: {case.clinical_impact[:49]:<50} ║\n"
            report += f"║  Mitigation: {case.suggested_mitigation[:46]:<46} ║\n"
            report += "║                                                              ║\n"

        report += """╠──────────────────────────────────────────────────────────────╣
║  COMMON CONTRIBUTING FACTORS                                 ║
╠──────────────────────────────────────────────────────────────╣
"""
        common = self._identify_common_causes(self.failure_cases)
        for factor, count in common[:3]:
            report += f"║  • {factor[:45]:<45} ({count:>3})    ║\n"

        report += """╠──────────────────────────────────────════════════════════════╣
║  FUTURE WORK                                                 ║
╠──────────────────────────────────────────────────────────────╣
║  Based on this analysis, key improvements needed:            ║
"""
        mitigations = set(c.suggested_mitigation for c in self.failure_cases[:5])
        for m in list(mitigations)[:3]:
            report += f"║  • {m[:56]:<56} ║\n"

        report += "╚══════════════════════════════════════════════════════════════╝"

        return report


def create_failure_examples() -> List[FailureCase]:
    """
    Create example failure cases for demonstration.

    These are realistic failures that should be included in the presentation
    to show scientific honesty.
    """
    examples = []

    # Example 1: Motion artifact confused as pathology
    examples.append(FailureCase(
        case_id="motion_confusion_001",
        failure_type=FailureType.FALSE_POSITIVE,
        image=np.zeros((256, 256)),  # Placeholder
        ground_truth=np.zeros((256, 256)),
        prediction=np.zeros((256, 256)),
        error_map=np.zeros((256, 256)),
        acceleration_factor=4.0,
        lesion_size_mm=0.0,
        noise_level=0.05,
        root_cause="Motion artifact created edge enhancement that mimicked lesion contrast",
        contributing_factors=[
            "Patient motion during scan",
            "Edge enhancement pattern similar to lesion boundary",
            "Motion artifacts not explicitly modeled in training"
        ],
        suggested_mitigation="Implement motion artifact detection as pre-filter",
        severity=0.75,
        clinical_impact="Would trigger unnecessary follow-up scan, wasting resources"
    ))

    # Example 2: Subtle hallucination missed
    examples.append(FailureCase(
        case_id="subtle_miss_002",
        failure_type=FailureType.FALSE_NEGATIVE,
        image=np.zeros((256, 256)),
        ground_truth=np.zeros((256, 256)),
        prediction=np.zeros((256, 256)),
        error_map=np.zeros((256, 256)),
        acceleration_factor=8.0,
        lesion_size_mm=3.0,
        noise_level=0.08,
        root_cause="Hallucination intensity below detection threshold at 8x acceleration",
        contributing_factors=[
            "High acceleration factor (8x) increased noise floor",
            "Small lesion size (3mm) near resolution limit",
            "Hallucination intensity only 15% above background"
        ],
        suggested_mitigation="Use acceleration-adaptive detection thresholds",
        severity=0.6,
        clinical_impact="Small hallucinated lesion could lead to unnecessary biopsy"
    ))

    return examples
