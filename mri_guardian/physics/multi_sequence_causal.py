"""
Multi-Sequence Causal Safety Model
===================================

PHD-LEVEL CONTRIBUTION #5:
An AI safety auditor that understands clinical relationships between
MRI sequences and enforces clinical logic.

SCIENTIFIC INSIGHT:
Different MRI sequences show pathology differently. A valid reconstruction
MUST respect these relationships. Violations indicate hallucinations.

CLINICAL RULES ENCODED:
-----------------------
1. TUMORS:
   - Bright on T2/FLAIR
   - Variable on T1 (bright with contrast)
   - May restrict on DWI if cellular

2. ACUTE STROKE:
   - MUST be bright on DWI (restricted diffusion)
   - Dark on ADC map
   - May be bright on FLAIR after ~6 hours

3. MS LESIONS:
   - Bright on FLAIR and T2
   - Usually not bright on T1 (unless enhancing)
   - Periventricular predilection

4. MICROBLEEDS:
   - Dark on SWI/GRE
   - NOT visible on FLAIR
   - NOT visible on standard T1/T2

5. EDEMA:
   - Bright on T2 and FLAIR
   - NOT bright on DWI (unless cytotoxic)
   - Dark on T1

6. HEMORRHAGE:
   - Complex evolution over time
   - Hyperacute: isointense T1, bright T2
   - Acute: dark T2 (deoxyhemoglobin)
   - Subacute: bright T1 (methemoglobin)

If AI reconstruction violates these rules → HALLUCINATION DETECTED

This is the foundation for clinically trustworthy generative MRI.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto


class PathologyType(Enum):
    """Types of pathology with specific multi-contrast signatures."""
    TUMOR = auto()
    ACUTE_STROKE = auto()
    MS_LESION = auto()
    MICROBLEED = auto()
    EDEMA = auto()
    HEMORRHAGE_HYPERACUTE = auto()
    HEMORRHAGE_ACUTE = auto()
    HEMORRHAGE_SUBACUTE = auto()
    HEMORRHAGE_CHRONIC = auto()
    ABSCESS = auto()
    INFARCT_OLD = auto()
    WHITE_MATTER_LESION = auto()


class MRISequence(Enum):
    """MRI sequences for cross-contrast analysis."""
    T1 = "t1"
    T1_CONTRAST = "t1_contrast"
    T2 = "t2"
    FLAIR = "flair"
    DWI = "dwi"
    ADC = "adc"
    SWI = "swi"
    GRE = "gre"
    PD = "pd"


class ContrastAppearance(Enum):
    """Expected signal intensity on different sequences."""
    BRIGHT = "bright"
    DARK = "dark"
    ISOINTENSE = "isointense"
    VARIABLE = "variable"
    NOT_VISIBLE = "not_visible"


@dataclass
class ClinicalRule:
    """A clinical rule defining expected cross-contrast behavior."""
    pathology: PathologyType
    sequence: MRISequence
    expected: ContrastAppearance
    mandatory: bool = False  # If True, violation = definite hallucination
    confidence: float = 0.95
    explanation: str = ""


# THE CLINICAL KNOWLEDGE BASE
# ===========================
# This is the core PhD contribution: encoding radiology knowledge into rules

CLINICAL_RULES: List[ClinicalRule] = [
    # ==================== ACUTE STROKE ====================
    ClinicalRule(
        pathology=PathologyType.ACUTE_STROKE,
        sequence=MRISequence.DWI,
        expected=ContrastAppearance.BRIGHT,
        mandatory=True,  # CRITICAL: Acute stroke MUST be bright on DWI
        explanation="Cytotoxic edema restricts water diffusion → bright DWI"
    ),
    ClinicalRule(
        pathology=PathologyType.ACUTE_STROKE,
        sequence=MRISequence.ADC,
        expected=ContrastAppearance.DARK,
        mandatory=True,  # CRITICAL: Low ADC confirms true restriction
        explanation="True restricted diffusion has low ADC values"
    ),
    ClinicalRule(
        pathology=PathologyType.ACUTE_STROKE,
        sequence=MRISequence.T2,
        expected=ContrastAppearance.VARIABLE,  # May be normal early
        mandatory=False,
        explanation="T2 changes may lag behind DWI by hours"
    ),
    ClinicalRule(
        pathology=PathologyType.ACUTE_STROKE,
        sequence=MRISequence.FLAIR,
        expected=ContrastAppearance.VARIABLE,  # Bright after ~6 hours
        mandatory=False,
        explanation="FLAIR positivity develops over 6-12 hours"
    ),

    # ==================== TUMORS ====================
    ClinicalRule(
        pathology=PathologyType.TUMOR,
        sequence=MRISequence.T2,
        expected=ContrastAppearance.BRIGHT,
        mandatory=False,  # Usually bright but not always
        confidence=0.85,
        explanation="Most tumors have increased water content → bright T2"
    ),
    ClinicalRule(
        pathology=PathologyType.TUMOR,
        sequence=MRISequence.FLAIR,
        expected=ContrastAppearance.BRIGHT,
        mandatory=False,
        confidence=0.85,
        explanation="Tumor edema is bright on FLAIR"
    ),
    ClinicalRule(
        pathology=PathologyType.TUMOR,
        sequence=MRISequence.T1_CONTRAST,
        expected=ContrastAppearance.VARIABLE,  # Enhancing vs non-enhancing
        mandatory=False,
        explanation="Enhancement depends on blood-brain barrier breakdown"
    ),
    ClinicalRule(
        pathology=PathologyType.TUMOR,
        sequence=MRISequence.SWI,
        expected=ContrastAppearance.NOT_VISIBLE,  # Unless hemorrhagic
        mandatory=False,
        explanation="SWI only shows tumor if hemorrhagic or calcified"
    ),

    # ==================== MS LESIONS ====================
    ClinicalRule(
        pathology=PathologyType.MS_LESION,
        sequence=MRISequence.FLAIR,
        expected=ContrastAppearance.BRIGHT,
        mandatory=True,  # MS lesions are ALWAYS bright on FLAIR
        explanation="Demyelination causes T2/FLAIR hyperintensity"
    ),
    ClinicalRule(
        pathology=PathologyType.MS_LESION,
        sequence=MRISequence.T2,
        expected=ContrastAppearance.BRIGHT,
        mandatory=True,
        explanation="MS lesions are bright on T2"
    ),
    ClinicalRule(
        pathology=PathologyType.MS_LESION,
        sequence=MRISequence.T1,
        expected=ContrastAppearance.VARIABLE,  # Dark if chronic (black holes)
        mandatory=False,
        explanation="Chronic MS lesions may be T1 hypointense (black holes)"
    ),
    ClinicalRule(
        pathology=PathologyType.MS_LESION,
        sequence=MRISequence.DWI,
        expected=ContrastAppearance.ISOINTENSE,  # Usually no restriction
        mandatory=False,
        explanation="Chronic MS lesions don't restrict diffusion"
    ),

    # ==================== MICROBLEEDS ====================
    ClinicalRule(
        pathology=PathologyType.MICROBLEED,
        sequence=MRISequence.SWI,
        expected=ContrastAppearance.DARK,
        mandatory=True,  # CRITICAL: Microbleeds MUST be dark on SWI
        explanation="Hemosiderin causes susceptibility artifact → dark SWI"
    ),
    ClinicalRule(
        pathology=PathologyType.MICROBLEED,
        sequence=MRISequence.GRE,
        expected=ContrastAppearance.DARK,
        mandatory=True,
        explanation="Hemosiderin is paramagnetic → dark on GRE"
    ),
    ClinicalRule(
        pathology=PathologyType.MICROBLEED,
        sequence=MRISequence.FLAIR,
        expected=ContrastAppearance.NOT_VISIBLE,
        mandatory=True,  # CRITICAL: Microbleeds are NOT visible on FLAIR
        explanation="FLAIR doesn't show susceptibility effects"
    ),
    ClinicalRule(
        pathology=PathologyType.MICROBLEED,
        sequence=MRISequence.T1,
        expected=ContrastAppearance.NOT_VISIBLE,
        mandatory=True,
        explanation="T1 doesn't show susceptibility effects at this size"
    ),

    # ==================== EDEMA ====================
    ClinicalRule(
        pathology=PathologyType.EDEMA,
        sequence=MRISequence.T2,
        expected=ContrastAppearance.BRIGHT,
        mandatory=True,
        explanation="Increased water content → bright T2"
    ),
    ClinicalRule(
        pathology=PathologyType.EDEMA,
        sequence=MRISequence.FLAIR,
        expected=ContrastAppearance.BRIGHT,
        mandatory=True,
        explanation="Vasogenic edema is bright on FLAIR"
    ),
    ClinicalRule(
        pathology=PathologyType.EDEMA,
        sequence=MRISequence.T1,
        expected=ContrastAppearance.DARK,
        mandatory=False,
        confidence=0.8,
        explanation="Edema is hypointense on T1"
    ),
    ClinicalRule(
        pathology=PathologyType.EDEMA,
        sequence=MRISequence.DWI,
        expected=ContrastAppearance.ISOINTENSE,  # Unless cytotoxic
        mandatory=False,
        explanation="Vasogenic edema doesn't restrict diffusion"
    ),

    # ==================== HEMORRHAGE EVOLUTION ====================
    # Hyperacute (< 6 hours)
    ClinicalRule(
        pathology=PathologyType.HEMORRHAGE_HYPERACUTE,
        sequence=MRISequence.T1,
        expected=ContrastAppearance.ISOINTENSE,
        mandatory=False,
        explanation="Oxyhemoglobin is isointense on T1"
    ),
    ClinicalRule(
        pathology=PathologyType.HEMORRHAGE_HYPERACUTE,
        sequence=MRISequence.T2,
        expected=ContrastAppearance.BRIGHT,
        mandatory=False,
        explanation="Oxyhemoglobin allows T2 bright signal"
    ),

    # Acute (6-72 hours)
    ClinicalRule(
        pathology=PathologyType.HEMORRHAGE_ACUTE,
        sequence=MRISequence.T2,
        expected=ContrastAppearance.DARK,
        mandatory=True,
        explanation="Deoxyhemoglobin causes T2 shortening → dark"
    ),
    ClinicalRule(
        pathology=PathologyType.HEMORRHAGE_ACUTE,
        sequence=MRISequence.SWI,
        expected=ContrastAppearance.DARK,
        mandatory=True,
        explanation="Deoxyhemoglobin is paramagnetic → dark SWI"
    ),

    # Subacute (3 days - 2 weeks)
    ClinicalRule(
        pathology=PathologyType.HEMORRHAGE_SUBACUTE,
        sequence=MRISequence.T1,
        expected=ContrastAppearance.BRIGHT,
        mandatory=True,  # CRITICAL: Subacute blood is BRIGHT on T1
        explanation="Methemoglobin causes T1 shortening → bright"
    ),
]


@dataclass
class CrossContrastViolation:
    """A detected violation of clinical logic."""
    rule: ClinicalRule
    observed: ContrastAppearance
    expected: ContrastAppearance
    severity: float  # 0-1, 1 = definite hallucination
    location: Optional[Tuple[int, int, int]] = None  # x, y, slice
    explanation: str = ""


@dataclass
class CausalAuditResult:
    """Result from multi-sequence causal audit."""
    # Overall verdict
    clinically_consistent: bool
    confidence: float

    # Violations found
    violations: List[CrossContrastViolation]
    mandatory_violations: int  # Count of definite hallucinations

    # Per-pathology analysis
    pathology_assessments: Dict[str, Dict]

    # Region-specific analysis
    suspicious_regions: List[Dict]

    # Clinical explanation
    clinical_summary: str
    recommendations: List[str]


class ClinicalLogicEngine:
    """
    Engine that encodes and enforces clinical MRI logic.

    This is the CORE of the causal safety model:
    Given observations across multiple sequences, does the
    pattern make clinical sense?
    """

    def __init__(self):
        self.rules = CLINICAL_RULES
        self._build_rule_index()

    def _build_rule_index(self):
        """Index rules by pathology and sequence for fast lookup."""
        self.rules_by_pathology: Dict[PathologyType, List[ClinicalRule]] = {}
        self.rules_by_sequence: Dict[MRISequence, List[ClinicalRule]] = {}
        self.mandatory_rules: List[ClinicalRule] = []

        for rule in self.rules:
            # By pathology
            if rule.pathology not in self.rules_by_pathology:
                self.rules_by_pathology[rule.pathology] = []
            self.rules_by_pathology[rule.pathology].append(rule)

            # By sequence
            if rule.sequence not in self.rules_by_sequence:
                self.rules_by_sequence[rule.sequence] = []
            self.rules_by_sequence[rule.sequence].append(rule)

            # Mandatory rules
            if rule.mandatory:
                self.mandatory_rules.append(rule)

    def get_expected_appearance(
        self,
        pathology: PathologyType,
        sequence: MRISequence
    ) -> Optional[ClinicalRule]:
        """Get expected appearance for pathology on sequence."""
        rules = self.rules_by_pathology.get(pathology, [])
        for rule in rules:
            if rule.sequence == sequence:
                return rule
        return None

    def check_consistency(
        self,
        pathology: PathologyType,
        observations: Dict[MRISequence, ContrastAppearance]
    ) -> List[CrossContrastViolation]:
        """
        Check if observations are clinically consistent for pathology.

        This is the KEY FUNCTION: given what we observe on multiple
        sequences, does it match expected clinical patterns?
        """
        violations = []
        rules = self.rules_by_pathology.get(pathology, [])

        for rule in rules:
            if rule.sequence in observations:
                observed = observations[rule.sequence]

                # Check for violation
                if self._is_violation(rule.expected, observed):
                    severity = 1.0 if rule.mandatory else 0.5
                    violations.append(CrossContrastViolation(
                        rule=rule,
                        observed=observed,
                        expected=rule.expected,
                        severity=severity,
                        explanation=self._explain_violation(rule, observed)
                    ))

        return violations

    def _is_violation(
        self,
        expected: ContrastAppearance,
        observed: ContrastAppearance
    ) -> bool:
        """Check if observed appearance violates expected."""
        # VARIABLE and NOT_VISIBLE have special handling
        if expected == ContrastAppearance.VARIABLE:
            return False  # Variable means anything is OK

        if expected == ContrastAppearance.NOT_VISIBLE:
            # If we observe it when it shouldn't be visible → violation
            return observed in [ContrastAppearance.BRIGHT, ContrastAppearance.DARK]

        # Direct comparison
        return expected != observed

    def _explain_violation(
        self,
        rule: ClinicalRule,
        observed: ContrastAppearance
    ) -> str:
        """Generate clinical explanation for violation."""
        return (
            f"CLINICAL LOGIC VIOLATION: {rule.pathology.name} should be "
            f"{rule.expected.value} on {rule.sequence.value}, but observed "
            f"{observed.value}. {rule.explanation}"
        )


class MultiSequenceAnalyzer(nn.Module):
    """
    Neural network that learns to classify contrast appearances.

    Given an image and ROI, determines if the region is:
    - BRIGHT
    - DARK
    - ISOINTENSE
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # ROI feature extractor
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        # Background feature extractor
        self.bg_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 16, hidden_dim),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # BRIGHT, DARK, ISOINTENSE, NOT_VISIBLE
        )

    def forward(
        self,
        image: torch.Tensor,
        roi_mask: torch.Tensor
    ) -> torch.Tensor:
        """Classify contrast appearance of ROI."""
        if image.dim() == 3:
            image = image.unsqueeze(1)

        # Extract ROI region
        roi_image = image * roi_mask.unsqueeze(1)
        roi_features = self.roi_encoder(roi_image).squeeze(-1).squeeze(-1)

        # Extract background
        bg_image = image * (1 - roi_mask.unsqueeze(1))
        bg_features = self.bg_encoder(bg_image)

        # Combine and classify
        combined = torch.cat([roi_features, bg_features], dim=-1)
        return self.classifier(combined)

    def classify_appearance(
        self,
        image: torch.Tensor,
        roi_mask: torch.Tensor
    ) -> ContrastAppearance:
        """Get predicted contrast appearance."""
        logits = self.forward(image, roi_mask)
        pred = logits.argmax(dim=-1).item()

        mapping = {
            0: ContrastAppearance.BRIGHT,
            1: ContrastAppearance.DARK,
            2: ContrastAppearance.ISOINTENSE,
            3: ContrastAppearance.NOT_VISIBLE,
        }
        return mapping[pred]


class MultiSequenceCausalAuditor:
    """
    The complete Multi-Sequence Causal Safety Model.

    PHD-LEVEL CONTRIBUTION:
    This auditor enforces clinical logic across multiple MRI sequences
    to detect biologically impossible reconstructions.

    WORKFLOW:
    1. Identify suspicious regions in each sequence
    2. For each region, classify appearance on all available sequences
    3. Hypothesize pathology type based on patterns
    4. Check if pattern matches clinical rules
    5. Flag violations as potential hallucinations
    """

    def __init__(self):
        self.logic_engine = ClinicalLogicEngine()
        self.appearance_classifier = MultiSequenceAnalyzer()

    def audit_multi_sequence(
        self,
        sequences: Dict[MRISequence, np.ndarray],
        roi_masks: Optional[Dict[str, np.ndarray]] = None,
        candidate_pathologies: Optional[List[PathologyType]] = None
    ) -> CausalAuditResult:
        """
        Perform multi-sequence causal audit.

        Args:
            sequences: Dict mapping sequence type to image
            roi_masks: Optional ROI masks to analyze
            candidate_pathologies: Pathologies to test for

        Returns:
            CausalAuditResult with comprehensive analysis
        """
        all_violations = []
        pathology_assessments = {}

        # If no ROIs provided, find bright regions automatically
        if roi_masks is None:
            roi_masks = self._find_candidate_regions(sequences)

        # If no pathologies specified, test all
        if candidate_pathologies is None:
            candidate_pathologies = list(PathologyType)

        # Analyze each ROI
        for roi_name, roi_mask in roi_masks.items():
            # Classify appearance on each available sequence
            observations = {}
            for seq_type, image in sequences.items():
                appearance = self._classify_region(image, roi_mask)
                observations[seq_type] = appearance

            # Test each pathology hypothesis
            for pathology in candidate_pathologies:
                violations = self.logic_engine.check_consistency(
                    pathology, observations
                )

                for v in violations:
                    v.location = roi_name
                    all_violations.append(v)

                # Store assessment
                key = f"{roi_name}_{pathology.name}"
                pathology_assessments[key] = {
                    'roi': roi_name,
                    'pathology': pathology.name,
                    'observations': {k.value: v.value for k, v in observations.items()},
                    'violations': len(violations),
                    'consistent': len(violations) == 0
                }

        # Count mandatory violations
        mandatory_count = sum(1 for v in all_violations if v.rule.mandatory)

        # Determine overall consistency
        clinically_consistent = mandatory_count == 0

        # Calculate confidence
        if len(all_violations) == 0:
            confidence = 0.95
        else:
            # More violations = lower confidence
            confidence = max(0.1, 1.0 - 0.2 * len(all_violations))

        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(
            all_violations, mandatory_count, sequences
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(all_violations)

        return CausalAuditResult(
            clinically_consistent=clinically_consistent,
            confidence=confidence,
            violations=all_violations,
            mandatory_violations=mandatory_count,
            pathology_assessments=pathology_assessments,
            suspicious_regions=[
                {'name': roi_name, 'mask_size': int(mask.sum())}
                for roi_name, mask in roi_masks.items()
            ],
            clinical_summary=clinical_summary,
            recommendations=recommendations
        )

    def _find_candidate_regions(
        self,
        sequences: Dict[MRISequence, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Find bright regions that might be pathology."""
        candidate_masks = {}

        # Look for bright regions on T2/FLAIR (most sensitive for pathology)
        for seq_type in [MRISequence.FLAIR, MRISequence.T2, MRISequence.DWI]:
            if seq_type in sequences:
                image = sequences[seq_type]
                threshold = np.percentile(image, 95)
                bright_mask = image > threshold

                # Simple connected component to find regions
                from scipy import ndimage
                labeled, num_features = ndimage.label(bright_mask)

                for i in range(1, min(num_features + 1, 5)):  # Max 5 regions
                    region_mask = (labeled == i).astype(float)
                    if region_mask.sum() > 10:  # Min 10 pixels
                        candidate_masks[f"{seq_type.value}_region_{i}"] = region_mask

        return candidate_masks

    def _classify_region(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray
    ) -> ContrastAppearance:
        """Classify contrast appearance of region."""
        # Simple intensity-based classification
        if roi_mask.sum() == 0:
            return ContrastAppearance.NOT_VISIBLE

        roi_intensity = image[roi_mask > 0.5].mean()
        bg_intensity = image[roi_mask <= 0.5].mean()
        image_std = image.std()

        relative = (roi_intensity - bg_intensity) / (image_std + 1e-8)

        if relative > 1.0:
            return ContrastAppearance.BRIGHT
        elif relative < -1.0:
            return ContrastAppearance.DARK
        else:
            return ContrastAppearance.ISOINTENSE

    def _generate_clinical_summary(
        self,
        violations: List[CrossContrastViolation],
        mandatory_count: int,
        sequences: Dict[MRISequence, np.ndarray]
    ) -> str:
        """Generate clinical summary of audit."""
        seq_names = [s.value for s in sequences.keys()]

        if mandatory_count > 0:
            return (
                f"CLINICAL LOGIC VIOLATION DETECTED. "
                f"Found {mandatory_count} mandatory rule violations across "
                f"{', '.join(seq_names)}. "
                f"The observed cross-contrast patterns are BIOLOGICALLY IMPOSSIBLE. "
                f"This strongly suggests AI hallucination."
            )
        elif len(violations) > 0:
            return (
                f"POTENTIAL INCONSISTENCY. "
                f"Found {len(violations)} non-critical violations across "
                f"{', '.join(seq_names)}. "
                f"Review recommended but not definite hallucination."
            )
        else:
            return (
                f"CLINICALLY CONSISTENT. "
                f"Cross-contrast patterns across {', '.join(seq_names)} "
                f"match expected clinical behavior. "
                f"No hallucinations detected by causal analysis."
            )

    def _generate_recommendations(
        self,
        violations: List[CrossContrastViolation]
    ) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []

        if len(violations) == 0:
            recommendations.append("No additional sequences required.")
            return recommendations

        # Recommend sequences that could resolve uncertainty
        violated_pathologies = set(v.rule.pathology for v in violations)

        for pathology in violated_pathologies:
            if pathology == PathologyType.ACUTE_STROKE:
                recommendations.append(
                    "Acquire ADC map to confirm/exclude restricted diffusion."
                )
            elif pathology == PathologyType.MICROBLEED:
                recommendations.append(
                    "Acquire SWI to evaluate for microbleeds."
                )
            elif pathology == PathologyType.TUMOR:
                recommendations.append(
                    "Acquire post-contrast T1 to evaluate enhancement."
                )

        recommendations.append(
            "Manual review by radiologist recommended for flagged regions."
        )

        return recommendations


def check_impossible_patterns(
    sequences: Dict[str, np.ndarray],
    roi_mask: np.ndarray
) -> Dict[str, bool]:
    """
    Quick check for biologically impossible patterns.

    This is a SIMPLE version of the causal auditor for fast detection.
    """
    results = {}

    # Check 1: Bright on DWI but NOT dark on ADC → T2 shine-through or hallucination
    if 'dwi' in sequences and 'adc' in sequences:
        dwi_bright = sequences['dwi'][roi_mask > 0.5].mean() > np.percentile(sequences['dwi'], 90)
        adc_dark = sequences['adc'][roi_mask > 0.5].mean() < np.percentile(sequences['adc'], 30)

        if dwi_bright and not adc_dark:
            results['dwi_without_adc_restriction'] = True
            # Could be T2 shine-through OR hallucination

    # Check 2: Visible on FLAIR but visible on SWI as dark → Should be hemorrhage
    if 'flair' in sequences and 'swi' in sequences:
        flair_bright = sequences['flair'][roi_mask > 0.5].mean() > np.percentile(sequences['flair'], 90)
        swi_dark = sequences['swi'][roi_mask > 0.5].mean() < np.percentile(sequences['swi'], 30)

        if flair_bright and swi_dark:
            # This pattern is unusual - need to rule out artifact
            results['flair_bright_swi_dark'] = True

    # Check 3: Microbleed visible on T2 → Impossible
    if 't2' in sequences and 'swi' in sequences:
        # Microbleeds should ONLY be visible on SWI, not T2
        t2_lesion = sequences['t2'][roi_mask > 0.5].mean() > np.percentile(sequences['t2'], 90)
        swi_dark = sequences['swi'][roi_mask > 0.5].mean() < np.percentile(sequences['swi'], 20)

        if t2_lesion and swi_dark:
            # If T2 bright AND SWI dark at same location → suspicious
            results['t2_swi_pattern_mismatch'] = True

    return results


def generate_causal_audit_report(result: CausalAuditResult) -> str:
    """Generate detailed clinical audit report."""
    lines = [
        "=" * 60,
        "MULTI-SEQUENCE CAUSAL SAFETY AUDIT REPORT",
        "=" * 60,
        "",
        f"VERDICT: {'CLINICALLY CONSISTENT' if result.clinically_consistent else 'VIOLATION DETECTED'}",
        f"Confidence: {result.confidence:.1%}",
        "",
        "SUMMARY:",
        result.clinical_summary,
        "",
    ]

    if result.violations:
        lines.extend([
            "-" * 40,
            f"VIOLATIONS ({len(result.violations)} found, {result.mandatory_violations} mandatory):",
        ])

        for i, v in enumerate(result.violations, 1):
            lines.extend([
                f"\n  [{i}] {v.rule.pathology.name} on {v.rule.sequence.value}:",
                f"      Expected: {v.expected.value}",
                f"      Observed: {v.observed.value}",
                f"      Severity: {'CRITICAL' if v.rule.mandatory else 'WARNING'}",
                f"      {v.explanation}",
            ])

    lines.extend([
        "",
        "-" * 40,
        "RECOMMENDATIONS:",
    ])
    for rec in result.recommendations:
        lines.append(f"  - {rec}")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
