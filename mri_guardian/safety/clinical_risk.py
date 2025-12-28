"""
Clinical Risk Scoring and Confidence Mapping

Combines all safety signals into actionable clinical outputs:
1. Clinical Risk Score - single number for overall trustworthiness
2. Clinical Confidence Map - spatial visualization for radiologists
3. Risk Level Classification - categorical assessment

This is the interface between technical AI safety and clinical practice.

Novel contribution: First unified clinical risk framework for
AI MRI reconstruction that translates technical metrics into
clinician-understandable outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class RiskLevel(Enum):
    """Clinical risk level categories."""
    MINIMAL = "minimal"  # Safe to use without additional review
    LOW = "low"  # Generally safe, minor concerns
    MODERATE = "moderate"  # Use with caution, recommend review
    HIGH = "high"  # Significant concerns, require expert review
    CRITICAL = "critical"  # Do not use for clinical decisions


@dataclass
class ClinicalRiskAssessment:
    """Complete clinical risk assessment output."""
    risk_level: RiskLevel
    risk_score: float  # 0-1, higher = more risk
    confidence_in_assessment: float  # 0-1
    contributing_factors: Dict[str, float]
    recommendations: List[str]
    clinical_notes: str
    requires_expert_review: bool
    confidence_map: Optional[torch.Tensor] = None


class ClinicalRiskScorer:
    """
    Compute clinical risk score from multiple safety signals.

    Combines:
    - Uncertainty estimates
    - Physics violation scores
    - Distribution shift indicators
    - Hallucination detection results
    - Lesion integrity metrics
    - Bias indicators

    Into a single, clinically meaningful risk assessment.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        risk_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            weights: Importance weights for each signal
            risk_thresholds: Thresholds for risk level classification
        """
        # Default weights based on clinical importance
        self.weights = weights or {
            'uncertainty': 0.20,
            'physics_violation': 0.25,
            'distribution_shift': 0.15,
            'hallucination': 0.20,
            'lesion_integrity': 0.15,
            'bias': 0.05
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Risk level thresholds
        self.risk_thresholds = risk_thresholds or {
            'minimal': 0.1,
            'low': 0.25,
            'moderate': 0.5,
            'high': 0.75,
            'critical': 0.9
        }

    def _classify_risk_level(self, score: float) -> RiskLevel:
        """Classify risk score into categorical level."""
        if score < self.risk_thresholds['minimal']:
            return RiskLevel.MINIMAL
        elif score < self.risk_thresholds['low']:
            return RiskLevel.LOW
        elif score < self.risk_thresholds['moderate']:
            return RiskLevel.MODERATE
        elif score < self.risk_thresholds['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        factors: Dict[str, float]
    ) -> List[str]:
        """Generate clinical recommendations based on risk factors."""
        recommendations = []

        # Level-based recommendations
        if risk_level == RiskLevel.MINIMAL:
            recommendations.append("Reconstruction appears reliable for clinical use")
        elif risk_level == RiskLevel.LOW:
            recommendations.append("Minor uncertainties detected; standard clinical workflow appropriate")
        elif risk_level == RiskLevel.MODERATE:
            recommendations.append("CAUTION: Moderate uncertainty - verify critical findings with alternative methods")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("WARNING: High risk - require expert radiologist review before clinical use")
        else:
            recommendations.append("CRITICAL: Do not use for clinical decisions - recommend re-acquisition")

        # Factor-specific recommendations
        if factors.get('physics_violation', 0) > 0.5:
            recommendations.append("Physics violations detected - possible reconstruction artifacts")

        if factors.get('hallucination', 0) > 0.5:
            recommendations.append("Potential hallucinated features - verify anatomical structures")

        if factors.get('distribution_shift', 0) > 0.5:
            recommendations.append("Unusual scan characteristics - model may not generalize")

        if factors.get('lesion_integrity', 0) > 0.5:
            recommendations.append("Lesion preservation uncertain - verify pathology visibility")

        if factors.get('uncertainty', 0) > 0.5:
            recommendations.append("High model uncertainty - interpret with caution")

        return recommendations

    def _generate_clinical_notes(
        self,
        risk_level: RiskLevel,
        factors: Dict[str, float]
    ) -> str:
        """Generate clinical notes summarizing the assessment."""
        notes = []

        notes.append(f"AI Reconstruction Risk Assessment: {risk_level.value.upper()}")
        notes.append("")

        # Summarize key factors
        high_risk_factors = [k for k, v in factors.items() if v > 0.5]
        moderate_risk_factors = [k for k, v in factors.items() if 0.25 < v <= 0.5]

        if high_risk_factors:
            notes.append(f"High-risk factors: {', '.join(high_risk_factors)}")

        if moderate_risk_factors:
            notes.append(f"Moderate-risk factors: {', '.join(moderate_risk_factors)}")

        if not high_risk_factors and not moderate_risk_factors:
            notes.append("No significant risk factors identified")

        return "\n".join(notes)

    def compute(
        self,
        uncertainty_score: float = 0.0,
        physics_violation_score: float = 0.0,
        distribution_shift_score: float = 0.0,
        hallucination_score: float = 0.0,
        lesion_integrity_score: float = 0.0,  # Note: higher = worse (risk)
        bias_score: float = 0.0,
        spatial_uncertainty: Optional[torch.Tensor] = None,
        spatial_physics: Optional[torch.Tensor] = None,
        spatial_hallucination: Optional[torch.Tensor] = None
    ) -> ClinicalRiskAssessment:
        """
        Compute comprehensive clinical risk assessment.

        Args:
            uncertainty_score: Model uncertainty (0-1, higher = more uncertain)
            physics_violation_score: Physics violation severity (0-1)
            distribution_shift_score: OOD score (0-1)
            hallucination_score: Hallucination detection score (0-1)
            lesion_integrity_score: 1 - preservation_rate (0-1, higher = worse)
            bias_score: Subgroup bias score (0-1)
            spatial_*: Optional spatial maps for confidence visualization

        Returns:
            ClinicalRiskAssessment with complete analysis
        """
        # Collect factors
        factors = {
            'uncertainty': uncertainty_score,
            'physics_violation': physics_violation_score,
            'distribution_shift': distribution_shift_score,
            'hallucination': hallucination_score,
            'lesion_integrity': lesion_integrity_score,
            'bias': bias_score
        }

        # Compute weighted risk score
        risk_score = sum(
            self.weights[k] * v for k, v in factors.items()
        )

        # Apply non-linear transformation (sigmoid to emphasize high risks)
        risk_score_transformed = 1 / (1 + np.exp(-4 * (risk_score - 0.5)))

        # If any critical factor is very high, boost overall risk
        max_factor = max(factors.values())
        if max_factor > 0.8:
            risk_score_transformed = max(risk_score_transformed, 0.7)

        # Classify risk level
        risk_level = self._classify_risk_level(risk_score_transformed)

        # Confidence in our assessment
        # Higher when factors are clearly high or low, lower when borderline
        factor_confidences = []
        for v in factors.values():
            # Confidence is higher when values are extreme
            confidence = abs(v - 0.5) * 2
            factor_confidences.append(confidence)
        assessment_confidence = np.mean(factor_confidences)

        # Generate recommendations and notes
        recommendations = self._generate_recommendations(risk_level, factors)
        clinical_notes = self._generate_clinical_notes(risk_level, factors)

        # Requires expert review?
        requires_review = risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

        # Compute confidence map if spatial data available
        confidence_map = None
        if any(x is not None for x in [spatial_uncertainty, spatial_physics, spatial_hallucination]):
            confidence_map = self._compute_confidence_map(
                spatial_uncertainty, spatial_physics, spatial_hallucination
            )

        return ClinicalRiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score_transformed,
            confidence_in_assessment=assessment_confidence,
            contributing_factors=factors,
            recommendations=recommendations,
            clinical_notes=clinical_notes,
            requires_expert_review=requires_review,
            confidence_map=confidence_map
        )

    def _compute_confidence_map(
        self,
        uncertainty: Optional[torch.Tensor],
        physics: Optional[torch.Tensor],
        hallucination: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Combine spatial risk maps into confidence map."""
        maps = []

        for m, w in [
            (uncertainty, self.weights['uncertainty']),
            (physics, self.weights['physics_violation']),
            (hallucination, self.weights['hallucination'])
        ]:
            if m is not None:
                # Normalize
                m = m.float()
                if m.max() > m.min():
                    m = (m - m.min()) / (m.max() - m.min())
                maps.append(m * w)

        if not maps:
            return None

        # Resize all to same size
        target_size = maps[0].shape[-2:]
        resized = []
        for m in maps:
            while m.dim() < 4:
                m = m.unsqueeze(0)
            if m.shape[-2:] != target_size:
                m = F.interpolate(m, size=target_size, mode='bilinear', align_corners=False)
            resized.append(m.squeeze())

        # Combine (average)
        combined = torch.stack(resized).mean(dim=0)

        # Convert to confidence (1 - risk)
        confidence_map = 1.0 - combined

        return confidence_map


class ClinicalConfidenceMap:
    """
    Generate radiologist-friendly confidence visualizations.

    Creates spatial maps showing where the AI reconstruction
    can and cannot be trusted, with clinical interpretation.
    """

    def __init__(self):
        # Custom colormap: green (high confidence) -> yellow -> red (low confidence)
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
        self.confidence_cmap = LinearSegmentedColormap.from_list(
            'confidence', colors, N=256
        )

        # Overlay colormap for flagging specific issues
        self.flag_cmap = 'Reds'

    def generate(
        self,
        image: torch.Tensor,
        confidence_map: torch.Tensor,
        risk_assessment: ClinicalRiskAssessment,
        issue_maps: Optional[Dict[str, torch.Tensor]] = None
    ) -> plt.Figure:
        """
        Generate clinical confidence visualization.

        Args:
            image: Reconstructed MRI image
            confidence_map: Spatial confidence values (0-1)
            risk_assessment: Risk assessment results
            issue_maps: Optional dict of specific issue maps

        Returns:
            matplotlib Figure ready for display
        """
        # Ensure 2D
        while image.dim() > 2:
            image = image.squeeze(0)
        while confidence_map.dim() > 2:
            confidence_map = confidence_map.squeeze(0)

        image_np = image.cpu().numpy()
        conf_np = confidence_map.cpu().numpy()

        # Create figure
        n_cols = 3 + (len(issue_maps) if issue_maps else 0)
        fig = plt.figure(figsize=(4 * n_cols, 5))

        # 1. Original reconstruction
        ax1 = fig.add_subplot(1, n_cols, 1)
        ax1.imshow(image_np, cmap='gray')
        ax1.set_title('AI Reconstruction', fontsize=12)
        ax1.axis('off')

        # 2. Confidence map
        ax2 = fig.add_subplot(1, n_cols, 2)
        im = ax2.imshow(conf_np, cmap=self.confidence_cmap, vmin=0, vmax=1)
        ax2.set_title('Confidence Map', fontsize=12)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Confidence')

        # 3. Overlay
        ax3 = fig.add_subplot(1, n_cols, 3)
        ax3.imshow(image_np, cmap='gray')
        # Overlay low-confidence regions in red
        low_conf_mask = conf_np < 0.5
        overlay = np.zeros((*conf_np.shape, 4))
        overlay[..., 0] = 1.0  # Red channel
        overlay[..., 3] = (1 - conf_np) * low_conf_mask * 0.5  # Alpha
        ax3.imshow(overlay)
        ax3.set_title('Low Confidence Regions', fontsize=12)
        ax3.axis('off')

        # 4+ Issue-specific maps
        if issue_maps:
            for i, (name, issue_map) in enumerate(issue_maps.items()):
                ax = fig.add_subplot(1, n_cols, 4 + i)
                issue_np = issue_map.squeeze().cpu().numpy()
                ax.imshow(image_np, cmap='gray')
                ax.imshow(issue_np, cmap=self.flag_cmap, alpha=0.4 * (issue_np > 0.3))
                ax.set_title(f'{name} Flags', fontsize=12)
                ax.axis('off')

        # Add risk assessment summary
        risk_color = {
            RiskLevel.MINIMAL: '#2ecc71',
            RiskLevel.LOW: '#27ae60',
            RiskLevel.MODERATE: '#f39c12',
            RiskLevel.HIGH: '#e74c3c',
            RiskLevel.CRITICAL: '#c0392b'
        }[risk_assessment.risk_level]

        summary_text = (
            f"Risk Level: {risk_assessment.risk_level.value.upper()}\n"
            f"Risk Score: {risk_assessment.risk_score:.2f}\n"
            f"Assessment Confidence: {risk_assessment.confidence_in_assessment:.2f}"
        )

        fig.text(
            0.5, 0.02, summary_text,
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=risk_color, alpha=0.3)
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        return fig

    def generate_report_image(
        self,
        image: torch.Tensor,
        confidence_map: torch.Tensor,
        risk_assessment: ClinicalRiskAssessment,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate a single report image combining all visualizations.

        Returns:
            RGB numpy array suitable for saving/display
        """
        fig = self.generate(image, confidence_map, risk_assessment)

        # Convert to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close(fig)

        return data

    def generate_text_report(
        self,
        risk_assessment: ClinicalRiskAssessment
    ) -> str:
        """Generate text report for clinical documentation."""
        lines = [
            "=" * 60,
            "AI RECONSTRUCTION SAFETY ASSESSMENT REPORT",
            "=" * 60,
            "",
            f"RISK LEVEL: {risk_assessment.risk_level.value.upper()}",
            f"Risk Score: {risk_assessment.risk_score:.3f}",
            f"Assessment Confidence: {risk_assessment.confidence_in_assessment:.3f}",
            "",
            "-" * 40,
            "CONTRIBUTING FACTORS:",
            "-" * 40,
        ]

        for factor, score in sorted(
            risk_assessment.contributing_factors.items(),
            key=lambda x: -x[1]
        ):
            status = "⚠️ HIGH" if score > 0.5 else ("⚡ MODERATE" if score > 0.25 else "✓ OK")
            lines.append(f"  {factor:20s}: {score:.3f} [{status}]")

        lines.extend([
            "",
            "-" * 40,
            "RECOMMENDATIONS:",
            "-" * 40,
        ])

        for rec in risk_assessment.recommendations:
            lines.append(f"  • {rec}")

        lines.extend([
            "",
            "-" * 40,
            "CLINICAL NOTES:",
            "-" * 40,
            risk_assessment.clinical_notes,
            "",
            "=" * 60,
        ])

        if risk_assessment.requires_expert_review:
            lines.insert(4, "*** EXPERT REVIEW REQUIRED ***")
            lines.insert(5, "")

        return "\n".join(lines)


class IntegratedSafetyScorer:
    """
    Integrated safety scoring using all safety modules.

    This is the main interface that combines all safety components
    into a unified assessment pipeline.
    """

    def __init__(
        self,
        uncertainty_estimator=None,
        physics_detector=None,
        distribution_detector=None,
        hallucination_detector=None,
        lesion_verifier=None,
        bias_detector=None,
        device: str = 'cuda'
    ):
        self.uncertainty_estimator = uncertainty_estimator
        self.physics_detector = physics_detector
        self.distribution_detector = distribution_detector
        self.hallucination_detector = hallucination_detector
        self.lesion_verifier = lesion_verifier
        self.bias_detector = bias_detector
        self.device = device

        self.risk_scorer = ClinicalRiskScorer()
        self.confidence_mapper = ClinicalConfidenceMap()

    def evaluate(
        self,
        model: nn.Module,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
        return_visualization: bool = True
    ) -> Dict:
        """
        Run complete safety evaluation.

        Args:
            model: Reconstruction model
            masked_kspace: Undersampled k-space
            mask: Sampling mask
            ground_truth: Optional ground truth for validation
            return_visualization: Whether to generate visualizations

        Returns:
            Dictionary with all safety results
        """
        results = {}

        # Get reconstruction
        model.eval()
        with torch.no_grad():
            result = model(masked_kspace, mask)
            reconstruction = result['output'] if isinstance(result, dict) else result

        results['reconstruction'] = reconstruction

        # 1. Uncertainty estimation
        if self.uncertainty_estimator is not None:
            uncertainty = self.uncertainty_estimator.estimate(
                model, {'masked_kspace': masked_kspace, 'mask': mask}
            )
            results['uncertainty'] = uncertainty
            uncertainty_score = uncertainty.total.mean().item()
            spatial_uncertainty = uncertainty.total
        else:
            uncertainty_score = 0.0
            spatial_uncertainty = None

        # 2. Physics violation detection
        if self.physics_detector is not None:
            physics_report = self.physics_detector.detect(
                reconstruction, masked_kspace, mask
            )
            results['physics'] = physics_report
            physics_score = physics_report.total_severity
            spatial_physics = physics_report.violation_map
        else:
            physics_score = 0.0
            spatial_physics = None

        # 3. Distribution shift detection
        if self.distribution_detector is not None:
            from mri_guardian.data.kspace_ops import ifft2c
            image_domain = torch.abs(ifft2c(masked_kspace))
            shift_result = self.distribution_detector.detect(
                image_domain, masked_kspace
            )
            results['distribution_shift'] = shift_result
            shift_score = shift_result.ood_score
        else:
            shift_score = 0.0

        # 4. Hallucination detection
        if self.hallucination_detector is not None and ground_truth is not None:
            hall_result = self.hallucination_detector.detect(
                reconstruction, ground_truth
            )
            results['hallucination'] = hall_result
            hall_score = hall_result.get('detection_score', 0.0)
            spatial_hall = hall_result.get('detection_map', None)
        else:
            hall_score = 0.0
            spatial_hall = None

        # 5. Compute clinical risk
        risk_assessment = self.risk_scorer.compute(
            uncertainty_score=uncertainty_score,
            physics_violation_score=physics_score,
            distribution_shift_score=shift_score,
            hallucination_score=hall_score,
            spatial_uncertainty=spatial_uncertainty,
            spatial_physics=spatial_physics,
            spatial_hallucination=spatial_hall
        )
        results['risk_assessment'] = risk_assessment

        # 6. Generate visualization
        if return_visualization and risk_assessment.confidence_map is not None:
            fig = self.confidence_mapper.generate(
                reconstruction,
                risk_assessment.confidence_map,
                risk_assessment
            )
            results['visualization'] = fig

        # 7. Text report
        results['text_report'] = self.confidence_mapper.generate_text_report(
            risk_assessment
        )

        return results
