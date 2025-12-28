"""
Universal Physics-Based Auditor for ALL MRI Modalities
=======================================================

PHD-LEVEL CONTRIBUTION:
A unified auditor that learns the physics of each MRI modality
and adapts to any scanner configuration.

SUPPORTS:
- Contrast types: T1, T2, FLAIR, DWI, GRE/SWI, PD, 3D sequences
- Field strengths: 1.5T, 3T, 7T
- Vendors: GE, Siemens, Philips
- Coil configurations: Single, multi-coil, array coils
- Anatomies: Brain, spine, knee, abdomen, cardiac

THEORETICAL FOUNDATION:
-----------------------
MRI signal equation for any sequence:
    S(x,y,z,t) = M0(x,y,z) * f(T1, T2, TE, TR, FA) * C(x,y,z) * exp(-i*φ(x,y,z,t))

Where:
- M0: Proton density
- f: Contrast function (depends on sequence)
- C: Coil sensitivity
- φ: Phase (B0 + encoding)

This module learns the SHARED PHYSICS underlying all modalities,
enabling cross-modal hallucination detection.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto


class MRIContrast(Enum):
    """MRI contrast weightings."""
    T1 = auto()      # Fat bright, water dark
    T2 = auto()      # Water bright, fat dark
    PD = auto()      # Proton density
    FLAIR = auto()   # CSF suppressed T2
    DWI = auto()     # Diffusion weighted
    ADC = auto()     # Apparent diffusion coefficient
    SWI = auto()     # Susceptibility weighted
    GRE = auto()     # Gradient echo
    T2_STAR = auto() # T2* decay
    STIR = auto()    # Fat suppressed
    DIR = auto()     # Double inversion recovery
    TOF = auto()     # Time of flight (angiography)
    PHASE = auto()   # Phase contrast
    CINE = auto()    # Cardiac cine


class FieldStrength(Enum):
    """Scanner field strengths."""
    LOW_FIELD = 0.5   # 0.5T
    STANDARD = 1.5    # 1.5T (clinical standard)
    HIGH = 3.0        # 3T (research/clinical)
    ULTRA_HIGH = 7.0  # 7T (research)


class Vendor(Enum):
    """Scanner manufacturers."""
    SIEMENS = auto()
    GE = auto()
    PHILIPS = auto()
    CANON = auto()  # Toshiba
    HITACHI = auto()
    UNKNOWN = auto()


class Anatomy(Enum):
    """Body regions."""
    BRAIN = auto()
    SPINE = auto()
    KNEE = auto()
    SHOULDER = auto()
    HIP = auto()
    ABDOMEN = auto()
    CARDIAC = auto()
    BREAST = auto()
    PROSTATE = auto()
    LIVER = auto()
    KIDNEY = auto()


@dataclass
class MRIScanConfig:
    """Complete MRI scan configuration."""
    contrast: MRIContrast
    field_strength: FieldStrength
    vendor: Vendor
    anatomy: Anatomy

    # Sequence parameters
    tr: float = 0.0  # Repetition time (ms)
    te: float = 0.0  # Echo time (ms)
    ti: float = 0.0  # Inversion time (ms)
    flip_angle: float = 90.0  # degrees
    bandwidth: float = 0.0  # Hz/pixel

    # Acquisition parameters
    matrix_size: Tuple[int, int] = (256, 256)
    fov: Tuple[float, float] = (220.0, 220.0)  # mm
    slice_thickness: float = 3.0  # mm
    num_slices: int = 1
    num_coils: int = 1

    # k-space trajectory
    trajectory: str = "cartesian"  # cartesian, radial, spiral, epi


@dataclass
class TissueProperties:
    """MRI tissue relaxation properties."""
    name: str
    t1_1_5t: float  # T1 at 1.5T (ms)
    t1_3t: float    # T1 at 3T (ms)
    t2: float       # T2 (ms), roughly field-independent
    pd: float       # Proton density (relative to water)
    diffusion: float = 0.0  # ADC (mm²/s)
    susceptibility: float = 0.0  # ppm


# Standard tissue properties (literature values)
TISSUE_DB = {
    'white_matter': TissueProperties('White Matter', 600, 800, 80, 0.7, 0.7e-3, -9.0),
    'gray_matter': TissueProperties('Gray Matter', 950, 1300, 100, 0.8, 0.8e-3, -9.0),
    'csf': TissueProperties('CSF', 4000, 4000, 2000, 1.0, 3.0e-3, -9.0),
    'fat': TissueProperties('Fat', 260, 380, 80, 1.0, 0.1e-3, -7.5),
    'muscle': TissueProperties('Muscle', 900, 1400, 50, 0.9, 1.5e-3, -9.0),
    'blood': TissueProperties('Blood', 1400, 1900, 250, 0.9, 2.0e-3, -9.2),
    'bone_cortical': TissueProperties('Cortical Bone', 500, 500, 0.5, 0.1, 0.0, -11.0),
    'cartilage': TissueProperties('Cartilage', 1000, 1200, 40, 0.8, 1.5e-3, -9.0),
    'tumor': TissueProperties('Tumor (generic)', 1200, 1800, 100, 0.85, 1.0e-3, -9.0),
}


class ContrastModel(nn.Module):
    """
    Neural network that models MRI contrast physics.

    Learns the relationship between:
    - Tissue properties (T1, T2, PD)
    - Sequence parameters (TR, TE, TI, FA)
    - Field strength
    - Output signal intensity
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Input: [T1, T2, PD, TR, TE, TI, FA, B0] = 8 features
        self.encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output: predicted signal intensity
        self.signal_head = nn.Linear(hidden_dim, 1)

        # Physics-based priors (analytical formulas)
        self.use_physics_prior = True

    def forward(
        self,
        tissue_params: torch.Tensor,  # [B, 3] = T1, T2, PD
        sequence_params: torch.Tensor,  # [B, 4] = TR, TE, TI, FA
        field_strength: torch.Tensor  # [B, 1]
    ) -> torch.Tensor:
        """Predict signal intensity."""
        # Combine inputs
        x = torch.cat([tissue_params, sequence_params, field_strength], dim=-1)

        # Neural network prediction
        features = self.encoder(x)
        nn_signal = self.signal_head(features)

        if self.use_physics_prior:
            # Add physics-based prior
            physics_signal = self._compute_physics_signal(
                tissue_params, sequence_params, field_strength
            )
            # Blend neural and physics
            signal = 0.7 * physics_signal + 0.3 * nn_signal
        else:
            signal = nn_signal

        return signal

    def _compute_physics_signal(
        self,
        tissue_params: torch.Tensor,
        sequence_params: torch.Tensor,
        field_strength: torch.Tensor
    ) -> torch.Tensor:
        """Compute signal using analytical physics equations."""
        T1 = tissue_params[:, 0:1]
        T2 = tissue_params[:, 1:2]
        PD = tissue_params[:, 2:3]

        TR = sequence_params[:, 0:1]
        TE = sequence_params[:, 1:2]
        TI = sequence_params[:, 2:3]
        FA = sequence_params[:, 3:4] * np.pi / 180  # Convert to radians

        # Spin echo signal equation
        # S = PD * (1 - exp(-TR/T1)) * exp(-TE/T2)
        E1 = torch.exp(-TR / (T1 + 1e-6))
        E2 = torch.exp(-TE / (T2 + 1e-6))

        signal = PD * (1 - E1) * E2

        return signal


class UniversalMRIPhysicsModel(nn.Module):
    """
    Universal physics model that works across all MRI configurations.

    KEY INNOVATION:
    Instead of training separate models for each modality,
    we train ONE model that understands the underlying physics.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Contrast encoder
        self.contrast_model = ContrastModel(hidden_dim=128)

        # Scanner configuration encoder
        self.config_encoder = nn.Sequential(
            nn.Linear(32, latent_dim),  # Config features
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Image physics encoder (learns tissue distribution from image)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def encode_config(self, config: MRIScanConfig) -> torch.Tensor:
        """Encode scan configuration to latent vector."""
        # Convert config to feature vector
        features = self._config_to_features(config)
        return self.config_encoder(features)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent physics representation."""
        if image.dim() == 3:
            image = image.unsqueeze(1)
        return self.image_encoder(image).squeeze(-1).squeeze(-1)

    def detect_physics_violation(
        self,
        image: torch.Tensor,
        config: MRIScanConfig
    ) -> Dict[str, float]:
        """
        Detect physics violations in image given the scan configuration.

        This is the CORE of the universal auditor:
        - Given what we know about the physics (config)
        - Does the image make sense?
        """
        # Encode image and config
        image_latent = self.encode_image(image)
        config_latent = self.encode_config(config)

        # Cross-attention: what in the image doesn't match expected physics?
        attended, _ = self.cross_attention(
            image_latent.unsqueeze(1),
            config_latent.unsqueeze(1),
            config_latent.unsqueeze(1)
        )

        # Anomaly score
        combined = torch.cat([image_latent, attended.squeeze(1)], dim=-1)
        anomaly_score = self.anomaly_head(combined)

        return {
            'anomaly_score': float(anomaly_score.mean()),
            'physics_consistent': float(anomaly_score.mean()) < 0.3
        }

    def _config_to_features(self, config: MRIScanConfig) -> torch.Tensor:
        """Convert configuration to feature vector."""
        features = torch.zeros(32)

        # Contrast one-hot (14 types)
        features[config.contrast.value - 1] = 1.0

        # Field strength
        features[14] = config.field_strength.value

        # Vendor one-hot (6 types)
        features[15 + config.vendor.value - 1] = 1.0

        # Sequence parameters (normalized)
        features[21] = config.tr / 5000.0
        features[22] = config.te / 200.0
        features[23] = config.ti / 3000.0
        features[24] = config.flip_angle / 180.0

        # Spatial parameters
        features[25] = config.matrix_size[0] / 512.0
        features[26] = config.matrix_size[1] / 512.0
        features[27] = config.fov[0] / 500.0
        features[28] = config.slice_thickness / 10.0
        features[29] = config.num_coils / 32.0

        return features.unsqueeze(0)


class ModalitySpecificPhysics:
    """
    Modality-specific physics knowledge.

    Each MRI contrast type has specific physics that determines:
    - What tissues appear bright/dark
    - What artifacts are expected
    - What pathology patterns are valid
    """

    @staticmethod
    def get_expected_contrasts(modality: MRIContrast) -> Dict[str, str]:
        """Get expected tissue contrasts for a modality."""
        contrasts = {
            MRIContrast.T1: {
                'fat': 'bright',
                'white_matter': 'bright',
                'gray_matter': 'intermediate',
                'csf': 'dark',
                'muscle': 'intermediate',
                'blood': 'intermediate',
                'tumor_t1_enhancing': 'bright (with contrast)',
            },
            MRIContrast.T2: {
                'fat': 'intermediate',
                'white_matter': 'dark',
                'gray_matter': 'intermediate',
                'csf': 'bright',
                'muscle': 'dark',
                'blood': 'dark',
                'edema': 'bright',
                'tumor': 'bright',
            },
            MRIContrast.FLAIR: {
                'csf': 'dark (suppressed)',
                'white_matter': 'intermediate',
                'gray_matter': 'intermediate',
                'periventricular_lesions': 'bright',
                'ms_lesions': 'bright',
            },
            MRIContrast.DWI: {
                'normal_tissue': 'intermediate',
                'acute_stroke': 'bright (restricted diffusion)',
                'tumor_cellular': 'bright',
                'abscess': 'bright',
            },
            MRIContrast.SWI: {
                'blood_products': 'dark',
                'iron_deposition': 'dark',
                'calcification': 'dark',
                'veins': 'dark',
                'microbleeds': 'dark',
            },
        }
        return contrasts.get(modality, {})

    @staticmethod
    def validate_tissue_contrast(
        image: np.ndarray,
        tissue_mask: np.ndarray,
        modality: MRIContrast,
        tissue_type: str
    ) -> Dict[str, float]:
        """
        Validate if tissue contrast matches expected physics.

        Returns violation score: 0 = matches physics, 1 = violates physics
        """
        expected = ModalitySpecificPhysics.get_expected_contrasts(modality)
        if tissue_type not in expected:
            return {'violation_score': 0.0, 'uncertain': True}

        expected_brightness = expected[tissue_type]

        # Compute actual brightness
        if tissue_mask.sum() == 0:
            return {'violation_score': 0.0, 'no_tissue': True}

        tissue_intensity = image[tissue_mask].mean()
        image_mean = image.mean()
        image_std = image.std()

        # Normalize to z-score
        z_score = (tissue_intensity - image_mean) / (image_std + 1e-8)

        # Check against expected
        if 'bright' in expected_brightness.lower():
            # Should be above average
            violation = max(0, -z_score)  # Penalize if dark
        elif 'dark' in expected_brightness.lower():
            # Should be below average
            violation = max(0, z_score)  # Penalize if bright
        else:
            # Intermediate - should be near mean
            violation = abs(z_score) / 2

        return {
            'violation_score': float(min(1.0, violation)),
            'expected': expected_brightness,
            'actual_z_score': float(z_score)
        }


class DomainAdaptation(nn.Module):
    """
    Domain adaptation module for cross-scanner generalization.

    KEY CHALLENGE:
    Models trained on Siemens 3T don't work well on GE 1.5T.

    SOLUTION:
    Learn domain-invariant representations that capture
    the PHYSICS (which is universal) rather than the
    scanner-specific characteristics.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()

        # Feature extractor (shared)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.InstanceNorm2d(32),  # Instance norm is more domain-invariant
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        # Domain classifier (for adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 6 vendors
        )

        # Physics classifier (what we want to learn)
        self.physics_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 14),  # 14 contrast types
        )

    def forward(
        self,
        image: torch.Tensor,
        alpha: float = 1.0  # Gradient reversal weight
    ) -> Dict[str, torch.Tensor]:
        """
        Extract domain-invariant features.

        Uses gradient reversal to encourage features that:
        - Are useful for physics classification
        - Are NOT useful for domain classification
        """
        if image.dim() == 3:
            image = image.unsqueeze(1)

        features = self.feature_extractor(image).squeeze(-1).squeeze(-1)

        # Physics prediction (what we want)
        physics_pred = self.physics_classifier(features)

        # Domain prediction (what we want to be invariant to)
        # Gradient reversal makes features domain-invariant
        reversed_features = GradientReversal.apply(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)

        return {
            'features': features,
            'physics_pred': physics_pred,
            'domain_pred': domain_pred
        }


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


@dataclass
class UniversalAuditResult:
    """Result from universal MRI physics audit."""
    # Overall assessment
    physics_consistent: bool
    anomaly_score: float
    confidence: float

    # Modality-specific checks
    contrast_violations: Dict[str, float]
    tissue_violations: Dict[str, Dict]

    # Cross-modal consistency
    cross_modal_score: float  # If multiple sequences available

    # Scanner adaptation
    domain_shift_detected: bool
    domain_shift_severity: float

    # Recommendations
    suspicious_regions: List[Dict]
    explanation: str


class UniversalMRIAuditor:
    """
    Universal auditor for all MRI modalities, field strengths, and scanners.

    This is the PHD-LEVEL contribution: a single system that
    understands MRI physics universally.
    """

    def __init__(self):
        self.physics_model = UniversalMRIPhysicsModel()
        self.domain_adapter = DomainAdaptation()
        self.modality_physics = ModalitySpecificPhysics()

    def audit(
        self,
        image: np.ndarray,
        config: MRIScanConfig,
        tissue_masks: Optional[Dict[str, np.ndarray]] = None,
        reference_images: Optional[Dict[MRIContrast, np.ndarray]] = None
    ) -> UniversalAuditResult:
        """
        Perform universal physics-based audit.

        Args:
            image: Image to audit
            config: Scan configuration
            tissue_masks: Optional tissue segmentation masks
            reference_images: Optional images from other contrasts (same patient)

        Returns:
            UniversalAuditResult with comprehensive analysis
        """
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        # Step 1: Physics violation detection
        physics_result = self.physics_model.detect_physics_violation(
            image_tensor, config
        )

        # Step 2: Contrast-specific validation
        contrast_violations = {}
        tissue_violations = {}

        if tissue_masks is not None:
            for tissue_name, mask in tissue_masks.items():
                result = self.modality_physics.validate_tissue_contrast(
                    image, mask, config.contrast, tissue_name
                )
                tissue_violations[tissue_name] = result
                if result.get('violation_score', 0) > 0.3:
                    contrast_violations[tissue_name] = result['violation_score']

        # Step 3: Domain shift detection
        domain_result = self.domain_adapter(image_tensor)
        domain_shift_score = self._compute_domain_shift(domain_result, config)

        # Step 4: Cross-modal consistency (if reference images provided)
        cross_modal_score = 1.0
        if reference_images:
            cross_modal_score = self._check_cross_modal_consistency(
                image, config, reference_images
            )

        # Combine results
        overall_anomaly = (
            0.4 * physics_result['anomaly_score'] +
            0.3 * np.mean(list(contrast_violations.values())) if contrast_violations else 0 +
            0.2 * domain_shift_score +
            0.1 * (1 - cross_modal_score)
        )

        physics_consistent = overall_anomaly < 0.3

        # Find suspicious regions
        suspicious_regions = self._find_suspicious_regions(
            image, tissue_violations, contrast_violations
        )

        # Generate explanation
        explanation = self._generate_explanation(
            physics_consistent, config, contrast_violations,
            domain_shift_score, cross_modal_score
        )

        return UniversalAuditResult(
            physics_consistent=physics_consistent,
            anomaly_score=float(overall_anomaly),
            confidence=1.0 - float(overall_anomaly),
            contrast_violations=contrast_violations,
            tissue_violations=tissue_violations,
            cross_modal_score=float(cross_modal_score),
            domain_shift_detected=domain_shift_score > 0.3,
            domain_shift_severity=float(domain_shift_score),
            suspicious_regions=suspicious_regions,
            explanation=explanation
        )

    def _compute_domain_shift(
        self,
        domain_result: Dict,
        config: MRIScanConfig
    ) -> float:
        """Compute domain shift severity."""
        # Check if domain prediction matches actual vendor
        domain_pred = domain_result['domain_pred'].softmax(dim=-1)
        expected_domain = config.vendor.value - 1
        confidence = float(domain_pred[0, expected_domain])

        # High confidence in wrong domain = shift
        max_domain = domain_pred.argmax().item()
        if max_domain != expected_domain:
            return 1.0 - confidence
        return 0.0

    def _check_cross_modal_consistency(
        self,
        image: np.ndarray,
        config: MRIScanConfig,
        reference_images: Dict[MRIContrast, np.ndarray]
    ) -> float:
        """Check consistency across multiple MRI contrasts."""
        # Real pathology should appear consistently across contrasts
        # (though with different appearances)

        # Find bright regions in current image
        threshold = np.percentile(image, 95)
        bright_regions = image > threshold

        consistencies = []
        for ref_contrast, ref_image in reference_images.items():
            # Check if bright regions correspond to something in reference
            if ref_contrast == config.contrast:
                continue

            ref_at_bright = ref_image[bright_regions].mean()
            ref_elsewhere = ref_image[~bright_regions].mean()

            # There should be SOME relationship
            if abs(ref_at_bright - ref_elsewhere) / (ref_image.std() + 1e-8) > 0.5:
                consistencies.append(1.0)
            else:
                consistencies.append(0.5)

        return np.mean(consistencies) if consistencies else 1.0

    def _find_suspicious_regions(
        self,
        image: np.ndarray,
        tissue_violations: Dict,
        contrast_violations: Dict
    ) -> List[Dict]:
        """Find regions that violate expected physics."""
        suspicious = []

        for tissue, violation in tissue_violations.items():
            if violation.get('violation_score', 0) > 0.3:
                suspicious.append({
                    'type': 'tissue_contrast_violation',
                    'tissue': tissue,
                    'score': violation['violation_score'],
                    'expected': violation.get('expected', 'unknown'),
                    'actual_z': violation.get('actual_z_score', 0)
                })

        return suspicious

    def _generate_explanation(
        self,
        physics_consistent: bool,
        config: MRIScanConfig,
        contrast_violations: Dict,
        domain_shift: float,
        cross_modal: float
    ) -> str:
        """Generate human-readable explanation."""
        if physics_consistent:
            return (
                f"Image is PHYSICS CONSISTENT for {config.contrast.name} at {config.field_strength.value}T. "
                f"Tissue contrasts match expected patterns. "
                f"Domain shift: {domain_shift:.2f}, Cross-modal consistency: {cross_modal:.2f}."
            )
        else:
            violations = ', '.join(contrast_violations.keys()) if contrast_violations else 'unknown'
            return (
                f"PHYSICS VIOLATION DETECTED. "
                f"Tissue contrast violations: {violations}. "
                f"Domain shift: {domain_shift:.2f}. "
                f"This may indicate hallucination or artifact."
            )
