"""
Dirty Test Set Generator
=========================

Creates corrupted MRI data to stress-test the auditor.

Corruption types:
1. Gaussian noise (sensor noise)
2. Motion artifacts (k-space phase errors)
3. Intensity scaling (calibration errors)
4. RF interference patterns
5. Gradient nonlinearity artifacts
6. Zipper artifacts
7. Wrap-around (aliasing)

The auditor should flag these specific corruptions with
EXPLANATIONS, not just generic "OOD" errors.

CRITICAL: Add "Why is this OOD?" explanation feature.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CorruptionType(Enum):
    """Types of MRI corruptions."""
    GAUSSIAN_NOISE = "gaussian_noise"
    MOTION_ARTIFACT = "motion_artifact"
    INTENSITY_SHIFT = "intensity_shift"
    RF_INTERFERENCE = "rf_interference"
    GRADIENT_ARTIFACT = "gradient_artifact"
    ZIPPER_ARTIFACT = "zipper_artifact"
    ALIASING = "aliasing"
    GIBBS_RINGING = "gibbs_ringing"
    CHEMICAL_SHIFT = "chemical_shift"
    SUSCEPTIBILITY = "susceptibility"


@dataclass
class CorruptionResult:
    """Result of applying a corruption."""
    corrupted_image: np.ndarray
    corruption_type: CorruptionType
    corruption_mask: np.ndarray  # Where the corruption is
    severity: float
    explanation: str
    detection_hint: str  # What the auditor should look for


class DirtyTestGenerator:
    """
    Generates corrupted MRI data for auditor stress testing.

    Each corruption comes with:
    1. The corrupted image
    2. A mask showing where corruption is
    3. An explanation for debugging
    4. A detection hint for the auditor
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def add_gaussian_noise(
        self,
        image: np.ndarray,
        sigma: float = 0.1
    ) -> CorruptionResult:
        """
        Add Gaussian noise (simulates sensor/thermal noise).

        Detection: Uniform noise distribution, no spatial pattern.
        """
        noise = self.rng.normal(0, sigma, image.shape)
        corrupted = np.clip(image + noise, 0, 1)

        return CorruptionResult(
            corrupted_image=corrupted,
            corruption_type=CorruptionType.GAUSSIAN_NOISE,
            corruption_mask=np.ones_like(image) * sigma,
            severity=sigma,
            explanation=f"Gaussian noise with σ={sigma:.3f}. "
                       f"Simulates thermal noise in RF coils.",
            detection_hint="Look for uniform noise distribution across image. "
                          "Check local variance - should be constant everywhere."
        )

    def add_motion_artifact(
        self,
        image: np.ndarray,
        motion_amount: float = 0.2
    ) -> CorruptionResult:
        """
        Add motion artifact (k-space phase errors).

        Real motion causes k-space phase corruption, resulting in
        ghosting and blurring in image domain.

        Detection: Periodic ghosting, directional blurring.
        """
        # Simulate k-space
        kspace = np.fft.fftshift(np.fft.fft2(image))

        # Motion causes phase errors proportional to position
        h, w = image.shape
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Random motion direction
        angle = self.rng.uniform(0, 2 * np.pi)
        motion_vector = np.cos(angle) * X + np.sin(angle) * Y

        # Phase error
        phase_error = motion_amount * motion_vector * 2 * np.pi
        motion_phase = np.exp(1j * phase_error)

        # Apply to k-space
        corrupted_kspace = kspace * motion_phase

        # Convert back
        corrupted = np.abs(np.fft.ifft2(np.fft.ifftshift(corrupted_kspace)))
        corrupted = np.clip(corrupted, 0, 1)

        # Motion mask: edges show most artifact
        edges = np.sqrt(ndimage.sobel(image, 0)**2 + ndimage.sobel(image, 1)**2)
        corruption_mask = edges * motion_amount

        return CorruptionResult(
            corrupted_image=corrupted.astype(np.float32),
            corruption_type=CorruptionType.MOTION_ARTIFACT,
            corruption_mask=corruption_mask.astype(np.float32),
            severity=motion_amount,
            explanation=f"Motion artifact with magnitude {motion_amount:.2f}. "
                       f"Direction: {np.degrees(angle):.1f}°. "
                       f"Simulates patient movement during scan.",
            detection_hint="Look for directional ghosting and edge blurring. "
                          "Check k-space for unexpected phase patterns."
        )

    def add_intensity_shift(
        self,
        image: np.ndarray,
        shift_type: str = 'bias_field'
    ) -> CorruptionResult:
        """
        Add intensity inhomogeneity (B1 field artifact).

        Detection: Smooth, low-frequency intensity variation.
        """
        h, w = image.shape

        if shift_type == 'bias_field':
            # Smooth bias field
            y = np.linspace(-1, 1, h)
            x = np.linspace(-1, 1, w)
            Y, X = np.meshgrid(y, x, indexing='ij')

            # Random polynomial bias field
            bias = 1.0 + 0.3 * (X**2 + Y**2) * self.rng.uniform(0.5, 1.5)
            bias = ndimage.gaussian_filter(bias, sigma=30)

            corrupted = np.clip(image * bias, 0, 1)
            corruption_mask = np.abs(bias - 1)

        elif shift_type == 'random_scale':
            # Random scaling
            scale = self.rng.uniform(0.7, 1.3)
            corrupted = np.clip(image * scale, 0, 1)
            corruption_mask = np.ones_like(image) * abs(scale - 1)

        return CorruptionResult(
            corrupted_image=corrupted.astype(np.float32),
            corruption_type=CorruptionType.INTENSITY_SHIFT,
            corruption_mask=corruption_mask.astype(np.float32),
            severity=float(np.mean(corruption_mask)),
            explanation=f"Intensity inhomogeneity ({shift_type}). "
                       f"Simulates B1 field non-uniformity.",
            detection_hint="Look for smooth intensity gradients. "
                          "Histogram will show shifted/widened distribution."
        )

    def add_zipper_artifact(
        self,
        image: np.ndarray,
        n_lines: int = 5,
        intensity: float = 0.3
    ) -> CorruptionResult:
        """
        Add zipper artifact (RF interference).

        Appears as bright/dark lines in fixed positions.
        Detection: Periodic lines at fixed frequency locations.
        """
        h, w = image.shape
        corrupted = image.copy()
        corruption_mask = np.zeros_like(image)

        # Random positions for zipper lines
        positions = self.rng.choice(range(10, w-10), size=n_lines, replace=False)

        for pos in positions:
            # Alternating bright/dark pattern
            pattern = np.sin(np.linspace(0, 20*np.pi, h)) * intensity
            corrupted[:, pos] = np.clip(corrupted[:, pos] + pattern, 0, 1)
            corruption_mask[:, pos] = intensity

        return CorruptionResult(
            corrupted_image=corrupted.astype(np.float32),
            corruption_type=CorruptionType.ZIPPER_ARTIFACT,
            corruption_mask=corruption_mask.astype(np.float32),
            severity=intensity,
            explanation=f"Zipper artifact: {n_lines} lines at intensity {intensity:.2f}. "
                       f"Caused by RF interference or spike noise.",
            detection_hint="Look for vertical/horizontal lines at fixed positions. "
                          "Check k-space for spike noise at specific frequencies."
        )

    def add_aliasing(
        self,
        image: np.ndarray,
        wrap_factor: float = 0.3
    ) -> CorruptionResult:
        """
        Add aliasing/wrap-around artifact.

        Occurs when FOV is too small for object.
        Detection: Image content appearing on wrong side.
        """
        h, w = image.shape
        corrupted = image.copy()

        # Simulate wrap-around by adding shifted copies
        shift = int(h * wrap_factor)

        # Wrap from top
        wrap_region = image[:shift, :] * 0.3
        corrupted[-shift:, :] = np.clip(corrupted[-shift:, :] + wrap_region, 0, 1)

        # Wrap from bottom
        wrap_region = image[-shift:, :] * 0.3
        corrupted[:shift, :] = np.clip(corrupted[:shift, :] + wrap_region, 0, 1)

        corruption_mask = np.zeros_like(image)
        corruption_mask[:shift, :] = wrap_factor
        corruption_mask[-shift:, :] = wrap_factor

        return CorruptionResult(
            corrupted_image=corrupted.astype(np.float32),
            corruption_type=CorruptionType.ALIASING,
            corruption_mask=corruption_mask.astype(np.float32),
            severity=wrap_factor,
            explanation=f"Aliasing artifact with wrap factor {wrap_factor:.2f}. "
                       f"Caused by insufficient FOV or sampling.",
            detection_hint="Look for duplicated content at image edges. "
                          "Phase-encode direction shows most wrap-around."
        )

    def add_gibbs_ringing(
        self,
        image: np.ndarray,
        severity: float = 0.5
    ) -> CorruptionResult:
        """
        Add Gibbs ringing (truncation artifact).

        Appears as oscillating lines near sharp edges.
        Detection: Periodic oscillations near high-contrast boundaries.
        """
        # Simulate by truncating k-space
        kspace = np.fft.fftshift(np.fft.fft2(image))
        h, w = image.shape

        # Create truncation mask
        center_frac = 1.0 - severity * 0.3
        mask = np.zeros_like(kspace)
        ch, cw = h // 2, w // 2
        rh, rw = int(h * center_frac / 2), int(w * center_frac / 2)
        mask[ch-rh:ch+rh, cw-rw:cw+rw] = 1

        # Truncated k-space
        truncated_kspace = kspace * mask

        corrupted = np.abs(np.fft.ifft2(np.fft.ifftshift(truncated_kspace)))
        corrupted = (corrupted - corrupted.min()) / (corrupted.max() - corrupted.min() + 1e-8)

        # Ringing mask: edges
        edges = np.sqrt(ndimage.sobel(image, 0)**2 + ndimage.sobel(image, 1)**2)
        corruption_mask = ndimage.gaussian_filter(edges, sigma=5) * severity

        return CorruptionResult(
            corrupted_image=corrupted.astype(np.float32),
            corruption_type=CorruptionType.GIBBS_RINGING,
            corruption_mask=corruption_mask.astype(np.float32),
            severity=severity,
            explanation=f"Gibbs ringing with severity {severity:.2f}. "
                       f"Caused by k-space truncation.",
            detection_hint="Look for oscillating intensity patterns near sharp edges. "
                          "Frequency depends on truncation level."
        )

    def generate_dirty_test_set(
        self,
        clean_images: List[np.ndarray],
        corruptions_per_image: int = 3
    ) -> List[Dict]:
        """
        Generate a complete dirty test set with multiple corruptions.

        Returns list of dicts with corrupted images and metadata.
        """
        dirty_set = []

        corruption_functions = [
            lambda img: self.add_gaussian_noise(img, sigma=self.rng.uniform(0.05, 0.2)),
            lambda img: self.add_motion_artifact(img, motion_amount=self.rng.uniform(0.1, 0.4)),
            lambda img: self.add_intensity_shift(img, shift_type='bias_field'),
            lambda img: self.add_zipper_artifact(img, n_lines=self.rng.randint(2, 8)),
            lambda img: self.add_aliasing(img, wrap_factor=self.rng.uniform(0.1, 0.3)),
            lambda img: self.add_gibbs_ringing(img, severity=self.rng.uniform(0.3, 0.7))
        ]

        for img_idx, clean_image in enumerate(clean_images):
            # Apply random corruptions
            selected_corruptions = self.rng.choice(
                len(corruption_functions),
                size=min(corruptions_per_image, len(corruption_functions)),
                replace=False
            )

            for corr_idx in selected_corruptions:
                result = corruption_functions[corr_idx](clean_image)

                dirty_set.append({
                    'original_idx': img_idx,
                    'clean_image': clean_image,
                    'corrupted_image': result.corrupted_image,
                    'corruption_type': result.corruption_type.value,
                    'corruption_mask': result.corruption_mask,
                    'severity': result.severity,
                    'explanation': result.explanation,
                    'detection_hint': result.detection_hint
                })

        return dirty_set


class CorruptionDetector:
    """
    Detects and explains specific types of corruption.

    This is the "Why is this OOD?" feature.
    """

    def __init__(self):
        self.detection_thresholds = {
            'noise_variance': 0.01,
            'motion_score': 0.1,
            'bias_field_range': 0.3,
            'zipper_score': 0.1,
            'ringing_score': 0.1
        }

    def detect_corruption_type(
        self,
        image: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Analyze image and detect specific corruption types.

        Returns explanation of what's wrong and why.
        """
        detections = []

        # 1. Check for noise
        noise_score = self._detect_noise(image)
        if noise_score > self.detection_thresholds['noise_variance']:
            detections.append({
                'type': 'GAUSSIAN_NOISE',
                'confidence': min(noise_score / 0.05, 1.0),
                'explanation': f"Elevated noise detected (variance: {noise_score:.4f}). "
                              f"Possible causes: thermal noise, low SNR acquisition."
            })

        # 2. Check for motion
        motion_score = self._detect_motion(image)
        if motion_score > self.detection_thresholds['motion_score']:
            detections.append({
                'type': 'MOTION_ARTIFACT',
                'confidence': min(motion_score / 0.3, 1.0),
                'explanation': f"Motion artifact detected (score: {motion_score:.3f}). "
                              f"Look for ghosting and edge blurring."
            })

        # 3. Check for intensity inhomogeneity
        bias_range = self._detect_bias_field(image)
        if bias_range > self.detection_thresholds['bias_field_range']:
            detections.append({
                'type': 'INTENSITY_INHOMOGENEITY',
                'confidence': min(bias_range / 0.5, 1.0),
                'explanation': f"Intensity inhomogeneity detected (range: {bias_range:.3f}). "
                              f"Possible B1 field non-uniformity."
            })

        # 4. Check for zipper artifacts
        zipper_score = self._detect_zipper(image)
        if zipper_score > self.detection_thresholds['zipper_score']:
            detections.append({
                'type': 'ZIPPER_ARTIFACT',
                'confidence': min(zipper_score / 0.3, 1.0),
                'explanation': f"Zipper artifact detected (score: {zipper_score:.3f}). "
                              f"Possible RF interference or spike noise."
            })

        # Generate summary
        if len(detections) == 0:
            summary = "No specific artifacts detected. Image appears clean."
            is_corrupted = False
        else:
            summary = f"Detected {len(detections)} corruption type(s): "
            summary += ", ".join([d['type'] for d in detections])
            is_corrupted = True

        return {
            'is_corrupted': is_corrupted,
            'detections': detections,
            'summary': summary,
            'overall_confidence': max([d['confidence'] for d in detections]) if detections else 0
        }

    def _detect_noise(self, image: np.ndarray) -> float:
        """Detect noise level using local variance."""
        # Use median filter to estimate noise
        from scipy.ndimage import median_filter
        filtered = median_filter(image, size=3)
        noise = image - filtered
        return float(np.var(noise))

    def _detect_motion(self, image: np.ndarray) -> float:
        """Detect motion artifacts using edge consistency."""
        edges = np.sqrt(ndimage.sobel(image, 0)**2 + ndimage.sobel(image, 1)**2)
        # Motion causes inconsistent edges
        edge_consistency = np.std(edges[edges > 0.1])
        return float(edge_consistency)

    def _detect_bias_field(self, image: np.ndarray) -> float:
        """Detect intensity inhomogeneity."""
        # Low-pass filter to get bias field
        from scipy.ndimage import gaussian_filter
        low_freq = gaussian_filter(image, sigma=30)
        # Range of low-frequency component
        return float(low_freq.max() - low_freq.min())

    def _detect_zipper(self, image: np.ndarray) -> float:
        """Detect zipper artifacts using column/row analysis."""
        # Check for anomalous columns
        col_means = np.mean(image, axis=0)
        col_std = np.std(col_means)
        col_median = np.median(col_means)
        outliers = np.abs(col_means - col_median) > 3 * col_std
        return float(outliers.sum() / len(col_means))
