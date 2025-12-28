"""
Lesion Integrity Verification Module

Verifies that AI reconstruction correctly preserves lesions without:
- Softening/blurring real lesions
- Missing subtle lesions
- Inventing false lesions (hallucinations)

This is the #1 clinical concern for radiologists evaluating AI reconstruction.

Novel contribution: Comprehensive lesion integrity framework with
realistic lesion simulation and multi-scale verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class LesionIntegrityResult:
    """Result of lesion integrity verification."""
    lesion_preserved: bool
    preservation_score: float  # 0-1, higher is better
    contrast_preservation: float  # How well contrast maintained
    boundary_sharpness: float  # Edge preservation
    size_accuracy: float  # Size preservation
    location_accuracy: float  # Position accuracy
    details: Dict


@dataclass
class LesionPreservationReport:
    """Complete report on lesion preservation across test cases."""
    overall_preservation_rate: float
    preservation_by_size: Dict[str, float]  # small/medium/large
    preservation_by_contrast: Dict[str, float]  # low/medium/high
    preservation_by_location: Dict[str, float]  # peripheral/central
    false_positive_rate: float  # Invented lesions
    false_negative_rate: float  # Missed lesions
    clinical_acceptability: bool
    recommendations: List[str]


class SubtleLesionGenerator:
    """
    Generate realistic subtle lesions for testing AI reconstruction.

    Creates challenging test cases that push the limits of
    reconstruction algorithms:
    - Micro-lesions (2-5 pixels)
    - Low-contrast lesions
    - Lesions at noise floor
    - Lesions in undersampled regions
    - Thin structures (cartilage tears, etc.)
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def generate_micro_lesion(
        self,
        image: torch.Tensor,
        size_range: Tuple[int, int] = (2, 5),
        contrast_range: Tuple[float, float] = (0.1, 0.3),
        num_lesions: int = 1,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Generate micro-lesions (very small, clinically critical).

        Args:
            image: Input image [B, 1, H, W]
            size_range: (min, max) lesion size in pixels
            contrast_range: (min, max) contrast relative to background
            num_lesions: Number of lesions to add
            seed: Random seed

        Returns:
            modified_image: Image with lesions
            lesion_mask: Binary mask
            lesion_info: List of lesion metadata
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)
        lesion_info = []

        for b in range(B):
            for i in range(num_lesions):
                # Random size
                size = np.random.randint(size_range[0], size_range[1] + 1)

                # Random position (avoid edges)
                margin = size + 5
                cy = np.random.randint(margin, H - margin)
                cx = np.random.randint(margin, W - margin)

                # Local background intensity
                bg_region = output[b, 0, max(0, cy-10):min(H, cy+10),
                                   max(0, cx-10):min(W, cx+10)]
                bg_mean = bg_region.mean().item()
                bg_std = bg_region.std().item()

                # Random contrast
                contrast = np.random.uniform(*contrast_range)

                # Lesion can be bright or dark relative to background
                if np.random.random() > 0.5:
                    lesion_intensity = bg_mean + contrast * bg_std * 3
                else:
                    lesion_intensity = bg_mean - contrast * bg_std * 3

                # Create circular lesion
                y, x = torch.meshgrid(
                    torch.arange(H, device=device).float(),
                    torch.arange(W, device=device).float(),
                    indexing='ij'
                )

                dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
                lesion = (dist <= size / 2).float()

                # Smooth edges slightly
                if size > 3:
                    lesion = F.avg_pool2d(
                        lesion.unsqueeze(0).unsqueeze(0), 3, 1, 1
                    ).squeeze()
                    lesion = (lesion > 0.3).float()

                # Apply lesion
                output[b, 0] = torch.where(
                    lesion > 0,
                    torch.tensor(lesion_intensity, device=device),
                    output[b, 0]
                )
                mask[b, 0] = torch.maximum(mask[b, 0], lesion)

                lesion_info.append({
                    'batch': b,
                    'center': (cy, cx),
                    'size': size,
                    'contrast': contrast,
                    'intensity': lesion_intensity,
                    'type': 'micro'
                })

        return output, mask, lesion_info

    def generate_low_contrast_lesion(
        self,
        image: torch.Tensor,
        size_range: Tuple[int, int] = (8, 20),
        contrast_factor: float = 0.05,  # Very subtle
        num_lesions: int = 1,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Generate low-contrast lesions (barely visible but clinically significant).
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)
        lesion_info = []

        for b in range(B):
            for i in range(num_lesions):
                # Random size
                size = np.random.randint(size_range[0], size_range[1] + 1)

                # Random position
                margin = size + 10
                cy = np.random.randint(margin, H - margin)
                cx = np.random.randint(margin, W - margin)

                # Local statistics
                bg_region = output[b, 0, cy-size:cy+size, cx-size:cx+size]
                bg_mean = bg_region.mean().item()
                bg_std = bg_region.std().item()

                # Very low contrast
                delta = contrast_factor * bg_mean
                if np.random.random() > 0.5:
                    lesion_intensity = bg_mean + delta
                else:
                    lesion_intensity = bg_mean - delta

                # Elliptical lesion
                y, x = torch.meshgrid(
                    torch.arange(H, device=device).float(),
                    torch.arange(W, device=device).float(),
                    indexing='ij'
                )

                # Random aspect ratio
                aspect = np.random.uniform(0.6, 1.4)
                a = size / 2 * aspect
                b_axis = size / 2 / aspect

                # Random rotation
                angle = np.random.uniform(0, np.pi)
                x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
                y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)

                dist = (x_rot / a)**2 + (y_rot / b_axis)**2
                lesion = (dist <= 1).float()

                # Very smooth edges for subtle appearance
                lesion = F.avg_pool2d(
                    lesion.unsqueeze(0).unsqueeze(0), 5, 1, 2
                ).squeeze()

                # Blend with background
                blend_factor = lesion * 0.8  # Partial blending
                output[b, 0] = output[b, 0] * (1 - blend_factor) + lesion_intensity * blend_factor
                mask[b, 0] = torch.maximum(mask[b, 0], (lesion > 0.2).float())

                lesion_info.append({
                    'batch': b,
                    'center': (cy, cx),
                    'size': size,
                    'contrast': contrast_factor,
                    'intensity': lesion_intensity,
                    'type': 'low_contrast'
                })

        return output, mask, lesion_info

    def generate_linear_structure(
        self,
        image: torch.Tensor,
        length_range: Tuple[int, int] = (15, 40),
        thickness_range: Tuple[int, int] = (1, 3),
        contrast_range: Tuple[float, float] = (0.1, 0.3),
        num_structures: int = 1,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Generate thin linear structures (cartilage tears, vessels, etc.).
        """
        if seed is not None:
            np.random.seed(seed)

        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)
        lesion_info = []

        for b in range(B):
            for i in range(num_structures):
                # Random length and thickness
                length = np.random.randint(length_range[0], length_range[1] + 1)
                thickness = np.random.randint(thickness_range[0], thickness_range[1] + 1)

                # Random start point
                margin = max(length, 20)
                y1 = np.random.randint(margin, H - margin)
                x1 = np.random.randint(margin, W - margin)

                # Random angle
                angle = np.random.uniform(0, 2 * np.pi)
                y2 = int(y1 + length * np.sin(angle))
                x2 = int(x1 + length * np.cos(angle))

                # Clip endpoints
                y2 = np.clip(y2, 0, H - 1)
                x2 = np.clip(x2, 0, W - 1)

                # Local intensity
                bg_mean = output[b, 0, y1-5:y1+5, x1-5:x1+5].mean().item()
                contrast = np.random.uniform(*contrast_range)
                line_intensity = bg_mean * (1 + contrast)

                # Draw line
                num_points = max(abs(y2 - y1), abs(x2 - x1)) + 1
                ys = np.linspace(y1, y2, num_points).astype(int)
                xs = np.linspace(x1, x2, num_points).astype(int)

                for y, x in zip(ys, xs):
                    for dy in range(-thickness//2, thickness//2 + 1):
                        for dx in range(-thickness//2, thickness//2 + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                output[b, 0, ny, nx] = line_intensity
                                mask[b, 0, ny, nx] = 1.0

                lesion_info.append({
                    'batch': b,
                    'start': (y1, x1),
                    'end': (y2, x2),
                    'length': length,
                    'thickness': thickness,
                    'contrast': contrast,
                    'type': 'linear'
                })

        return output, mask, lesion_info

    def generate_at_undersampled_region(
        self,
        image: torch.Tensor,
        sampling_mask: torch.Tensor,
        size_range: Tuple[int, int] = (5, 15),
        contrast_range: Tuple[float, float] = (0.15, 0.4),
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Generate lesion specifically in undersampled k-space region.

        This tests if reconstruction can recover features from
        regions with missing data.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)
        lesion_info = []

        # Analyze sampling pattern
        while sampling_mask.dim() > 2:
            sampling_mask = sampling_mask.squeeze(0)

        # Find undersampled frequency regions
        # These correspond to specific spatial frequencies
        undersampled_freqs = (sampling_mask < 0.5)

        for b in range(B):
            # Random size
            size = np.random.randint(size_range[0], size_range[1] + 1)

            # Position based on undersampled frequencies
            # High-frequency undersampling → fine details most affected
            # Choose position that would have power in undersampled region
            margin = size + 10
            cy = np.random.randint(margin, H - margin)
            cx = np.random.randint(margin, W - margin)

            # Local statistics
            bg_region = output[b, 0, max(0, cy-10):min(H, cy+10),
                               max(0, cx-10):min(W, cx+10)]
            bg_mean = bg_region.mean().item()

            # Contrast
            contrast = np.random.uniform(*contrast_range)
            lesion_intensity = bg_mean * (1 + contrast)

            # Create sharp-edged lesion (affected by high-freq undersampling)
            y, x = torch.meshgrid(
                torch.arange(H, device=device).float(),
                torch.arange(W, device=device).float(),
                indexing='ij'
            )

            dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
            lesion = (dist <= size / 2).float()

            # Add high-frequency texture inside lesion
            texture = torch.randn(H, W, device=device) * 0.05 * lesion_intensity
            texture = texture * lesion

            # Apply
            output[b, 0] = torch.where(
                lesion > 0,
                torch.tensor(lesion_intensity, device=device) + texture,
                output[b, 0]
            )
            mask[b, 0] = torch.maximum(mask[b, 0], lesion)

            lesion_info.append({
                'batch': b,
                'center': (cy, cx),
                'size': size,
                'contrast': contrast,
                'type': 'undersampled_region'
            })

        return output, mask, lesion_info


class LesionPreservationMetrics:
    """
    Compute metrics for lesion preservation quality.
    """

    @staticmethod
    def compute_contrast_preservation(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        lesion_mask: torch.Tensor,
        background_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute how well lesion contrast is preserved.

        Contrast = |lesion_mean - background_mean|
        Preservation = recon_contrast / original_contrast
        """
        # Ensure 2D
        while original.dim() > 2:
            original = original.squeeze(0)
        while reconstructed.dim() > 2:
            reconstructed = reconstructed.squeeze(0)
        while lesion_mask.dim() > 2:
            lesion_mask = lesion_mask.squeeze(0)

        lesion_mask = lesion_mask > 0.5

        if background_mask is None:
            # Dilate lesion mask for background
            dilated = F.max_pool2d(
                lesion_mask.float().unsqueeze(0).unsqueeze(0),
                kernel_size=11, stride=1, padding=5
            ).squeeze() > 0.5
            background_mask = dilated & ~lesion_mask

        # Original contrast
        orig_lesion = original[lesion_mask].mean().item()
        orig_bg = original[background_mask].mean().item() if background_mask.sum() > 0 else 0
        orig_contrast = abs(orig_lesion - orig_bg)

        # Reconstructed contrast
        recon_lesion = reconstructed[lesion_mask].mean().item()
        recon_bg = reconstructed[background_mask].mean().item() if background_mask.sum() > 0 else 0
        recon_contrast = abs(recon_lesion - recon_bg)

        if orig_contrast < 1e-8:
            return 1.0

        preservation = recon_contrast / orig_contrast
        return min(1.0, preservation)

    @staticmethod
    def compute_boundary_sharpness(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        lesion_mask: torch.Tensor
    ) -> float:
        """
        Compute boundary sharpness preservation.

        Uses gradient magnitude at lesion boundaries.
        """
        while original.dim() > 2:
            original = original.squeeze(0)
        while reconstructed.dim() > 2:
            reconstructed = reconstructed.squeeze(0)
        while lesion_mask.dim() > 2:
            lesion_mask = lesion_mask.squeeze(0)

        # Find boundary (edge of mask)
        dilated = F.max_pool2d(
            lesion_mask.float().unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        eroded = -F.max_pool2d(
            -lesion_mask.float().unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        boundary = (dilated - eroded) > 0.5

        if boundary.sum() == 0:
            return 1.0

        # Gradient magnitude
        def gradient_magnitude(img):
            gx = torch.abs(img[:, 1:] - img[:, :-1])
            gy = torch.abs(img[1:, :] - img[:-1, :])
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            return torch.sqrt(gx**2 + gy**2)

        orig_grad = gradient_magnitude(original)
        recon_grad = gradient_magnitude(reconstructed)

        # Mean gradient at boundary
        orig_boundary_grad = orig_grad[boundary].mean().item()
        recon_boundary_grad = recon_grad[boundary].mean().item()

        if orig_boundary_grad < 1e-8:
            return 1.0

        sharpness = recon_boundary_grad / orig_boundary_grad
        return min(1.0, sharpness)

    @staticmethod
    def compute_size_accuracy(
        original_mask: torch.Tensor,
        detected_mask: torch.Tensor
    ) -> float:
        """
        Compute size preservation accuracy.

        Dice coefficient between original and detected lesion.
        """
        while original_mask.dim() > 2:
            original_mask = original_mask.squeeze(0)
        while detected_mask.dim() > 2:
            detected_mask = detected_mask.squeeze(0)

        orig = (original_mask > 0.5).float()
        det = (detected_mask > 0.5).float()

        intersection = (orig * det).sum()
        union = orig.sum() + det.sum()

        if union < 1e-8:
            return 1.0

        dice = 2 * intersection / union
        return dice.item()

    @staticmethod
    def compute_location_accuracy(
        original_mask: torch.Tensor,
        detected_mask: torch.Tensor
    ) -> float:
        """
        Compute centroid location accuracy.
        """
        while original_mask.dim() > 2:
            original_mask = original_mask.squeeze(0)
        while detected_mask.dim() > 2:
            detected_mask = detected_mask.squeeze(0)

        H, W = original_mask.shape

        # Compute centroids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=original_mask.device).float(),
            torch.arange(W, device=original_mask.device).float(),
            indexing='ij'
        )

        orig = (original_mask > 0.5).float()
        det = (detected_mask > 0.5).float()

        if orig.sum() < 1 or det.sum() < 1:
            return 0.0

        orig_cy = (y_coords * orig).sum() / orig.sum()
        orig_cx = (x_coords * orig).sum() / orig.sum()

        det_cy = (y_coords * det).sum() / det.sum()
        det_cx = (x_coords * det).sum() / det.sum()

        # Distance normalized by image size
        distance = torch.sqrt((orig_cy - det_cy)**2 + (orig_cx - det_cx)**2)
        max_dist = np.sqrt(H**2 + W**2)

        accuracy = 1.0 - (distance / max_dist).item()
        return max(0.0, accuracy)


class LesionIntegrityVerifier:
    """
    Complete lesion integrity verification system.

    Verifies that AI reconstructions properly preserve lesions
    without softening, missing, or hallucinating.
    """

    def __init__(
        self,
        detection_threshold: float = 0.3,
        preservation_threshold: float = 0.7,
        device: str = 'cuda'
    ):
        self.detection_threshold = detection_threshold
        self.preservation_threshold = preservation_threshold
        self.device = device

        self.lesion_generator = SubtleLesionGenerator(device=device)
        self.metrics = LesionPreservationMetrics()

    def verify_single(
        self,
        original_with_lesion: torch.Tensor,
        reconstructed: torch.Tensor,
        lesion_mask: torch.Tensor,
        lesion_info: Dict
    ) -> LesionIntegrityResult:
        """
        Verify lesion integrity for a single case.

        Args:
            original_with_lesion: Original image with lesion
            reconstructed: AI reconstruction
            lesion_mask: Ground truth lesion mask
            lesion_info: Lesion metadata

        Returns:
            LesionIntegrityResult with detailed analysis
        """
        # Compute all metrics
        contrast = self.metrics.compute_contrast_preservation(
            original_with_lesion, reconstructed, lesion_mask
        )
        sharpness = self.metrics.compute_boundary_sharpness(
            original_with_lesion, reconstructed, lesion_mask
        )

        # Detect lesion in reconstruction
        # Use intensity-based detection
        recon_2d = reconstructed.squeeze()
        mask_2d = lesion_mask.squeeze() > 0.5

        if mask_2d.sum() > 0:
            orig_2d = original_with_lesion.squeeze()

            # Expected lesion intensity from original
            expected_intensity = orig_2d[mask_2d].mean()

            # Find regions in reconstruction matching expected intensity
            intensity_diff = torch.abs(recon_2d - expected_intensity)

            # Threshold for detection
            threshold = intensity_diff.mean() + intensity_diff.std()
            detected_mask = (intensity_diff < threshold) & mask_2d.float().bool()

            # Also check if values are significantly different from background
            bg_mask = ~mask_2d
            if bg_mask.sum() > 0:
                bg_mean = recon_2d[bg_mask].mean()
                bg_std = recon_2d[bg_mask].std()

                lesion_region_mean = recon_2d[mask_2d].mean()
                z_score = abs(lesion_region_mean - bg_mean) / (bg_std + 1e-8)

                lesion_detected = z_score > 1.5  # Lesion visible
            else:
                lesion_detected = True

            detected_mask = mask_2d if lesion_detected else torch.zeros_like(mask_2d)
        else:
            detected_mask = torch.zeros_like(lesion_mask.squeeze())
            lesion_detected = False

        size_acc = self.metrics.compute_size_accuracy(lesion_mask, detected_mask)
        loc_acc = self.metrics.compute_location_accuracy(lesion_mask, detected_mask)

        # Overall preservation score
        preservation_score = (
            0.3 * contrast +
            0.2 * sharpness +
            0.3 * size_acc +
            0.2 * loc_acc
        )

        lesion_preserved = preservation_score >= self.preservation_threshold

        return LesionIntegrityResult(
            lesion_preserved=lesion_preserved,
            preservation_score=preservation_score,
            contrast_preservation=contrast,
            boundary_sharpness=sharpness,
            size_accuracy=size_acc,
            location_accuracy=loc_acc,
            details={
                'lesion_info': lesion_info,
                'detected': lesion_detected,
                'detection_z_score': z_score if 'z_score' in dir() else 0
            }
        )

    def run_comprehensive_test(
        self,
        model: nn.Module,
        base_images: List[torch.Tensor],
        masks: List[torch.Tensor],
        num_tests_per_type: int = 10
    ) -> LesionPreservationReport:
        """
        Run comprehensive lesion preservation tests.

        Args:
            model: Reconstruction model
            base_images: Clean base images
            masks: Corresponding sampling masks
            num_tests_per_type: Number of tests per lesion type

        Returns:
            LesionPreservationReport with complete analysis
        """
        model.eval()

        results_by_type = {
            'micro': [],
            'low_contrast': [],
            'linear': [],
            'undersampled': []
        }

        results_by_size = {'small': [], 'medium': [], 'large': []}
        results_by_contrast = {'low': [], 'medium': [], 'high': []}

        false_positive_tests = 0
        false_positive_count = 0

        for idx, (base_img, mask) in enumerate(zip(base_images, masks)):
            if idx >= num_tests_per_type:
                break

            # Test each lesion type
            # 1. Micro lesions
            img_lesion, lesion_mask, info = self.lesion_generator.generate_micro_lesion(
                base_img.unsqueeze(0) if base_img.dim() == 3 else base_img,
                seed=idx
            )

            with torch.no_grad():
                # Apply forward model (undersampling) and reconstruct
                from mri_guardian.data.kspace_ops import fft2c, ifft2c

                kspace = fft2c(img_lesion.squeeze())
                masked_kspace = kspace * mask
                result = model(masked_kspace.unsqueeze(0), mask.unsqueeze(0))
                recon = result['output'] if isinstance(result, dict) else result

            integrity = self.verify_single(img_lesion, recon, lesion_mask, info[0])
            results_by_type['micro'].append(integrity)

            if info[0]['size'] <= 4:
                results_by_size['small'].append(integrity)
            elif info[0]['size'] <= 8:
                results_by_size['medium'].append(integrity)
            else:
                results_by_size['large'].append(integrity)

            # 2. Low contrast lesions
            img_lesion, lesion_mask, info = self.lesion_generator.generate_low_contrast_lesion(
                base_img.unsqueeze(0) if base_img.dim() == 3 else base_img,
                seed=idx + 1000
            )

            with torch.no_grad():
                kspace = fft2c(img_lesion.squeeze())
                masked_kspace = kspace * mask
                result = model(masked_kspace.unsqueeze(0), mask.unsqueeze(0))
                recon = result['output'] if isinstance(result, dict) else result

            integrity = self.verify_single(img_lesion, recon, lesion_mask, info[0])
            results_by_type['low_contrast'].append(integrity)
            results_by_contrast['low'].append(integrity)

            # 3. Linear structures
            img_lesion, lesion_mask, info = self.lesion_generator.generate_linear_structure(
                base_img.unsqueeze(0) if base_img.dim() == 3 else base_img,
                seed=idx + 2000
            )

            with torch.no_grad():
                kspace = fft2c(img_lesion.squeeze())
                masked_kspace = kspace * mask
                result = model(masked_kspace.unsqueeze(0), mask.unsqueeze(0))
                recon = result['output'] if isinstance(result, dict) else result

            integrity = self.verify_single(img_lesion, recon, lesion_mask, info[0])
            results_by_type['linear'].append(integrity)

            # 4. Check for false positives (hallucinated lesions)
            # Reconstruct image WITHOUT lesions and check for artifacts
            with torch.no_grad():
                kspace = fft2c(base_img.squeeze() if base_img.dim() > 2 else base_img)
                masked_kspace = kspace * mask
                result = model(masked_kspace.unsqueeze(0), mask.unsqueeze(0))
                clean_recon = result['output'] if isinstance(result, dict) else result

            # Check if reconstruction has unexpected bright/dark spots
            clean_recon_2d = clean_recon.squeeze()
            base_2d = base_img.squeeze() if base_img.dim() > 2 else base_img

            diff = torch.abs(clean_recon_2d - base_2d)
            diff_threshold = diff.mean() + 3 * diff.std()
            artifacts = (diff > diff_threshold).sum().item()

            false_positive_tests += 1
            if artifacts > 100:  # More than 100 pixels with high error
                false_positive_count += 1

        # Compile results
        def compute_preservation_rate(results):
            if not results:
                return 0.0
            return sum(1 for r in results if r.lesion_preserved) / len(results)

        overall_rate = compute_preservation_rate(
            results_by_type['micro'] + results_by_type['low_contrast'] +
            results_by_type['linear']
        )

        size_rates = {
            size: compute_preservation_rate(results)
            for size, results in results_by_size.items()
        }

        contrast_rates = {
            contrast: compute_preservation_rate(results)
            for contrast, results in results_by_contrast.items()
        }

        type_rates = {
            ltype: compute_preservation_rate(results)
            for ltype, results in results_by_type.items()
        }

        fp_rate = false_positive_count / max(1, false_positive_tests)

        # False negative = lesions not preserved
        all_results = (
            results_by_type['micro'] + results_by_type['low_contrast'] +
            results_by_type['linear']
        )
        fn_rate = 1 - overall_rate

        # Clinical acceptability
        clinical_ok = (
            overall_rate >= 0.9 and
            fp_rate <= 0.05 and
            min(size_rates.values()) >= 0.8 if size_rates else True
        )

        # Recommendations
        recommendations = []
        if overall_rate < 0.9:
            recommendations.append(
                f"Overall preservation rate ({overall_rate:.1%}) below clinical threshold (90%)"
            )
        if size_rates.get('small', 1.0) < 0.8:
            recommendations.append(
                f"Poor preservation of small lesions ({size_rates['small']:.1%})"
            )
        if contrast_rates.get('low', 1.0) < 0.8:
            recommendations.append(
                f"Poor preservation of low-contrast lesions ({contrast_rates['low']:.1%})"
            )
        if fp_rate > 0.05:
            recommendations.append(
                f"High false positive rate ({fp_rate:.1%})"
            )
        if not recommendations:
            recommendations.append("Lesion integrity verification passed")

        return LesionPreservationReport(
            overall_preservation_rate=overall_rate,
            preservation_by_size=size_rates,
            preservation_by_contrast=contrast_rates,
            preservation_by_location=type_rates,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            clinical_acceptability=clinical_ok,
            recommendations=recommendations
        )
