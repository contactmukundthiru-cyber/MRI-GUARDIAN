"""
Physics Violation Detector

Detects where AI reconstructions violate MRI physics principles.
This is a "physics police officer" for AI reconstruction.

Violations detected:
1. K-space inconsistency (data not matching measurements)
2. Phase coherence violations (non-physical phase patterns)
3. Gradient/intensity violations (impossible tissue properties)
4. Energy conservation violations
5. Conjugate symmetry violations (for real-valued images)
6. Ringing artifacts from impossible frequency spikes

Novel contribution: First systematic framework for detecting
physics violations in AI-reconstructed MRI images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class PhysicsViolation:
    """Container for a detected physics violation."""
    violation_type: str
    severity: float  # 0-1 scale
    location: Optional[torch.Tensor]  # Spatial map of violation
    description: str
    details: Dict


@dataclass
class PhysicsViolationReport:
    """Complete report of all detected physics violations."""
    violations: List[PhysicsViolation]
    total_severity: float
    is_physically_plausible: bool
    violation_map: torch.Tensor  # Combined spatial violation map
    summary: str


class KSpaceConsistencyChecker:
    """
    Check consistency between reconstructed image and measured k-space.

    Key principle: F(reconstructed_image) should match measured k-space
    at all acquired locations. Any significant deviation indicates
    either reconstruction error or physics violation.
    """

    def __init__(
        self,
        tolerance_factor: float = 3.0,  # Multiplier on noise std
        use_relative_error: bool = True
    ):
        self.tolerance_factor = tolerance_factor
        self.use_relative_error = use_relative_error

    def check(
        self,
        reconstructed: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
        noise_std: Optional[float] = None
    ) -> PhysicsViolation:
        """
        Check k-space data consistency.

        Args:
            reconstructed: Reconstructed image [B, 1, H, W] or complex [B, 1, H, W, 2]
            measured_kspace: Measured k-space (complex)
            mask: Sampling mask
            noise_std: Estimated noise standard deviation

        Returns:
            PhysicsViolation with details
        """
        # Convert reconstructed image to k-space
        if reconstructed.shape[-1] == 2:
            # Complex format
            recon_complex = torch.complex(reconstructed[..., 0], reconstructed[..., 1])
        else:
            # Real image - assume zero phase
            recon_complex = reconstructed.squeeze(-3) + 0j

        # FFT to k-space
        recon_kspace = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(recon_complex, dim=(-2, -1)),
                dim=(-2, -1),
                norm='ortho'
            ),
            dim=(-2, -1)
        )

        # Get measured k-space in complex form
        if measured_kspace.shape[-1] == 2:
            meas_complex = torch.complex(measured_kspace[..., 0], measured_kspace[..., 1])
        else:
            meas_complex = measured_kspace.squeeze(-3)

        # Ensure mask has correct shape
        while mask.dim() < recon_kspace.dim():
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(recon_kspace.real)

        # Compute error only at measured locations
        error = torch.abs(recon_kspace - meas_complex) * mask

        # Estimate noise level if not provided
        if noise_std is None:
            # Use median absolute deviation of high-frequency components
            corners = []
            H, W = error.shape[-2:]
            corner_size = min(H, W) // 8
            for corner in [
                error[..., :corner_size, :corner_size],
                error[..., :corner_size, -corner_size:],
                error[..., -corner_size:, :corner_size],
                error[..., -corner_size:, -corner_size:]
            ]:
                corners.append(corner.flatten())
            corner_values = torch.cat(corners)
            noise_std = 1.4826 * torch.median(corner_values).item()  # MAD estimator

        noise_std = max(noise_std, 1e-8)

        # Compute error relative to signal or noise
        if self.use_relative_error:
            signal_magnitude = torch.abs(meas_complex) * mask
            relative_error = error / (signal_magnitude + noise_std)
            threshold = self.tolerance_factor
            violation_mask = (relative_error > threshold).float()
            max_violation = relative_error.max().item()
        else:
            threshold = self.tolerance_factor * noise_std
            violation_mask = (error > threshold).float()
            max_violation = (error / noise_std).max().item()

        # Severity based on fraction of violated points and magnitude
        num_measured = mask.sum().item()
        num_violated = violation_mask.sum().item()
        violation_fraction = num_violated / (num_measured + 1e-8)

        severity = min(1.0, violation_fraction * 10 + max_violation / 10)

        # Create spatial violation map (in image domain)
        # High-error k-space regions affect specific spatial frequencies
        violation_kspace = error * violation_mask
        violation_map = torch.abs(
            torch.fft.ifft2(
                torch.fft.ifftshift(violation_kspace, dim=(-2, -1)),
                dim=(-2, -1),
                norm='ortho'
            )
        )

        description = (
            f"K-space data consistency violation: {violation_fraction*100:.1f}% of "
            f"measured points exceed tolerance (max error: {max_violation:.2f}σ)"
        )

        return PhysicsViolation(
            violation_type='kspace_consistency',
            severity=severity,
            location=violation_map.float(),
            description=description,
            details={
                'violation_fraction': violation_fraction,
                'max_violation': max_violation,
                'noise_std': noise_std,
                'threshold': threshold
            }
        )


class PhaseCoherenceChecker:
    """
    Check phase coherence in reconstructed images.

    MRI phase should be:
    1. Smooth within tissues (no random jumps)
    2. Consistent with local field inhomogeneity patterns
    3. Properly wrapped (gradual transitions)

    Phase violations often indicate hallucinated structures.
    """

    def __init__(
        self,
        gradient_threshold: float = 0.5,  # Max phase gradient (radians/pixel)
        jump_threshold: float = 2.0  # Phase jump threshold (radians)
    ):
        self.gradient_threshold = gradient_threshold
        self.jump_threshold = jump_threshold

    def check(
        self,
        complex_image: torch.Tensor,
        tissue_mask: Optional[torch.Tensor] = None
    ) -> PhysicsViolation:
        """
        Check phase coherence.

        Args:
            complex_image: Complex MRI image
            tissue_mask: Optional mask of tissue regions

        Returns:
            PhysicsViolation with details
        """
        # Extract phase
        if complex_image.shape[-1] == 2:
            phase = torch.atan2(complex_image[..., 1], complex_image[..., 0])
            magnitude = torch.sqrt(complex_image[..., 0]**2 + complex_image[..., 1]**2)
        elif complex_image.is_complex():
            phase = torch.angle(complex_image)
            magnitude = torch.abs(complex_image)
        else:
            # Real image - no phase to check
            return PhysicsViolation(
                violation_type='phase_coherence',
                severity=0.0,
                location=None,
                description='No phase information available (real-valued image)',
                details={}
            )

        # Ensure 2D
        while phase.dim() > 2:
            phase = phase.squeeze(0)
            magnitude = magnitude.squeeze(0)

        # Create tissue mask if not provided
        if tissue_mask is None:
            threshold = magnitude.mean() + 0.3 * magnitude.std()
            tissue_mask = (magnitude > threshold).float()

        # Phase gradients
        grad_x = phase[:, 1:] - phase[:, :-1]
        grad_y = phase[1:, :] - phase[:-1, :]

        # Handle phase wrapping (gradients > π should wrap)
        grad_x = torch.where(grad_x > np.pi, grad_x - 2*np.pi, grad_x)
        grad_x = torch.where(grad_x < -np.pi, grad_x + 2*np.pi, grad_x)
        grad_y = torch.where(grad_y > np.pi, grad_y - 2*np.pi, grad_y)
        grad_y = torch.where(grad_y < -np.pi, grad_y + 2*np.pi, grad_y)

        # Pad to original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

        # Apply tissue mask
        grad_mag_masked = grad_mag * tissue_mask

        # Detect violations
        gradient_violations = (grad_mag_masked > self.gradient_threshold).float()

        # Phase jumps (discontinuities)
        jump_x = torch.abs(grad_x) > self.jump_threshold
        jump_y = torch.abs(grad_y) > self.jump_threshold
        phase_jumps = ((jump_x | jump_y).float() * tissue_mask)

        # Combined violation map
        violation_map = torch.maximum(gradient_violations, phase_jumps)

        # Severity
        tissue_area = tissue_mask.sum().item()
        violation_area = violation_map.sum().item()
        violation_fraction = violation_area / (tissue_area + 1e-8)

        max_gradient = grad_mag_masked.max().item()
        severity = min(1.0, violation_fraction * 5 + max_gradient / 3)

        description = (
            f"Phase coherence violation: {violation_fraction*100:.1f}% of tissue "
            f"shows abnormal phase (max gradient: {max_gradient:.2f} rad/pixel)"
        )

        return PhysicsViolation(
            violation_type='phase_coherence',
            severity=severity,
            location=violation_map,
            description=description,
            details={
                'violation_fraction': violation_fraction,
                'max_gradient': max_gradient,
                'mean_gradient': grad_mag_masked.mean().item()
            }
        )


class GradientPhysicsChecker:
    """
    Check if image gradients are physically plausible.

    Real MRI images have gradient patterns constrained by:
    1. Tissue boundaries (sharp transitions only at edges)
    2. Maximum intensity changes (based on tissue properties)
    3. Spatial frequency content (limited by acquisition)
    """

    def __init__(
        self,
        max_gradient: float = 0.3,  # Max normalized gradient in tissue
        edge_threshold: float = 0.1  # Edge detection threshold
    ):
        self.max_gradient = max_gradient
        self.edge_threshold = edge_threshold

    def check(
        self,
        image: torch.Tensor,
        tissue_mask: Optional[torch.Tensor] = None
    ) -> PhysicsViolation:
        """
        Check gradient physics.

        Args:
            image: MRI image [B, 1, H, W] or [H, W]
            tissue_mask: Optional tissue mask

        Returns:
            PhysicsViolation with details
        """
        # Ensure 2D
        while image.dim() > 2:
            image = image.squeeze(0)

        # Normalize image
        img_min = image.min()
        img_max = image.max()
        img_norm = (image - img_min) / (img_max - img_min + 1e-8)

        # Compute gradients
        grad_x = img_norm[:, 1:] - img_norm[:, :-1]
        grad_y = img_norm[1:, :] - img_norm[:-1, :]

        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

        # Detect edges (expected high gradients)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0) / 8
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0) / 8

        edge_x = F.conv2d(img_norm.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
        edge_y = F.conv2d(img_norm.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2).squeeze()

        # Edges mask - high gradients are expected here
        edge_mask = (edge_mag > self.edge_threshold).float()

        # Tissue mask
        if tissue_mask is None:
            tissue_mask = (img_norm > 0.1).float()
        while tissue_mask.dim() > 2:
            tissue_mask = tissue_mask.squeeze(0)

        # Violation: high gradient in non-edge tissue regions
        non_edge_tissue = tissue_mask * (1 - edge_mask)
        violation_map = (grad_mag > self.max_gradient).float() * non_edge_tissue

        # Second-order gradient (should be smooth)
        laplacian = F.conv2d(
            img_norm.unsqueeze(0).unsqueeze(0),
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                         dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()

        # High Laplacian in smooth regions indicates ringing/artifacts
        laplacian_violation = (torch.abs(laplacian) > 0.2).float() * non_edge_tissue

        # Combined violation
        combined_violation = torch.maximum(violation_map, laplacian_violation)

        # Severity
        non_edge_area = non_edge_tissue.sum().item()
        violation_area = combined_violation.sum().item()
        violation_fraction = violation_area / (non_edge_area + 1e-8)

        severity = min(1.0, violation_fraction * 10)

        description = (
            f"Gradient physics violation: {violation_fraction*100:.1f}% of "
            f"non-edge tissue shows impossible gradients"
        )

        return PhysicsViolation(
            violation_type='gradient_physics',
            severity=severity,
            location=combined_violation,
            description=description,
            details={
                'violation_fraction': violation_fraction,
                'max_non_edge_gradient': (grad_mag * non_edge_tissue).max().item()
            }
        )


class EnergyConservationChecker:
    """
    Check energy conservation (Parseval's theorem).

    For properly normalized FFT:
    sum(|image|^2) = sum(|kspace|^2)

    Violation indicates numerical errors or non-physical modifications.
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def check(
        self,
        image: torch.Tensor,
        kspace: torch.Tensor
    ) -> PhysicsViolation:
        """
        Check energy conservation.

        Args:
            image: Image domain representation
            kspace: K-space representation

        Returns:
            PhysicsViolation with details
        """
        # Compute energies
        if image.shape[-1] == 2:
            image_energy = (image[..., 0]**2 + image[..., 1]**2).sum().item()
        elif image.is_complex():
            image_energy = (torch.abs(image)**2).sum().item()
        else:
            image_energy = (image**2).sum().item()

        if kspace.shape[-1] == 2:
            kspace_energy = (kspace[..., 0]**2 + kspace[..., 1]**2).sum().item()
        elif kspace.is_complex():
            kspace_energy = (torch.abs(kspace)**2).sum().item()
        else:
            kspace_energy = (kspace**2).sum().item()

        # Relative difference
        mean_energy = (image_energy + kspace_energy) / 2
        if mean_energy < 1e-8:
            relative_diff = 0
        else:
            relative_diff = abs(image_energy - kspace_energy) / mean_energy

        severity = min(1.0, relative_diff / self.tolerance)

        if relative_diff > self.tolerance:
            description = (
                f"Energy conservation violated: image energy = {image_energy:.2e}, "
                f"k-space energy = {kspace_energy:.2e} (diff: {relative_diff*100:.2f}%)"
            )
        else:
            description = f"Energy conservation satisfied (diff: {relative_diff*100:.4f}%)"

        return PhysicsViolation(
            violation_type='energy_conservation',
            severity=severity,
            location=None,  # Global violation, no spatial map
            description=description,
            details={
                'image_energy': image_energy,
                'kspace_energy': kspace_energy,
                'relative_difference': relative_diff
            }
        )


class ConjugateSymmetryChecker:
    """
    Check conjugate symmetry for real-valued images.

    For real images: F(k) = F*(-k)
    This must hold for physically valid reconstructions.
    """

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance

    def check(
        self,
        kspace: torch.Tensor,
        is_real_image: bool = True
    ) -> PhysicsViolation:
        """
        Check conjugate symmetry.

        Args:
            kspace: K-space data
            is_real_image: Whether the image should be real-valued

        Returns:
            PhysicsViolation with details
        """
        if not is_real_image:
            return PhysicsViolation(
                violation_type='conjugate_symmetry',
                severity=0.0,
                location=None,
                description='Conjugate symmetry check not applicable for complex images',
                details={}
            )

        # Convert to complex
        if kspace.shape[-1] == 2:
            ks_complex = torch.complex(kspace[..., 0], kspace[..., 1])
        elif kspace.is_complex():
            ks_complex = kspace
        else:
            # Already real k-space?
            ks_complex = kspace + 0j

        # Ensure 2D
        while ks_complex.dim() > 2:
            ks_complex = ks_complex.squeeze(0)

        # Flip and conjugate
        ks_conj_flip = torch.conj(torch.flip(ks_complex, dims=[0, 1]))

        # Roll by one to align properly (due to FFT conventions)
        ks_conj_flip = torch.roll(ks_conj_flip, shifts=(1, 1), dims=(0, 1))

        # Compute asymmetry
        asymmetry = torch.abs(ks_complex - ks_conj_flip)
        magnitude = torch.abs(ks_complex)

        # Relative asymmetry
        relative_asymmetry = asymmetry / (magnitude + 1e-8)

        # Mean asymmetry
        mean_asymmetry = relative_asymmetry.mean().item()
        max_asymmetry = relative_asymmetry.max().item()

        severity = min(1.0, mean_asymmetry / self.tolerance)

        # Create violation map (transform asymmetry to image domain)
        asymmetry_image = torch.abs(torch.fft.ifft2(
            torch.fft.ifftshift(asymmetry + 0j, dim=(-2, -1)),
            dim=(-2, -1),
            norm='ortho'
        ))

        description = (
            f"Conjugate symmetry: mean asymmetry = {mean_asymmetry*100:.2f}%, "
            f"max = {max_asymmetry*100:.2f}%"
        )

        return PhysicsViolation(
            violation_type='conjugate_symmetry',
            severity=severity,
            location=asymmetry_image.float(),
            description=description,
            details={
                'mean_asymmetry': mean_asymmetry,
                'max_asymmetry': max_asymmetry
            }
        )


class FrequencySpikesChecker:
    """
    Detect impossible high-frequency spikes in k-space.

    AI models sometimes create isolated spikes in k-space that
    correspond to non-physical periodic patterns in the image.
    """

    def __init__(
        self,
        spike_threshold: float = 5.0,  # Times local neighborhood
        neighborhood_size: int = 5
    ):
        self.spike_threshold = spike_threshold
        self.neighborhood_size = neighborhood_size

    def check(self, kspace: torch.Tensor) -> PhysicsViolation:
        """
        Detect k-space spikes.

        Args:
            kspace: K-space data

        Returns:
            PhysicsViolation with details
        """
        # Get magnitude
        if kspace.shape[-1] == 2:
            magnitude = torch.sqrt(kspace[..., 0]**2 + kspace[..., 1]**2)
        elif kspace.is_complex():
            magnitude = torch.abs(kspace)
        else:
            magnitude = torch.abs(kspace)

        # Ensure 2D
        while magnitude.dim() > 2:
            magnitude = magnitude.squeeze(0)

        # Local neighborhood median
        ks = self.neighborhood_size
        padded = F.pad(magnitude.unsqueeze(0).unsqueeze(0),
                       (ks//2, ks//2, ks//2, ks//2), mode='reflect')

        # Unfold to get neighborhoods
        unfolded = padded.unfold(2, ks, 1).unfold(3, ks, 1)
        unfolded = unfolded.reshape(1, 1, magnitude.shape[0], magnitude.shape[1], -1)

        # Compute median of neighborhood
        local_median = unfolded.median(dim=-1)[0].squeeze()

        # Spike ratio
        spike_ratio = magnitude / (local_median + 1e-8)

        # Exclude central DC region (naturally high)
        H, W = magnitude.shape
        center_h, center_w = H // 2, W // 2
        dc_mask = torch.zeros_like(magnitude)
        dc_region = min(H, W) // 8
        dc_mask[center_h-dc_region:center_h+dc_region,
                center_w-dc_region:center_w+dc_region] = 1

        # Detect spikes outside DC
        non_dc_spike_ratio = spike_ratio * (1 - dc_mask)
        spikes = (non_dc_spike_ratio > self.spike_threshold).float()

        num_spikes = spikes.sum().item()
        max_ratio = non_dc_spike_ratio.max().item()

        # Severity based on number and magnitude of spikes
        severity = min(1.0, num_spikes / 10 + max_ratio / 20)

        # Transform spike locations to image domain to show ringing
        spike_kspace = magnitude * spikes
        if spike_kspace.sum() > 0:
            ringing_pattern = torch.abs(torch.fft.ifft2(
                torch.fft.ifftshift(spike_kspace + 0j, dim=(-2, -1)),
                dim=(-2, -1),
                norm='ortho'
            ))
        else:
            ringing_pattern = torch.zeros_like(magnitude)

        description = (
            f"Frequency spikes detected: {int(num_spikes)} isolated spikes, "
            f"max ratio = {max_ratio:.1f}× local neighborhood"
        )

        return PhysicsViolation(
            violation_type='frequency_spikes',
            severity=severity,
            location=ringing_pattern.float(),
            description=description,
            details={
                'num_spikes': int(num_spikes),
                'max_ratio': max_ratio,
                'spike_locations': torch.where(spikes > 0)
            }
        )


class PhysicsViolationDetector:
    """
    Comprehensive physics violation detector.

    Combines all physics checks into a unified framework.
    """

    def __init__(
        self,
        check_kspace_consistency: bool = True,
        check_phase_coherence: bool = True,
        check_gradient_physics: bool = True,
        check_energy_conservation: bool = True,
        check_conjugate_symmetry: bool = True,
        check_frequency_spikes: bool = True
    ):
        self.checkers = {}

        if check_kspace_consistency:
            self.checkers['kspace'] = KSpaceConsistencyChecker()

        if check_phase_coherence:
            self.checkers['phase'] = PhaseCoherenceChecker()

        if check_gradient_physics:
            self.checkers['gradient'] = GradientPhysicsChecker()

        if check_energy_conservation:
            self.checkers['energy'] = EnergyConservationChecker()

        if check_conjugate_symmetry:
            self.checkers['symmetry'] = ConjugateSymmetryChecker()

        if check_frequency_spikes:
            self.checkers['spikes'] = FrequencySpikesChecker()

    def detect(
        self,
        reconstructed_image: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
        complex_image: Optional[torch.Tensor] = None
    ) -> PhysicsViolationReport:
        """
        Perform comprehensive physics violation detection.

        Args:
            reconstructed_image: Reconstructed image
            measured_kspace: Original measured k-space
            mask: Sampling mask
            complex_image: Complex reconstruction (if available)

        Returns:
            PhysicsViolationReport with all violations
        """
        violations = []

        # Ensure proper shapes
        while reconstructed_image.dim() < 4:
            reconstructed_image = reconstructed_image.unsqueeze(0)

        # K-space consistency
        if 'kspace' in self.checkers:
            v = self.checkers['kspace'].check(
                reconstructed_image, measured_kspace, mask
            )
            violations.append(v)

        # Phase coherence
        if 'phase' in self.checkers and complex_image is not None:
            v = self.checkers['phase'].check(complex_image)
            violations.append(v)

        # Gradient physics
        if 'gradient' in self.checkers:
            v = self.checkers['gradient'].check(reconstructed_image)
            violations.append(v)

        # Reconstruct k-space for energy and symmetry checks
        recon_img = reconstructed_image.squeeze()
        if recon_img.dim() == 2:
            recon_kspace = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(recon_img), norm='ortho')
            )
        else:
            recon_kspace = measured_kspace  # Use provided if reconstruction fails

        # Energy conservation
        if 'energy' in self.checkers:
            v = self.checkers['energy'].check(reconstructed_image, recon_kspace)
            violations.append(v)

        # Conjugate symmetry
        if 'symmetry' in self.checkers:
            v = self.checkers['symmetry'].check(recon_kspace, is_real_image=True)
            violations.append(v)

        # Frequency spikes
        if 'spikes' in self.checkers:
            v = self.checkers['spikes'].check(recon_kspace)
            violations.append(v)

        # Combine violation maps
        spatial_violations = [v.location for v in violations if v.location is not None]
        if spatial_violations:
            # Normalize and combine
            normalized = []
            for vm in spatial_violations:
                while vm.dim() < 2:
                    vm = vm.unsqueeze(0)
                vm_norm = vm / (vm.max() + 1e-8)
                normalized.append(vm_norm)

            # Resize to common size
            target_size = reconstructed_image.shape[-2:]
            resized = []
            for vm in normalized:
                if vm.shape[-2:] != target_size:
                    vm = F.interpolate(
                        vm.unsqueeze(0).unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                resized.append(vm)

            combined_map = torch.stack(resized, dim=0).max(dim=0)[0]
        else:
            combined_map = torch.zeros(reconstructed_image.shape[-2:])

        # Total severity
        severities = [v.severity for v in violations]
        total_severity = max(severities) if severities else 0.0

        # Is physically plausible?
        is_plausible = total_severity < 0.5

        # Summary
        violated_types = [v.violation_type for v in violations if v.severity > 0.3]
        if violated_types:
            summary = f"Physics violations detected: {', '.join(violated_types)}"
        else:
            summary = "Reconstruction appears physically plausible"

        return PhysicsViolationReport(
            violations=violations,
            total_severity=total_severity,
            is_physically_plausible=is_plausible,
            violation_map=combined_map,
            summary=summary
        )
