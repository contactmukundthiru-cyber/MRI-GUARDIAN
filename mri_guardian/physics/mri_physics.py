"""
MRI Physics Module

This module implements the fundamental physics of MRI signal acquisition.
Understanding this is CRITICAL for the Guardian model.

MRI PHYSICS INTUITION:
======================

1. THE MRI SIGNAL EQUATION (Simplified 2D):
   s(k_x, k_y) = ∫∫ ρ(x,y) * exp(-2πi(k_x*x + k_y*y)) dx dy

   Where:
   - ρ(x,y) is the image we want (tissue magnetization density)
   - s(k_x, k_y) is the measured signal at position (k_x, k_y) in k-space
   - The integral is over the entire image

   This is exactly the 2D Fourier Transform!
   Image = FFT⁻¹(k-space)
   k-space = FFT(Image)

2. WHY K-SPACE?
   - MRI machines measure k-space DIRECTLY, not images
   - Each point in k-space is measured by applying magnetic gradients
   - Traversing k-space takes time → slow scans

3. UNDERSAMPLING PROBLEM:
   - Full k-space measurement is slow (10-60 minutes for some scans)
   - Skip some k-space points → "undersampling" → faster scan
   - But: missing k-space causes aliasing artifacts in image
   - Challenge: recover the original image from incomplete data

4. DATA CONSISTENCY:
   - Whatever reconstruction we compute, it MUST agree with measured data
   - Measured k-space points = "ground truth"
   - Only fill in the MISSING points with learning
   - DC(x) = keeps measured samples, only changes missing samples

5. PARALLEL IMAGING (MULTI-COIL):
   - Multiple receiver coils around the patient
   - Each coil has different "sensitivity" at each location
   - More information → better reconstruction
   - SENSE equation: image * sensitivity_map = coil_image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from ..data.kspace_ops import fft2c, ifft2c, complex_to_channels, channels_to_complex


class MRIPhysics(nn.Module):
    """
    MRI Physics Model

    Implements the forward and adjoint (backward) operations for MRI.

    Forward: Image → K-space (with mask)
    Adjoint: K-space → Image (inverse of forward, accounting for sampling)

    For single-coil:
        Forward: y = M * F * x
        Adjoint: x = F^H * M^H * y

    Where:
        - x: Image
        - y: Measured k-space
        - F: 2D FFT
        - M: Sampling mask
        - ^H: Hermitian (conjugate transpose)
    """

    def __init__(
        self,
        use_complex: bool = True,
        norm: str = "ortho"
    ):
        """
        Args:
            use_complex: If True, use complex tensors. If False, use 2-channel real.
            norm: FFT normalization mode
        """
        super().__init__()
        self.use_complex = use_complex
        self.norm = norm

    def forward_op(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward MRI operator: Image → Measured K-space

        Args:
            image: Image tensor
                - Complex: (B, H, W)
                - 2-channel: (B, 2, H, W)
            mask: Sampling mask (B, 1, H, W) or (1, 1, H, W)
            sens_maps: Optional sensitivity maps (B, num_coils, H, W) complex

        Returns:
            Masked k-space with same format as input
        """
        if self.use_complex:
            # Complex tensor path
            if sens_maps is not None:
                # Multi-coil: x * S_i → FFT → mask
                coil_images = image.unsqueeze(1) * sens_maps  # (B, C, H, W)
                kspace = fft2c(coil_images, norm=self.norm)
            else:
                # Single-coil
                kspace = fft2c(image, norm=self.norm)

            # Apply mask
            if mask.ndim == 4:
                mask = mask.squeeze(1)  # (B, H, W)
            masked_kspace = kspace * mask.unsqueeze(1) if sens_maps is not None else kspace * mask

            return masked_kspace

        else:
            # 2-channel real tensor path
            # Convert to complex
            image_complex = channels_to_complex(image)

            if sens_maps is not None:
                coil_images = image_complex.unsqueeze(1) * sens_maps
                kspace = fft2c(coil_images, norm=self.norm)
            else:
                kspace = fft2c(image_complex, norm=self.norm)

            # Apply mask
            if mask.ndim == 4:
                mask_2d = mask.squeeze(1)
            else:
                mask_2d = mask

            if sens_maps is not None:
                masked_kspace = kspace * mask_2d.unsqueeze(1)
            else:
                masked_kspace = kspace * mask_2d

            # Convert back to 2-channel
            return complex_to_channels(masked_kspace)

    def adjoint_op(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Adjoint MRI operator: K-space → Image

        This is the "pseudo-inverse" of the forward operator.
        For undersampled data, this gives aliased images.

        Args:
            kspace: K-space tensor (same format as forward output)
            mask: Sampling mask
            sens_maps: Optional sensitivity maps

        Returns:
            Image tensor (same format as forward input)
        """
        if self.use_complex:
            # Apply mask (adjoint of mask is itself for real masks)
            if mask.ndim == 4:
                mask = mask.squeeze(1)

            if sens_maps is not None:
                masked_kspace = kspace * mask.unsqueeze(1)
                # IFFT then combine coils
                coil_images = ifft2c(masked_kspace, norm=self.norm)
                # Combine: sum(conj(S_i) * x_i)
                image = torch.sum(torch.conj(sens_maps) * coil_images, dim=1)
            else:
                masked_kspace = kspace * mask
                image = ifft2c(masked_kspace, norm=self.norm)

            return image

        else:
            # 2-channel path
            kspace_complex = channels_to_complex(kspace)

            if mask.ndim == 4:
                mask_2d = mask.squeeze(1)
            else:
                mask_2d = mask

            if sens_maps is not None:
                masked_kspace = kspace_complex * mask_2d.unsqueeze(1)
                coil_images = ifft2c(masked_kspace, norm=self.norm)
                image = torch.sum(torch.conj(sens_maps) * coil_images, dim=1)
            else:
                masked_kspace = kspace_complex * mask_2d
                image = ifft2c(masked_kspace, norm=self.norm)

            return complex_to_channels(image.unsqueeze(1)).squeeze(2)

    def normal_op(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Normal operator: A^H * A

        This is forward followed by adjoint.
        Important for iterative reconstruction algorithms.
        """
        kspace = self.forward_op(image, mask, sens_maps)
        return self.adjoint_op(kspace, mask, sens_maps)


def forward_model(
    image: torch.Tensor,
    mask: torch.Tensor,
    norm: str = "ortho"
) -> torch.Tensor:
    """
    Functional forward model for single-coil MRI.

    Args:
        image: Complex image (B, H, W) or 2-channel (B, 2, H, W)
        mask: Sampling mask (B, 1, H, W) or (B, H, W)
        norm: FFT normalization

    Returns:
        Masked k-space
    """
    # Handle 2-channel input
    if image.ndim == 4 and image.shape[1] == 2:
        image = channels_to_complex(image)

    # Forward FFT
    kspace = fft2c(image, norm=norm)

    # Apply mask
    if mask.ndim == 4:
        mask = mask.squeeze(1)

    return kspace * mask


def adjoint_model(
    kspace: torch.Tensor,
    mask: torch.Tensor,
    norm: str = "ortho"
) -> torch.Tensor:
    """
    Functional adjoint model for single-coil MRI.

    Args:
        kspace: K-space data (B, H, W) complex or (B, 2, H, W) real
        mask: Sampling mask
        norm: FFT normalization

    Returns:
        Image (complex or 2-channel)
    """
    # Handle 2-channel input
    return_2ch = False
    if kspace.ndim == 4 and kspace.shape[1] == 2:
        kspace = channels_to_complex(kspace)
        return_2ch = True

    # Apply mask
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    masked_kspace = kspace * mask

    # Inverse FFT
    image = ifft2c(masked_kspace, norm=norm)

    if return_2ch:
        return complex_to_channels(image)
    return image


def compute_sense_model(
    kspace: torch.Tensor,
    sens_maps: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SENSE (SENSitivity Encoding) model for multi-coil MRI.

    SENSE uses coil sensitivity information to unfold aliased images
    from accelerated parallel imaging.

    Args:
        kspace: Multi-coil k-space (B, num_coils, H, W) complex
        sens_maps: Sensitivity maps (B, num_coils, H, W) complex
        mask: Sampling mask (B, 1, H, W)

    Returns:
        image: Reconstructed image (B, H, W) complex
        residual: Reconstruction residual for data consistency
    """
    B, num_coils, H, W = kspace.shape

    # Aliased coil images
    coil_images = ifft2c(kspace)  # (B, num_coils, H, W)

    # SENSE reconstruction: weighted combination
    # image = sum(conj(S_i) * x_i) / sum(|S_i|^2)
    numerator = torch.sum(torch.conj(sens_maps) * coil_images, dim=1)
    denominator = torch.sum(torch.abs(sens_maps) ** 2, dim=1) + 1e-8

    image = numerator / denominator

    # Compute residual (for data consistency)
    # Forward project: image * sens_maps → FFT → mask
    predicted_kspace = fft2c(image.unsqueeze(1) * sens_maps)
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    residual = kspace - predicted_kspace * mask.unsqueeze(1)

    return image, residual


class ConjugateGradient(nn.Module):
    """
    Conjugate Gradient (CG) solver for MRI reconstruction.

    Solves: A^H * A * x = A^H * b
    Where A is the forward MRI model.

    This is a classical iterative algorithm that finds the
    least-squares solution efficiently.
    """

    def __init__(
        self,
        num_iterations: int = 10,
        tolerance: float = 1e-6
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.tolerance = tolerance

    def forward(
        self,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
        initial_guess: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Solve for image given measured k-space.

        Args:
            measured_kspace: Measured k-space (B, H, W) complex
            mask: Sampling mask
            initial_guess: Optional initial image estimate

        Returns:
            Reconstructed image (B, H, W) complex
        """
        # Initialize
        if initial_guess is None:
            x = ifft2c(measured_kspace)
        else:
            x = initial_guess

        # A^H * b (right-hand side)
        b = ifft2c(measured_kspace)

        # A^H * A * x
        def normal_op(img):
            k = fft2c(img)
            if mask.ndim == 4:
                m = mask.squeeze(1)
            else:
                m = mask
            return ifft2c(k * m)

        # Initial residual: r = b - A^H*A*x
        r = b - normal_op(x)
        p = r.clone()
        rs_old = torch.sum(torch.conj(r) * r, dim=(-2, -1), keepdim=True).real

        for _ in range(self.num_iterations):
            Ap = normal_op(p)
            pAp = torch.sum(torch.conj(p) * Ap, dim=(-2, -1), keepdim=True).real + 1e-10

            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = torch.sum(torch.conj(r) * r, dim=(-2, -1), keepdim=True).real

            if torch.all(rs_new < self.tolerance):
                break

            beta = rs_new / (rs_old + 1e-10)
            p = r + beta * p
            rs_old = rs_new

        return x


class ADMM_Solver(nn.Module):
    """
    ADMM (Alternating Direction Method of Multipliers) for MRI reconstruction.

    Solves: minimize ||Ax - b||_2^2 + λ * R(x)

    Where R(x) is a regularizer (e.g., total variation, learned prior).

    INTUITION:
    ADMM splits the problem into simpler subproblems:
    1. Data consistency step (make x match measurements)
    2. Regularization step (make x look like a real image)
    3. Update dual variables (balance between 1 and 2)
    """

    def __init__(
        self,
        num_iterations: int = 10,
        rho: float = 1.0,
        lambda_reg: float = 0.01
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.rho = rho
        self.lambda_reg = lambda_reg

    def forward(
        self,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
        denoiser: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        ADMM reconstruction with optional learned denoiser.

        Args:
            measured_kspace: Measured k-space (B, H, W) complex
            mask: Sampling mask
            denoiser: Optional denoising network for regularization

        Returns:
            Reconstructed image
        """
        # Initialize
        x = ifft2c(measured_kspace)  # Initial estimate
        z = x.clone()  # Auxiliary variable
        u = torch.zeros_like(x)  # Dual variable

        if mask.ndim == 4:
            mask = mask.squeeze(1)

        for _ in range(self.num_iterations):
            # Step 1: x-update (data consistency)
            # minimize ||Ax - b||^2 + (rho/2)||x - z + u||^2
            # Solution via proximal gradient

            # Gradient of data term
            kspace_pred = fft2c(x)
            grad_data = ifft2c((kspace_pred - measured_kspace) * mask)

            # Gradient of augmented Lagrangian term
            grad_aug = self.rho * (x - z + u)

            # Gradient step
            x = x - 0.5 * (grad_data + grad_aug)

            # Step 2: z-update (proximal of regularizer)
            v = x + u  # "noisy" estimate

            if denoiser is not None:
                # Use learned denoiser as proximal operator
                # Convert to real for denoiser
                v_mag = torch.abs(v)
                v_phase = torch.angle(v)

                # Denoise magnitude
                v_mag_input = v_mag.unsqueeze(1)  # (B, 1, H, W)
                z_mag = denoiser(v_mag_input).squeeze(1)  # (B, H, W)

                # Reconstruct complex
                z = z_mag * torch.exp(1j * v_phase)
            else:
                # Simple soft thresholding (TV-like)
                threshold = self.lambda_reg / self.rho
                z = torch.sign(v) * torch.maximum(
                    torch.abs(v) - threshold,
                    torch.zeros_like(torch.abs(v))
                )

            # Step 3: u-update (dual variable)
            u = u + x - z

        return x
