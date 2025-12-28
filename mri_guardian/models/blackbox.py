"""
Black-Box Models and Hallucination Injection

These models simulate "untrustworthy" reconstruction systems
that can hallucinate structures not present in the measurements.

PURPOSE:
=======
1. Test whether Guardian can detect hallucinations
2. Simulate real-world AI systems that may have flaws
3. Generate controlled test cases for auditing

HALLUCINATION TYPES:
===================
1. False positives: Add structures that don't exist (synthetic lesions)
2. False negatives: Remove structures that do exist (missed pathology)
3. Texture hallucination: Invent fine details not in data
4. Over-sharpening: Create artificial edges
5. Over-smoothing: Lose real details
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from .unet import UNet
from ..data.kspace_ops import complex_abs


@dataclass
class HallucinationConfig:
    """Configuration for hallucination injection."""
    # Synthetic lesion injection
    lesion_prob: float = 0.3  # Probability of adding lesions
    num_lesions_range: Tuple[int, int] = (1, 3)  # Range of lesions to add
    lesion_size_range: Tuple[int, int] = (4, 12)  # Size in pixels
    lesion_intensity_range: Tuple[float, float] = (0.3, 0.8)  # Relative intensity

    # Texture hallucination
    texture_prob: float = 0.2
    texture_patch_size: int = 16
    texture_strength: float = 0.1

    # Over-sharpening
    sharpen_prob: float = 0.2
    sharpen_strength: float = 0.5

    # Over-smoothing (missing details)
    smooth_prob: float = 0.2
    smooth_kernel_size: int = 5

    # Structure removal (false negatives)
    removal_prob: float = 0.1
    removal_size_range: Tuple[int, int] = (5, 15)


class HallucinationInjector(nn.Module):
    """
    Injects various types of hallucinations into MRI reconstructions.

    This is used to create test cases for the Guardian auditor.
    """

    def __init__(self, config: Optional[HallucinationConfig] = None):
        super().__init__()
        self.config = config or HallucinationConfig()

        # Learnable sharpening kernel
        self.sharpen_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        sharpen_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.sharpen_conv.weight.data = sharpen_kernel
        self.sharpen_conv.weight.requires_grad = False

    def inject_synthetic_lesion(
        self,
        image: torch.Tensor,
        return_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inject synthetic lesion(s) into image.

        Creates small bright spots that simulate pathology.
        These are HALLUCINATIONS - they don't exist in the original.

        Args:
            image: Input image (B, 1, H, W)
            return_mask: Return binary mask of injected regions

        Returns:
            Hallucinated image, optional mask
        """
        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)

        for b in range(B):
            if np.random.random() > self.config.lesion_prob:
                continue

            num_lesions = np.random.randint(*self.config.num_lesions_range)

            for _ in range(num_lesions):
                # Random position (avoid edges)
                margin = self.config.lesion_size_range[1]
                cy = np.random.randint(margin, H - margin)
                cx = np.random.randint(margin, W - margin)

                # Random size
                size = np.random.randint(*self.config.lesion_size_range)

                # Random intensity (relative to local region)
                local_mean = output[b, 0, max(0, cy-10):min(H, cy+10),
                                    max(0, cx-10):min(W, cx+10)].mean()
                intensity = local_mean * np.random.uniform(*self.config.lesion_intensity_range)

                # Create elliptical lesion
                y, x = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )
                y = y.float()
                x = x.float()

                # Random ellipse axes
                a = size / 2 * np.random.uniform(0.7, 1.3)
                b_axis = size / 2 * np.random.uniform(0.7, 1.3)

                dist = ((x - cx) / a) ** 2 + ((y - cy) / b_axis) ** 2
                lesion_mask = (dist <= 1).float()

                # Smooth edges
                lesion_mask = F.avg_pool2d(
                    lesion_mask.unsqueeze(0).unsqueeze(0),
                    3, stride=1, padding=1
                ).squeeze()

                # Add lesion
                output[b, 0] = output[b, 0] + intensity * lesion_mask
                mask[b, 0] = torch.maximum(mask[b, 0], lesion_mask)

        if return_mask:
            return output, mask
        return output, None

    def inject_texture_hallucination(
        self,
        image: torch.Tensor,
        return_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inject hallucinated texture in local patches.

        Creates artificial fine-scale patterns not in the original data.

        Args:
            image: Input image (B, 1, H, W)
            return_mask: Return mask of affected regions

        Returns:
            Hallucinated image, optional mask
        """
        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)

        patch_size = self.config.texture_patch_size

        for b in range(B):
            if np.random.random() > self.config.texture_prob:
                continue

            # Select random patch
            py = np.random.randint(0, H - patch_size)
            px = np.random.randint(0, W - patch_size)

            # Generate random texture (high-frequency noise)
            texture = torch.randn(1, 1, patch_size, patch_size, device=device)
            texture = texture * self.config.texture_strength

            # Scale by local intensity
            local_std = output[b, 0, py:py+patch_size, px:px+patch_size].std()
            texture = texture * local_std

            # Apply with soft edges
            edge_mask = torch.ones(patch_size, patch_size, device=device)
            margin = patch_size // 4
            for i in range(margin):
                factor = i / margin
                edge_mask[i, :] *= factor
                edge_mask[-(i+1), :] *= factor
                edge_mask[:, i] *= factor
                edge_mask[:, -(i+1)] *= factor

            texture = texture.squeeze() * edge_mask

            output[b, 0, py:py+patch_size, px:px+patch_size] += texture
            mask[b, 0, py:py+patch_size, px:px+patch_size] = edge_mask

        if return_mask:
            return output, mask
        return output, None

    def inject_over_sharpening(
        self,
        image: torch.Tensor,
        return_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Over-sharpen image to create artificial edges.

        Makes the image look "too crisp" with edges that
        aren't supported by the underlying data.
        """
        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)

        for b in range(B):
            if np.random.random() > self.config.sharpen_prob:
                continue

            # Apply sharpening
            sharpened = self.sharpen_conv(image[b:b+1])
            alpha = self.config.sharpen_strength

            output[b:b+1] = (1 - alpha) * image[b:b+1] + alpha * sharpened

            # Mask: where sharpening changed things significantly
            diff = torch.abs(output[b:b+1] - image[b:b+1])
            mask[b:b+1] = (diff > diff.mean() + diff.std()).float()

        if return_mask:
            return output, mask
        return output, None

    def inject_over_smoothing(
        self,
        image: torch.Tensor,
        return_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Over-smooth image, losing real details.

        This simulates false negatives - the model "misses" fine structures.
        """
        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)

        k = self.config.smooth_kernel_size

        for b in range(B):
            if np.random.random() > self.config.smooth_prob:
                continue

            # Apply local smoothing to random region
            region_size = np.random.randint(32, 64)
            py = np.random.randint(0, H - region_size)
            px = np.random.randint(0, W - region_size)

            patch = output[b, 0, py:py+region_size, px:px+region_size]

            # Gaussian blur
            smoothed = F.avg_pool2d(
                patch.unsqueeze(0).unsqueeze(0),
                k, stride=1, padding=k//2
            ).squeeze()

            # Make patch same size
            smoothed = F.interpolate(
                smoothed.unsqueeze(0).unsqueeze(0),
                size=(region_size, region_size),
                mode='bilinear',
                align_corners=True
            ).squeeze()

            # Blend with soft edges
            blend = torch.ones(region_size, region_size, device=device)
            margin = region_size // 4
            for i in range(margin):
                factor = i / margin
                blend[i, :] *= factor
                blend[-(i+1), :] *= factor
                blend[:, i] *= factor
                blend[:, -(i+1)] *= factor

            output[b, 0, py:py+region_size, px:px+region_size] = (
                blend * smoothed + (1 - blend) * patch
            )
            mask[b, 0, py:py+region_size, px:px+region_size] = blend

        if return_mask:
            return output, mask
        return output, None

    def inject_structure_removal(
        self,
        image: torch.Tensor,
        return_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Remove small structures (simulate missed pathology).

        This is a FALSE NEGATIVE - real structures are hidden.
        """
        B, C, H, W = image.shape
        device = image.device

        output = image.clone()
        mask = torch.zeros(B, 1, H, W, device=device)

        for b in range(B):
            if np.random.random() > self.config.removal_prob:
                continue

            # Find a bright spot to remove
            # (In real pathology detection, this would be a lesion)
            size = np.random.randint(*self.config.removal_size_range)

            cy = np.random.randint(size, H - size)
            cx = np.random.randint(size, W - size)

            # Replace with local average
            local_region = output[b, 0,
                                  max(0, cy-size*2):min(H, cy+size*2),
                                  max(0, cx-size*2):min(W, cx+size*2)]
            fill_value = local_region.median()

            # Create removal mask
            y, x = torch.meshgrid(
                torch.arange(H, device=device).float(),
                torch.arange(W, device=device).float(),
                indexing='ij'
            )
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            removal_mask = torch.clamp(1 - dist / size, 0, 1)

            # Remove structure
            output[b, 0] = output[b, 0] * (1 - removal_mask) + fill_value * removal_mask
            mask[b, 0] = torch.maximum(mask[b, 0], removal_mask)

        if return_mask:
            return output, mask
        return output, None

    def forward(
        self,
        image: torch.Tensor,
        hallucination_types: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply multiple hallucination types.

        Args:
            image: Input image (B, 1, H, W)
            hallucination_types: List of types to apply
                Options: "lesion", "texture", "sharpen", "smooth", "remove"
                If None, randomly selects.

        Returns:
            Dict with:
                - 'output': Hallucinated image
                - 'mask': Combined hallucination mask
                - 'types': List of applied hallucination types
        """
        if hallucination_types is None:
            hallucination_types = ["lesion", "texture", "sharpen", "smooth", "remove"]

        output = image.clone()
        combined_mask = torch.zeros_like(image)
        applied_types = []

        for h_type in hallucination_types:
            if h_type == "lesion":
                output, mask = self.inject_synthetic_lesion(output)
                if mask is not None and mask.sum() > 0:
                    combined_mask = torch.maximum(combined_mask, mask)
                    applied_types.append("lesion")

            elif h_type == "texture":
                output, mask = self.inject_texture_hallucination(output)
                if mask is not None and mask.sum() > 0:
                    combined_mask = torch.maximum(combined_mask, mask)
                    applied_types.append("texture")

            elif h_type == "sharpen":
                output, mask = self.inject_over_sharpening(output)
                if mask is not None and mask.sum() > 0:
                    combined_mask = torch.maximum(combined_mask, mask)
                    applied_types.append("sharpen")

            elif h_type == "smooth":
                output, mask = self.inject_over_smoothing(output)
                if mask is not None and mask.sum() > 0:
                    combined_mask = torch.maximum(combined_mask, mask)
                    applied_types.append("smooth")

            elif h_type == "remove":
                output, mask = self.inject_structure_removal(output)
                if mask is not None and mask.sum() > 0:
                    combined_mask = torch.maximum(combined_mask, mask)
                    applied_types.append("remove")

        return {
            'output': output,
            'mask': combined_mask,
            'types': applied_types
        }


class BlackBoxModel(nn.Module):
    """
    Black-Box MRI Reconstruction Model.

    This is a standard UNet that we treat as "untrustworthy".
    We don't assume knowledge of its internals.
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_levels: int = 4,
        include_hallucination: bool = False
    ):
        """
        Args:
            base_channels: Number of base features
            num_levels: UNet depth
            include_hallucination: Add hallucinations to output
        """
        super().__init__()

        self.unet = UNet(
            in_channels=1,
            out_channels=1,
            base_channels=base_channels,
            num_levels=num_levels
        )

        self.include_hallucination = include_hallucination
        if include_hallucination:
            self.hallucinator = HallucinationInjector()

    def forward(
        self,
        zf_recon: torch.Tensor,
        return_hallucination_mask: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Black-box reconstruction.

        Args:
            zf_recon: Zero-filled reconstruction (B, 1, H, W)
            return_hallucination_mask: Return mask of hallucinated regions

        Returns:
            Dict with 'output' and optionally 'hallucination_mask'
        """
        # Standard UNet reconstruction
        recon = self.unet(zf_recon)

        result = {'output': recon}

        # Optionally add hallucinations
        if self.include_hallucination and self.training is False:
            halluc_result = self.hallucinator(recon)
            result['output'] = halluc_result['output']
            result['hallucination_mask'] = halluc_result['mask']

        return result


class HallucinatingModel(nn.Module):
    """
    Model specifically designed to hallucinate.

    Used for controlled testing of the auditor.
    Always adds some form of hallucination.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        hallucination_config: Optional[HallucinationConfig] = None
    ):
        super().__init__()

        if base_model is None:
            base_model = UNet(1, 1, 32, 4)

        self.base_model = base_model
        self.hallucinator = HallucinationInjector(hallucination_config)

    def forward(
        self,
        x: torch.Tensor,
        hallucination_types: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct and hallucinate.

        Args:
            x: Input (B, 1, H, W)
            hallucination_types: Types of hallucinations to inject

        Returns:
            Dict with output, hallucination_mask, hallucination_types
        """
        # Base reconstruction
        recon = self.base_model(x)

        # Always add hallucinations
        result = self.hallucinator(recon, hallucination_types)

        return {
            'output': result['output'],
            'clean_output': recon,
            'hallucination_mask': result['mask'],
            'hallucination_types': result['types']
        }
