"""
SIREN: Sinusoidal Representation Networks

SIREN uses sine activations instead of ReLU, which enables
learning high-frequency details and smooth derivatives.

WHY SIREN FOR MRI?
=================
1. Standard networks (ReLU) struggle with high-frequency details
2. SIREN naturally represents smooth, continuous signals
3. MRI images have smooth intensity variations
4. SIREN allows arbitrary resolution sampling

INTUITION:
=========
Traditional: Image[x][y] = pixel value (fixed grid)
SIREN: f(x, y) = intensity (any continuous x, y)

The network learns a continuous function that maps
coordinates to intensities. It's like storing the "recipe"
for the image rather than the pixel values.

Key insight: sin(ωx) is periodic and smooth, perfect for
representing natural signals like MRI.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SIRENConfig:
    """Configuration for SIREN network."""
    in_features: int = 2  # (x, y) coordinates
    hidden_features: int = 256
    hidden_layers: int = 5
    out_features: int = 1  # Intensity
    first_omega: float = 30.0  # First layer frequency
    hidden_omega: float = 30.0  # Hidden layer frequency
    use_fourier_features: bool = True  # Input Fourier mapping
    fourier_scale: float = 10.0


class SineLayer(nn.Module):
    """
    Single SIREN layer with sine activation.

    The key innovation: use sin(ω * Wx + b) instead of ReLU(Wx + b)

    Why sine?
    - Periodic: naturally captures repeating patterns
    - Smooth: infinitely differentiable
    - Frequency control: ω controls how fast the function oscillates
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 30.0,
        is_first: bool = False,
        use_bias: bool = True
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            omega: Frequency multiplier (higher = more detail)
            is_first: Is this the first layer?
            use_bias: Include bias term
        """
        super().__init__()
        self.omega = omega
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        # Special initialization for SIREN
        self._init_weights()

    def _init_weights(self):
        """
        SIREN-specific weight initialization.

        Critical for training stability and convergence.
        """
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/n, 1/n]
                bound = 1 / self.linear.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/n)/omega, sqrt(6/n)/omega]
                bound = np.sqrt(6 / self.linear.in_features) / self.omega

            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation followed by sine activation."""
        return torch.sin(self.omega * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN Network for Implicit MRI Representation.

    Maps (x, y) coordinates to intensity values.
    Can be sampled at ANY resolution after training.

    Architecture:
    Input (x,y) → [SineLayer]×N → Linear → Output (intensity)

    USAGE EXAMPLE:
    ==============
    # Train SIREN on an MRI slice
    siren = SIREN(config)
    coords = create_coordinate_grid(320, 320)  # (320*320, 2)
    target_intensities = image.flatten()  # (320*320,)

    for step in range(1000):
        pred = siren(coords)
        loss = (pred - target_intensities).pow(2).mean()
        loss.backward()
        optimizer.step()

    # Sample at HIGHER resolution!
    high_res_coords = create_coordinate_grid(640, 640)
    high_res_image = siren(high_res_coords).reshape(640, 640)
    """

    def __init__(self, config: Optional[SIRENConfig] = None):
        """
        Args:
            config: SIREN configuration
        """
        super().__init__()

        if config is None:
            config = SIRENConfig()

        self.config = config

        # Optional Fourier feature mapping for input
        if config.use_fourier_features:
            from .fourier_features import GaussianFourierFeatures
            self.fourier = GaussianFourierFeatures(
                config.in_features,
                mapping_size=128,
                scale=config.fourier_scale
            )
            actual_in = 128 * 2  # Sin and cos components
        else:
            self.fourier = None
            actual_in = config.in_features

        # Build network
        layers = []

        # First layer (special initialization)
        layers.append(SineLayer(
            actual_in,
            config.hidden_features,
            omega=config.first_omega,
            is_first=True
        ))

        # Hidden layers
        for _ in range(config.hidden_layers):
            layers.append(SineLayer(
                config.hidden_features,
                config.hidden_features,
                omega=config.hidden_omega,
                is_first=False
            ))

        # Output layer (linear, no sine)
        self.output_linear = nn.Linear(config.hidden_features, config.out_features)

        # Initialize output layer
        with torch.no_grad():
            bound = np.sqrt(6 / config.hidden_features) / config.hidden_omega
            self.output_linear.weight.uniform_(-bound, bound)
            if self.output_linear.bias is not None:
                self.output_linear.bias.zero_()

        self.layers = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Map coordinates to intensities.

        Args:
            coords: Coordinate tensor (..., 2) with values in [-1, 1]

        Returns:
            Intensities (..., 1)
        """
        original_shape = coords.shape[:-1]

        # Flatten for processing
        coords_flat = coords.reshape(-1, coords.shape[-1])

        # Optional Fourier features
        if self.fourier is not None:
            coords_flat = self.fourier(coords_flat)

        # Pass through SIREN
        h = self.layers(coords_flat)
        out = self.output_linear(h)

        # Restore shape
        return out.reshape(*original_shape, -1)

    def forward_with_gradients(
        self,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute output AND spatial gradients.

        SIREN naturally provides smooth gradients, useful for:
        - Edge detection
        - Surface normals
        - Physics-informed losses

        Args:
            coords: Coordinates with grad enabled

        Returns:
            output: Intensity values
            gradient: Spatial gradients ∂f/∂x, ∂f/∂y
        """
        coords = coords.requires_grad_(True)
        output = self.forward(coords)

        # Compute gradient
        gradient = torch.autograd.grad(
            outputs=output,
            inputs=coords,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]

        return output, gradient


class ModulatedSIREN(nn.Module):
    """
    Modulated SIREN for conditional INR.

    Allows conditioning on external signals (e.g., different slices,
    different subjects, or different time points in dynamic MRI).

    The modulation signal adjusts the network's behavior.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        cond_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 5,
        out_dim: int = 1,
        omega: float = 30.0
    ):
        """
        Args:
            coord_dim: Coordinate dimension (2 for 2D, 3 for 3D)
            cond_dim: Conditioning dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            out_dim: Output dimension
            omega: SIREN frequency
        """
        super().__init__()

        self.omega = omega

        # Coordinate processing
        self.coord_layer = SineLayer(coord_dim, hidden_dim, omega, is_first=True)

        # Modulation networks (one per layer)
        self.mod_networks = nn.ModuleList()
        for _ in range(num_layers):
            self.mod_networks.append(
                nn.Sequential(
                    nn.Linear(cond_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * 2)  # Scale and shift
                )
            )

        # Main layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        coords: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with conditioning.

        Args:
            coords: Coordinates (..., coord_dim)
            condition: Conditioning vector (B, cond_dim)

        Returns:
            Output (..., out_dim)
        """
        # Initial coordinate embedding
        h = self.coord_layer(coords)

        # Process through modulated layers
        for layer, mod_net in zip(self.layers, self.mod_networks):
            # Get modulation parameters
            mod = mod_net(condition)
            scale, shift = mod.chunk(2, dim=-1)

            # Expand for broadcasting
            while scale.dim() < h.dim():
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)

            # Apply layer with modulation
            h = layer(h)
            h = h * (1 + scale) + shift
            h = torch.sin(self.omega * h)

        return self.output_layer(h)


class MultiScaleSIREN(nn.Module):
    """
    Multi-scale SIREN for capturing both global and local features.

    Uses multiple SIRENs at different frequency scales,
    similar to multi-resolution analysis.
    """

    def __init__(
        self,
        scales: List[float] = [1.0, 5.0, 20.0],
        hidden_features: int = 128,
        hidden_layers: int = 3,
        in_features: int = 2,
        out_features: int = 1
    ):
        """
        Args:
            scales: Frequency scales (omega values)
            hidden_features: Features per scale
            hidden_layers: Layers per scale
            in_features: Input dimension
            out_features: Output dimension
        """
        super().__init__()

        self.sirens = nn.ModuleList()
        for omega in scales:
            config = SIRENConfig(
                in_features=in_features,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                out_features=out_features,
                first_omega=omega,
                hidden_omega=omega,
                use_fourier_features=False
            )
            self.sirens.append(SIREN(config))

        # Learned combination weights
        self.weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Combine multi-scale outputs."""
        outputs = []
        for siren in self.sirens:
            outputs.append(siren(coords))

        # Weighted sum
        weights = torch.softmax(self.weights, dim=0)
        result = sum(w * out for w, out in zip(weights, outputs))

        return result


def create_coordinate_grid(
    height: int,
    width: int,
    device: str = 'cpu',
    normalized: bool = True
) -> torch.Tensor:
    """
    Create a grid of (x, y) coordinates.

    Args:
        height: Grid height
        width: Grid width
        device: Device to create tensor on
        normalized: If True, coordinates in [-1, 1], else [0, H-1]

    Returns:
        Coordinates tensor (H, W, 2)
    """
    if normalized:
        y = torch.linspace(-1, 1, height, device=device)
        x = torch.linspace(-1, 1, width, device=device)
    else:
        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)

    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)

    return coords


def sample_image_from_siren(
    siren: SIREN,
    height: int,
    width: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample a complete image from trained SIREN.

    Args:
        siren: Trained SIREN network
        height: Output height
        width: Output width
        device: Device

    Returns:
        Image tensor (H, W)
    """
    coords = create_coordinate_grid(height, width, device)
    with torch.no_grad():
        intensities = siren(coords)
    return intensities.squeeze(-1)
