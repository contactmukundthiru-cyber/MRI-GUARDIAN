"""
Visualization Module for MRI-GUARDIAN

Provides plotting utilities for:
- MRI images and k-space
- Reconstruction comparisons
- Detection results
- Experiment outputs
"""

from .plotting import (
    plot_image,
    plot_comparison,
    plot_kspace,
    plot_metrics_bar,
    plot_training_curves,
    save_figure,
)
from .kspace_viz import (
    visualize_kspace_magnitude,
    visualize_sampling_mask,
    visualize_kspace_difference,
)
from .comparison import (
    create_comparison_figure,
    create_hallucination_detection_figure,
    create_robustness_figure,
)

__all__ = [
    # Basic plotting
    "plot_image",
    "plot_comparison",
    "plot_kspace",
    "plot_metrics_bar",
    "plot_training_curves",
    "save_figure",
    # K-space visualization
    "visualize_kspace_magnitude",
    "visualize_sampling_mask",
    "visualize_kspace_difference",
    # Comparison figures
    "create_comparison_figure",
    "create_hallucination_detection_figure",
    "create_robustness_figure",
]
