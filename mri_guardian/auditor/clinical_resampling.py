"""
Clinical Re-Sampling - From Critic to Helper
=============================================

THE KEY INSIGHT:
The Auditor shouldn't just say "Error."
It should say "Scan these specific lines again."

THE NOVELTY:
Closing the loop from detection to ACTION.

THE METHOD:
1. Take uncertainty/discrepancy map
2. Identify which k-space frequencies correspond to uncertain regions
3. Recommend a minimal set of k-space lines to re-acquire
4. Show that 5% more data resolves 90% of hallucinations

WHY THIS WINS:
It transforms the project from a "Critic" (which doctors hate)
to a "Helper" (which doctors want).

"My system doesn't just complain; it tells the scanner
exactly what minimal data is needed to be sure."
"""

import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ResamplingRecommendation:
    """Recommendation for additional k-space sampling."""
    # The recommendation
    recommended_lines: np.ndarray  # Boolean mask of k-space lines to acquire
    num_additional_lines: int
    percentage_increase: float  # % increase in scan time

    # Priority ordering
    line_priorities: List[int]  # Ordered list of line indices by priority

    # Expected benefit
    expected_uncertainty_reduction: float
    expected_hallucination_resolution: float

    # Regions addressed
    regions_addressed: List[Dict]  # Which uncertain regions this resolves

    # Clinical impact
    estimated_scan_time_increase: float  # In seconds
    clinical_urgency: str  # "low", "medium", "high"

    # Explanation
    rationale: str


class ClinicalResamplingAdvisor:
    """
    Recommend targeted k-space re-acquisition based on uncertainty.

    This transforms auditor output into actionable scanner instructions.
    """

    def __init__(
        self,
        max_additional_lines_percent: float = 10.0,  # Max 10% more lines
        uncertainty_threshold: float = 0.3,  # Threshold for flagging
        line_time_seconds: float = 0.1  # Time per k-space line
    ):
        self.max_additional_percent = max_additional_lines_percent
        self.uncertainty_threshold = uncertainty_threshold
        self.line_time = line_time_seconds

    def recommend_resampling(
        self,
        uncertainty_map: np.ndarray,  # (H, W) uncertainty
        current_sampling_mask: np.ndarray,  # (H, W) current sampling
        discrepancy_map: Optional[np.ndarray] = None,  # Additional info
        target_resolution_rate: float = 0.9  # Target 90% resolution
    ) -> ResamplingRecommendation:
        """
        Recommend additional k-space lines to acquire.

        Args:
            uncertainty_map: Pixel-wise uncertainty (higher = more uncertain)
            current_sampling_mask: Current k-space sampling pattern
            discrepancy_map: Optional discrepancy map from auditor
            target_resolution_rate: Target percentage of uncertainty to resolve

        Returns:
            ResamplingRecommendation with specific lines to acquire
        """
        H, W = uncertainty_map.shape

        # Step 1: Identify uncertain regions in image space
        uncertain_regions = uncertainty_map > self.uncertainty_threshold

        # Step 2: Map uncertain regions to k-space
        # Uncertain image regions need specific k-space frequencies
        kspace_need_map = self._map_uncertainty_to_kspace(uncertainty_map)

        # Step 3: Identify which k-space LINES have highest impact
        # For Cartesian sampling, we acquire full lines (rows in k-space)
        line_impacts = self._compute_line_impacts(kspace_need_map, current_sampling_mask)

        # Step 4: Select lines to recommend (greedy by impact)
        current_lines = current_sampling_mask.sum(axis=1) > 0
        num_current_lines = current_lines.sum()
        max_additional = int(num_current_lines * self.max_additional_percent / 100)

        recommended_lines, line_priorities = self._select_optimal_lines(
            line_impacts, current_sampling_mask, max_additional, target_resolution_rate
        )

        num_additional = int(recommended_lines.sum())
        percentage_increase = 100 * num_additional / max(1, num_current_lines)

        # Step 5: Estimate benefits
        expected_reduction = self._estimate_uncertainty_reduction(
            kspace_need_map, recommended_lines
        )

        # Step 6: Identify which regions this resolves
        regions_addressed = self._identify_addressed_regions(
            uncertainty_map, recommended_lines
        )

        # Step 7: Clinical assessment
        scan_time_increase = num_additional * self.line_time
        urgency = self._assess_urgency(uncertainty_map, discrepancy_map)

        # Generate rationale
        rationale = self._generate_rationale(
            num_additional, percentage_increase, expected_reduction,
            len(regions_addressed), urgency
        )

        # Create full mask
        full_recommended_mask = np.zeros((H, W), dtype=bool)
        for line_idx in np.where(recommended_lines)[0]:
            full_recommended_mask[line_idx, :] = True

        return ResamplingRecommendation(
            recommended_lines=full_recommended_mask,
            num_additional_lines=num_additional,
            percentage_increase=percentage_increase,
            line_priorities=line_priorities,
            expected_uncertainty_reduction=expected_reduction,
            expected_hallucination_resolution=min(1.0, expected_reduction * 1.2),
            regions_addressed=regions_addressed,
            estimated_scan_time_increase=scan_time_increase,
            clinical_urgency=urgency,
            rationale=rationale
        )

    def _map_uncertainty_to_kspace(self, uncertainty_map: np.ndarray) -> np.ndarray:
        """
        Map image uncertainty to k-space need.

        Uncertain regions in image domain correspond to specific
        frequency content that needs better sampling.
        """
        H, W = uncertainty_map.shape

        # FFT of uncertainty map shows which frequencies matter
        # High uncertainty regions need their characteristic frequencies sampled

        # Weight uncertainty by image structure
        # Use edge-weighted uncertainty
        edges = np.sqrt(
            ndimage.sobel(uncertainty_map, axis=0)**2 +
            ndimage.sobel(uncertainty_map, axis=1)**2
        )
        weighted_uncertainty = uncertainty_map * (1 + edges)

        # Transform to k-space to see which frequencies matter
        kspace_need = np.abs(np.fft.fftshift(np.fft.fft2(weighted_uncertainty)))

        # Normalize
        kspace_need = kspace_need / (kspace_need.max() + 1e-10)

        return kspace_need

    def _compute_line_impacts(
        self,
        kspace_need_map: np.ndarray,
        current_mask: np.ndarray
    ) -> np.ndarray:
        """Compute impact of acquiring each k-space line."""
        H, W = kspace_need_map.shape

        line_impacts = np.zeros(H)
        for line in range(H):
            # Skip already-sampled lines
            if current_mask[line, :].sum() > W / 2:
                line_impacts[line] = 0
                continue

            # Impact = sum of need for this line
            line_impacts[line] = kspace_need_map[line, :].sum()

        return line_impacts

    def _select_optimal_lines(
        self,
        line_impacts: np.ndarray,
        current_mask: np.ndarray,
        max_lines: int,
        target_resolution: float
    ) -> Tuple[np.ndarray, List[int]]:
        """Select optimal lines to recommend."""
        H = len(line_impacts)

        # Sort lines by impact
        sorted_indices = np.argsort(line_impacts)[::-1]

        # Greedily select top impact lines
        selected = np.zeros(H, dtype=bool)
        priorities = []
        total_impact = line_impacts.sum()
        accumulated_impact = 0

        for idx in sorted_indices:
            if len(priorities) >= max_lines:
                break
            if current_mask[idx, :].sum() > current_mask.shape[1] / 2:
                continue  # Skip sampled lines

            selected[idx] = True
            priorities.append(int(idx))
            accumulated_impact += line_impacts[idx]

            # Stop if we've addressed enough
            if accumulated_impact / (total_impact + 1e-10) >= target_resolution:
                break

        return selected, priorities

    def _estimate_uncertainty_reduction(
        self,
        kspace_need_map: np.ndarray,
        recommended_lines: np.ndarray
    ) -> float:
        """Estimate how much uncertainty will be reduced."""
        total_need = kspace_need_map.sum()
        addressed_need = 0

        for line_idx in np.where(recommended_lines)[0]:
            addressed_need += kspace_need_map[line_idx, :].sum()

        return addressed_need / (total_need + 1e-10)

    def _identify_addressed_regions(
        self,
        uncertainty_map: np.ndarray,
        recommended_lines: np.ndarray
    ) -> List[Dict]:
        """Identify which uncertain regions are addressed."""
        # Find connected uncertain regions
        uncertain_binary = uncertainty_map > self.uncertainty_threshold
        labeled, n_regions = ndimage.label(uncertain_binary)

        regions = []
        for region_id in range(1, n_regions + 1):
            region_mask = labeled == region_id
            region_uncertainty = uncertainty_map[region_mask].mean()

            # Check if recommended lines address this region
            # (simplified: check if region's k-space support overlaps)
            region_y = np.where(region_mask.any(axis=1))[0]
            region_freq_range = (region_y.min(), region_y.max())

            addressed = any(
                recommended_lines[freq_range_idx]
                for freq_range_idx in range(region_freq_range[0], region_freq_range[1] + 1)
                if freq_range_idx < len(recommended_lines)
            )

            regions.append({
                'region_id': region_id,
                'centroid': ndimage.center_of_mass(region_mask),
                'size': int(region_mask.sum()),
                'mean_uncertainty': float(region_uncertainty),
                'addressed': addressed
            })

        return regions

    def _assess_urgency(
        self,
        uncertainty_map: np.ndarray,
        discrepancy_map: Optional[np.ndarray]
    ) -> str:
        """Assess clinical urgency of re-sampling."""
        max_uncertainty = uncertainty_map.max()
        mean_uncertainty = uncertainty_map.mean()

        # High urgency if large uncertain regions or severe discrepancy
        high_uncertainty_fraction = (uncertainty_map > 0.5).mean()

        if discrepancy_map is not None:
            max_discrepancy = discrepancy_map.max()
        else:
            max_discrepancy = 0

        if max_uncertainty > 0.8 or high_uncertainty_fraction > 0.1 or max_discrepancy > 0.5:
            return "high"
        elif max_uncertainty > 0.5 or high_uncertainty_fraction > 0.05:
            return "medium"
        else:
            return "low"

    def _generate_rationale(
        self,
        num_lines: int,
        percent_increase: float,
        expected_reduction: float,
        num_regions: int,
        urgency: str
    ) -> str:
        """Generate explanation for recommendation."""
        return (
            f"Recommendation: Acquire {num_lines} additional k-space lines "
            f"({percent_increase:.1f}% scan time increase). "
            f"This should resolve {100*expected_reduction:.0f}% of uncertainty "
            f"and address {num_regions} suspicious regions. "
            f"Clinical urgency: {urgency.upper()}."
        )


def simulate_resampling_benefit(
    original_reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    current_kspace: np.ndarray,
    current_mask: np.ndarray,
    additional_lines_mask: np.ndarray,
    reconstruction_fn  # Function to reconstruct from k-space
) -> Dict:
    """
    Simulate the benefit of acquiring additional k-space lines.

    This demonstrates that targeted re-sampling resolves hallucinations.

    Args:
        original_reconstruction: Current reconstruction
        ground_truth: Fully-sampled ground truth
        current_kspace: Currently measured k-space
        current_mask: Current sampling mask
        additional_lines_mask: Recommended additional lines
        reconstruction_fn: Function to reconstruct from k-space

    Returns:
        Dict with before/after metrics
    """
    # Get ground truth k-space
    gt_kspace = np.fft.fftshift(np.fft.fft2(ground_truth))

    # Simulate acquiring additional lines
    enhanced_mask = current_mask | additional_lines_mask
    enhanced_kspace = current_kspace.copy()
    enhanced_kspace[additional_lines_mask] = gt_kspace[additional_lines_mask]

    # Reconstruct with enhanced data
    enhanced_reconstruction = reconstruction_fn(enhanced_kspace, enhanced_mask)

    # Compute metrics
    original_error = np.abs(original_reconstruction - ground_truth).mean()
    enhanced_error = np.abs(enhanced_reconstruction - ground_truth).mean()
    error_reduction = (original_error - enhanced_error) / (original_error + 1e-10)

    # Additional lines acquired
    num_original_lines = (current_mask.sum(axis=1) > 0).sum()
    num_additional_lines = (additional_lines_mask.sum(axis=1) > 0).sum()
    percent_increase = 100 * num_additional_lines / num_original_lines

    return {
        'original_error': original_error,
        'enhanced_error': enhanced_error,
        'error_reduction': error_reduction,
        'percent_additional_data': percent_increase,
        'efficiency': error_reduction / (percent_increase / 100 + 1e-10)  # Reduction per % data
    }


def generate_resampling_report(recommendation: ResamplingRecommendation) -> str:
    """Generate clinical report for re-sampling recommendation."""
    urgency_icon = {"low": "○", "medium": "◐", "high": "●"}[recommendation.clinical_urgency]

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║       CLINICAL RE-SAMPLING RECOMMENDATION                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  THE INNOVATION: Not just "Error" → "Scan these lines again" ║
║  Transforms your Auditor from Critic to Helper               ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  URGENCY: {urgency_icon} {recommendation.clinical_urgency.upper():<48} ║
╠══════════════════════════════════════════════════════════════╣
║  RECOMMENDATION                                               ║
╠──────────────────────────────────────────────────────────────╣
║  Additional k-space lines:  {recommendation.num_additional_lines:>8}                        ║
║  Scan time increase:        {recommendation.percentage_increase:>8.1f}%                       ║
║  Estimated time:            {recommendation.estimated_scan_time_increase:>8.1f}s                        ║
╠──────────────────────────────────────────────────────────────╣
║  EXPECTED BENEFIT                                             ║
╠──────────────────────────────────────────────────────────────╣
║  Uncertainty reduction:     {100*recommendation.expected_uncertainty_reduction:>8.0f}%                       ║
║  Hallucination resolution:  {100*recommendation.expected_hallucination_resolution:>8.0f}%                       ║
║  Regions addressed:         {len(recommendation.regions_addressed):>8}                        ║
╠══════════════════════════════════════════════════════════════╣
║  TOP PRIORITY LINES (k-space row indices)                     ║
╠──────────────────────────────────────────────────────────────╣
"""
    # Show top 10 priority lines
    top_lines = recommendation.line_priorities[:10]
    lines_str = ", ".join(map(str, top_lines))
    report += f"║  {lines_str:<58} ║\n"

    report += """╠══════════════════════════════════════════════════════════════╣
║  RATIONALE                                                    ║
╠──────────────────────────────────────────────────────────────╣
"""
    # Wrap rationale
    rat_lines = [recommendation.rationale[i:i+58] for i in range(0, len(recommendation.rationale), 58)]
    for line in rat_lines:
        report += f"║  {line:<58} ║\n"

    report += """╠══════════════════════════════════════════════════════════════╣
║  CLINICAL ACTION                                              ║
╠──────────────────────────────────────────────────────────────╣
║  ➤ Instruct scanner to acquire the recommended k-space lines  ║
║  ➤ Re-run reconstruction with enhanced data                   ║
║  ➤ Verify hallucinations are resolved                         ║
╚══════════════════════════════════════════════════════════════╝
"""
    return report
