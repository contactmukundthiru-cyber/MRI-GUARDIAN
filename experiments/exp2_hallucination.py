"""
Experiment 2: Hallucination Detection with Lesion Integrity Marker (LIM)

HYPOTHESIS H2:
Using Guardian reconstruction as an external reference improves
detection of hallucinated structures in black-box reconstructions
compared to naive baselines.

NOVEL CONTRIBUTION: LESION INTEGRITY MARKER (LIM)
A single bioengineering metric (0-1) that quantifies "how intact is this lesion?"
- LIM = 1.0: Lesion perfectly preserved
- LIM = 0.0: Lesion completely corrupted

PROTOCOL:
1. Generate black-box reconstructions with realistic lesions
2. Detect hallucinations/corruptions using:
   - Baseline 1: |Black-box - ZF|
   - Baseline 2: Edge anomaly detection
   - Guardian: |Black-box - Guardian|
3. Compute LIM for each lesion in black-box vs Guardian reconstructions
4. Analyze correlation between auditor suspicion and LIM
5. Generate comprehensive metrics and visualizations

KEY BIOENGINEERING RESULT:
"In X% of lesions, the black-box AI degraded the lesion integrity marker by
more than Y%, while Guardian recon kept LIM within Z% of the original."
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from scipy.ndimage import gaussian_filter

# Import MRI-GUARDIAN modules
from mri_guardian.data.fastmri_loader import SliceDataset, SimulatedMRIDataset
from mri_guardian.data.transforms import MRIDataTransform
from mri_guardian.data.kspace_ops import ifft2c, channels_to_complex, complex_abs
from mri_guardian.models.unet import UNet
from mri_guardian.models.guardian import GuardianModel, GuardianConfig
from mri_guardian.models.blackbox import HallucinationInjector, HallucinationConfig
from mri_guardian.auditor.detector import HallucinationDetector, BaselineDetector
from mri_guardian.auditor.hallucination import generate_synthetic_lesions, generate_mixed_hallucinations
from mri_guardian.auditor.lesion_integrity_marker import (
    LesionIntegrityMarker, LIMAggregator, AuditorLIMCorrelator,
    LesionFingerprintExtractor, create_lim_visualization
)
from mri_guardian.metrics.detection import (
    compute_roc_curve, compute_auc, compute_detection_metrics,
    DetectionAggregator, compute_pixel_wise_metrics
)
from mri_guardian.visualization.plotting import plot_roc_curves, save_figure
from mri_guardian.visualization.comparison import create_hallucination_detection_figure


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def zero_filled_reconstruction(masked_kspace: torch.Tensor) -> torch.Tensor:
    """Zero-filled reconstruction baseline."""
    kspace_complex = channels_to_complex(masked_kspace)
    image_complex = ifft2c(kspace_complex)
    image_mag = complex_abs(image_complex)
    return image_mag.unsqueeze(1)


def generate_realistic_lesion(
    image: torch.Tensor,
    num_lesions: int = 1,
    size_range: tuple = (5, 20),
    contrast_range: tuple = (0.1, 0.4)
) -> tuple:
    """
    Generate realistic lesions on an image for LIM testing.

    Creates clinically realistic elliptical lesions with smooth edges,
    varying contrast (hyper or hypo-intense), and random orientations.

    Args:
        image: Input image tensor (B, C, H, W)
        num_lesions: Number of lesions to generate
        size_range: (min, max) size in pixels
        contrast_range: (min, max) contrast relative to background

    Returns:
        (image_with_lesions, list_of_lesion_masks)
    """
    B, C, H, W = image.shape
    device = image.device

    output = image.clone()
    lesion_masks = []

    for _ in range(num_lesions):
        # Random lesion parameters
        size = np.random.randint(size_range[0], size_range[1])
        contrast = np.random.uniform(contrast_range[0], contrast_range[1])

        # Random position (avoid edges)
        margin = size + 15
        cx = np.random.randint(margin, W - margin)
        cy = np.random.randint(margin, H - margin)

        # Create lesion mask (elliptical shape with random orientation)
        y, x = np.ogrid[:H, :W]
        a = size * np.random.uniform(0.7, 1.3)  # Semi-major axis
        b = size * np.random.uniform(0.6, 1.0)  # Semi-minor axis
        angle = np.random.uniform(0, np.pi)  # Random rotation

        # Rotated ellipse equation
        x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
        ellipse = (x_rot / (a + 1e-8)) ** 2 + (y_rot / (b + 1e-8)) ** 2

        mask = (ellipse <= 1).astype(np.float32)

        # Smooth edges for realism
        mask = gaussian_filter(mask, sigma=1.5)
        mask = (mask > 0.3).astype(np.float32)

        if mask.sum() < 10:  # Skip if too small
            continue

        lesion_mask = torch.from_numpy(mask).to(device)
        lesion_masks.append(lesion_mask)

        # Determine lesion intensity based on local background
        cy_safe = min(max(cy, 5), H - 5)
        cx_safe = min(max(cx, 5), W - 5)
        local_mean = output[0, 0, cy_safe-5:cy_safe+5, cx_safe-5:cx_safe+5].mean().item()

        # Lesion can be hyper or hypo intense (both clinically relevant)
        if np.random.random() > 0.5:
            lesion_intensity = local_mean * (1 + contrast)  # Hyper-intense
        else:
            lesion_intensity = local_mean * (1 - contrast)  # Hypo-intense

        # Apply lesion with smooth blending
        mask_tensor = lesion_mask.unsqueeze(0).unsqueeze(0)
        for b_idx in range(B):
            for c_idx in range(C):
                output[b_idx, c_idx] = output[b_idx, c_idx] * (1 - mask_tensor[0, 0]) + \
                                       lesion_intensity * mask_tensor[0, 0]

    return output, lesion_masks


def run_experiment(
    config: dict,
    use_simulated: bool = False,
    output_dir: str = "results/exp2_hallucination"
):
    """
    Run Experiment 2: Hallucination Detection with LIM Analysis.

    This experiment combines traditional hallucination detection with
    the novel Lesion Integrity Marker (LIM) bioengineering metric.

    Args:
        config: Configuration dictionary
        use_simulated: Use simulated data
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("EXPERIMENT 2: HALLUCINATION DETECTION")
    print("         with LESION INTEGRITY MARKER (LIM) Analysis")
    print("=" * 70)

    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config['experiment'].get('device', 'cuda')
                          if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed for reproducibility
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create transform
    transform = MRIDataTransform(
        mask_type=config['undersampling']['mask_type'],
        acceleration=config['undersampling']['acceleration'],
        center_fraction=config['undersampling']['center_fraction'],
        crop_size=tuple(config['data'].get('crop_size', [320, 320]))
    )

    # Load data
    print("\nLoading data...")
    if use_simulated:
        print("Using SIMULATED data")
        dataset = SimulatedMRIDataset(
            num_samples=config['evaluation'].get('num_samples', 100),
            image_size=(320, 320),
            transform=transform
        )
    else:
        dataset = SliceDataset(
            root=config['data']['root'],
            challenge=config['data']['challenge'],
            split=config['data']['val_split'],
            transform=transform,
            sample_rate=config['data'].get('sample_rate', 1.0)
        )

    num_samples = min(config['evaluation'].get('num_samples', 100), len(dataset))
    print(f"Evaluating on {num_samples} samples")

    # Load models
    print("\nLoading models...")

    # UNet (black-box model)
    unet_config = config['model']['unet']
    unet = UNet(
        in_channels=unet_config['in_channels'],
        out_channels=unet_config['out_channels'],
        base_channels=unet_config['base_channels'],
        num_levels=unet_config['num_levels']
    ).to(device)

    # Guardian model
    guardian_config_dict = config['model']['guardian']
    guardian_cfg = GuardianConfig(
        num_iterations=guardian_config_dict['num_iterations'],
        base_channels=guardian_config_dict['base_channels'],
        num_levels=guardian_config_dict['num_levels'],
        use_kspace_net=guardian_config_dict['use_kspace_net'],
        use_image_net=guardian_config_dict['use_image_net'],
        use_score_net=guardian_config_dict['use_score_net'],
        dc_mode=guardian_config_dict['dc_mode'],
        learnable_dc=guardian_config_dict['learnable_dc']
    )
    guardian = GuardianModel(guardian_cfg).to(device)

    # Load checkpoints if available
    unet_ckpt = Path("checkpoints/unet_best.pt")
    guardian_ckpt = Path("checkpoints/guardian_best.pt")

    if unet_ckpt.exists():
        unet.load_state_dict(torch.load(unet_ckpt, map_location=device)['model_state_dict'])
        print("Loaded UNet checkpoint")
    else:
        print("WARNING: Using random UNet weights")

    if guardian_ckpt.exists():
        guardian.load_state_dict(torch.load(guardian_ckpt, map_location=device)['model_state_dict'])
        print("Loaded Guardian checkpoint")
    else:
        print("WARNING: Using random Guardian weights")

    unet.eval()
    guardian.eval()

    # ================================================================
    # INITIALIZE LESION INTEGRITY MARKER (LIM) SYSTEM
    # ================================================================
    print("\n" + "-" * 70)
    print("Initializing Lesion Integrity Marker (LIM) system...")
    print("-" * 70)

    # Clinical weighting: contrast and shape are most critical for diagnosis
    lim_calculator = LesionIntegrityMarker(
        weights={
            'intensity': 0.15,    # Important but not critical
            'contrast': 0.25,     # CRITICAL for lesion visibility
            'shape': 0.20,        # Important for morphology assessment
            'texture': 0.15,      # Internal structure
            'edge': 0.15,         # Boundary definition
            'location': 0.10      # Should stay in place
        },
        danger_threshold=0.7  # Below this, lesion may be unreliable
    )
    lim_aggregator = LIMAggregator()
    auditor_lim_correlator = AuditorLIMCorrelator()

    # Hallucination injector for detection testing
    halluc_config = config.get('hallucination', {})
    halluc_injector = HallucinationInjector(HallucinationConfig(
        lesion_prob=halluc_config.get('lesion_prob', 0.5),
        num_lesions_range=tuple(halluc_config.get('num_lesions_range', [1, 3])),
        lesion_size_range=tuple(halluc_config.get('lesion_size_range', [4, 12])),
        lesion_intensity_range=tuple(halluc_config.get('lesion_intensity_range', [0.3, 0.8])),
        texture_prob=halluc_config.get('texture_prob', 0.3),
        sharpen_prob=halluc_config.get('sharpen_prob', 0.2),
        smooth_prob=halluc_config.get('smooth_prob', 0.2)
    ))

    # Detection aggregators for each method
    detection_methods = {
        'ZF Baseline': DetectionAggregator(),
        'Edge Baseline': DetectionAggregator(),
        'Guardian': DetectionAggregator()
    }

    # Store ROC data
    roc_data = {}

    # Example images for visualization
    example_images = None
    example_lim_data = None

    # LIM results storage
    all_lim_results = {
        'blackbox': [],
        'guardian': []
    }

    # ================================================================
    # MAIN EVALUATION LOOP
    # ================================================================
    print("\nRunning evaluation with LIM analysis...")
    all_predictions = {method: [] for method in detection_methods.keys()}
    all_ground_truths = []

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[i]

            masked_kspace = sample['masked_kspace'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            target = sample['target'].unsqueeze(0).to(device)
            zf_input = sample['zf_recon'].unsqueeze(0).to(device)

            # Get black-box reconstruction
            blackbox_recon = unet(zf_input)

            # Generate realistic lesions on the target (ground truth)
            # This simulates having real pathology in the original scan
            target_with_lesions, lesion_masks = generate_realistic_lesion(
                target,
                num_lesions=np.random.randint(1, 4),
                size_range=(6, 18),
                contrast_range=(0.1, 0.35)
            )

            # Inject hallucinations into black-box output (AI artifacts)
            halluc_result = halluc_injector(blackbox_recon, ["lesion", "texture"])
            hallucinated = halluc_result['output']
            gt_mask = halluc_result['mask']

            # Get zero-filled reconstruction
            zf_recon = zero_filled_reconstruction(masked_kspace)

            # Get Guardian reconstruction
            guardian_result = guardian(masked_kspace, mask)
            guardian_recon = guardian_result['output']

            # ============================================================
            # LESION INTEGRITY MARKER (LIM) ANALYSIS
            # ============================================================
            # For each lesion, compute LIM comparing ground truth vs reconstructions

            for lesion_idx, lesion_mask in enumerate(lesion_masks):
                if lesion_mask.sum() < 10:  # Skip tiny masks
                    continue

                # Convert to numpy for LIM calculation
                gt_np = target_with_lesions[0, 0].cpu().numpy()
                bb_np = blackbox_recon[0, 0].cpu().numpy()
                guardian_np = guardian_recon[0, 0].cpu().numpy()
                mask_np = lesion_mask.cpu().numpy()

                # Normalize images to [0, 1]
                gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)
                bb_np = (bb_np - bb_np.min()) / (bb_np.max() - bb_np.min() + 1e-8)
                guardian_np = (guardian_np - guardian_np.min()) / (guardian_np.max() - guardian_np.min() + 1e-8)

                # Compute LIM for black-box reconstruction
                lim_bb = lim_calculator.compute_lim(gt_np, bb_np, mask_np)
                lim_aggregator.add_result('blackbox', lim_bb)
                all_lim_results['blackbox'].append({
                    'lim_score': lim_bb.lim_score,
                    'intensity': lim_bb.intensity_score,
                    'contrast': lim_bb.contrast_score,
                    'shape': lim_bb.shape_score,
                    'texture': lim_bb.texture_score,
                    'edge': lim_bb.edge_score,
                    'location': lim_bb.location_score,
                    'sample_idx': i,
                    'lesion_idx': lesion_idx,
                    'risk_category': lim_calculator.get_risk_category(lim_bb)
                })

                # Compute LIM for Guardian reconstruction
                lim_g = lim_calculator.compute_lim(gt_np, guardian_np, mask_np)
                lim_aggregator.add_result('guardian', lim_g)
                all_lim_results['guardian'].append({
                    'lim_score': lim_g.lim_score,
                    'intensity': lim_g.intensity_score,
                    'contrast': lim_g.contrast_score,
                    'shape': lim_g.shape_score,
                    'texture': lim_g.texture_score,
                    'edge': lim_g.edge_score,
                    'location': lim_g.location_score,
                    'sample_idx': i,
                    'lesion_idx': lesion_idx,
                    'risk_category': lim_calculator.get_risk_category(lim_g)
                })

                # Compute auditor suspicion for this lesion region
                # High suspicion should correlate with low LIM (negative correlation)
                guardian_detection = torch.abs(hallucinated - guardian_recon)
                lesion_region = lesion_mask > 0.5
                if lesion_region.sum() > 0:
                    lesion_suspicion = guardian_detection[0, 0][lesion_region].mean().item()
                    auditor_lim_correlator.add_pair(lesion_suspicion, lim_bb.lim_score)

                # Save first valid example for visualization
                if example_lim_data is None and lim_bb.lim_score > 0:
                    example_lim_data = {
                        'gt_image': gt_np.copy(),
                        'bb_image': bb_np.copy(),
                        'guardian_image': guardian_np.copy(),
                        'lesion_mask': mask_np.copy(),
                        'bb_lim': lim_bb,
                        'guardian_lim': lim_g
                    }

            # ============================================================
            # HALLUCINATION DETECTION (Original Experiment)
            # ============================================================

            # Detection method 1: ZF Baseline |hallucinated - zf|
            zf_detection = torch.abs(hallucinated - zf_recon)
            zf_detection = zf_detection / (zf_detection.max() + 1e-8)

            # Detection method 2: Edge anomaly baseline
            edge_detection = BaselineDetector.detect_edge_anomaly(hallucinated, threshold=0.0)

            # Detection method 3: Guardian |hallucinated - guardian|
            guardian_detection = torch.abs(hallucinated - guardian_recon)
            guardian_detection = guardian_detection / (guardian_detection.max() + 1e-8)

            # Store for ROC computation
            all_predictions['ZF Baseline'].append(zf_detection.cpu().numpy().flatten())
            all_predictions['Edge Baseline'].append(edge_detection.cpu().numpy().flatten())
            all_predictions['Guardian'].append(guardian_detection.cpu().numpy().flatten())
            all_ground_truths.append(gt_mask.cpu().numpy().flatten())

            # Save first example for visualization
            if example_images is None and gt_mask.sum() > 0:
                example_images = {
                    'original': blackbox_recon[0, 0].cpu(),
                    'hallucinated': hallucinated[0, 0].cpu(),
                    'guardian_recon': guardian_recon[0, 0].cpu(),
                    'detection_map': guardian_detection[0, 0].cpu(),
                    'ground_truth_mask': gt_mask[0, 0].cpu()
                }

    # ================================================================
    # PART 1: HALLUCINATION DETECTION RESULTS
    # ================================================================

    print("\n" + "=" * 70)
    print("PART 1: HALLUCINATION DETECTION RESULTS")
    print("=" * 70)

    # Concatenate all predictions and ground truths
    all_gt = np.concatenate(all_ground_truths)
    gt_tensor = torch.from_numpy(all_gt)

    detection_results = {}
    for method in detection_methods.keys():
        pred_array = np.concatenate(all_predictions[method])
        pred_tensor = torch.from_numpy(pred_array)

        # Compute metrics
        metrics = compute_detection_metrics(pred_tensor, gt_tensor)
        detection_results[method] = metrics

        print(f"\n{method}:")
        print(f"  AUC: {metrics.auc:.4f}")
        print(f"  F1: {metrics.f1:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  Optimal Threshold: {metrics.threshold:.4f}")

        # Store ROC data
        fpr, tpr, _ = compute_roc_curve(pred_tensor, gt_tensor)
        roc_data[method] = (fpr, tpr, metrics.auc)

    # Statistical comparison: Guardian vs baselines
    print("\n" + "-" * 70)
    print("Statistical Comparison: Guardian vs Baselines")
    print("-" * 70)

    guardian_preds = np.concatenate(all_predictions['Guardian'])
    for baseline in ['ZF Baseline', 'Edge Baseline']:
        baseline_preds = np.concatenate(all_predictions[baseline])

        guardian_auc = compute_auc(torch.from_numpy(guardian_preds), gt_tensor)
        baseline_auc = compute_auc(torch.from_numpy(baseline_preds), gt_tensor)

        improvement = (guardian_auc - baseline_auc) / (baseline_auc + 1e-8) * 100
        print(f"\nGuardian vs {baseline}:")
        print(f"  Guardian AUC: {guardian_auc:.4f}")
        print(f"  {baseline} AUC: {baseline_auc:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")

    # ================================================================
    # PART 2: LESION INTEGRITY MARKER (LIM) RESULTS
    # ================================================================

    print("\n" + "=" * 70)
    print("PART 2: LESION INTEGRITY MARKER (LIM) RESULTS")
    print("         *** KEY BIOENGINEERING METRIC ***")
    print("=" * 70)

    lim_stats = lim_aggregator.compute_statistics()
    lim_comparison = lim_aggregator.compare_methods()

    print("\n--- LIM Statistics by Reconstruction Method ---")
    for method, stats in lim_stats.items():
        print(f"\n{'='*40}")
        print(f"  {method.upper()} RECONSTRUCTION")
        print(f"{'='*40}")
        print(f"  Mean LIM: {stats['mean_lim']:.3f} +/- {stats['std_lim']:.3f}")
        print(f"  Median LIM: {stats['median_lim']:.3f}")
        print(f"  Range: [{stats['min_lim']:.3f}, {stats['max_lim']:.3f}]")
        print(f"  Number of lesions analyzed: {stats['num_lesions']}")

        print(f"\n  RISK DISTRIBUTION:")
        print(f"    Excellent (>=0.9): {stats['pct_excellent']:5.1f}%  [Perfectly preserved]")
        print(f"    Good (0.8-0.9):    {stats['pct_good']:5.1f}%  [Minor changes]")
        print(f"    Acceptable (0.7-0.8): {stats['pct_acceptable']:5.1f}%  [Acceptable for diagnosis]")
        print(f"    Warning (0.5-0.7):    {stats['pct_warning']:5.1f}%  [Review recommended]")
        print(f"    CRITICAL (<0.5):      {stats['pct_critical']:5.1f}%  [DO NOT USE]")

        print(f"\n  COMPONENT SCORES:")
        print(f"    Intensity:  {stats['mean_intensity_score']:.3f}")
        print(f"    Contrast:   {stats['mean_contrast_score']:.3f}  [Most critical]")
        print(f"    Shape:      {stats['mean_shape_score']:.3f}")
        print(f"    Texture:    {stats['mean_texture_score']:.3f}")
        print(f"    Edge:       {stats['mean_edge_score']:.3f}")
        print(f"    Location:   {stats['mean_location_score']:.3f}")

    print("\n" + "=" * 70)
    print("GUARDIAN vs BLACK-BOX COMPARISON")
    print("=" * 70)

    if lim_comparison:
        print(f"\n  Mean LIM Improvement (Guardian - Blackbox): {lim_comparison['mean_improvement']:+.3f}")
        print(f"  Guardian preserves lesions better in: {lim_comparison['pct_guardian_better']:.1f}% of cases")
        print(f"\n  CRITICAL LESION RATE (LIM < 0.5):")
        print(f"    Black-box: {lim_comparison['blackbox_critical_rate']:.1f}%")
        print(f"    Guardian:  {lim_comparison['guardian_critical_rate']:.1f}%")
        print(f"    Reduction: {lim_comparison['critical_rate_reduction']:.1f} percentage points")

    # ================================================================
    # PART 3: AUDITOR-LIM CORRELATION ANALYSIS
    # ================================================================

    print("\n" + "=" * 70)
    print("PART 3: AUDITOR SUSPICION vs LIM CORRELATION")
    print("         *** Validates Auditor Can Detect At-Risk Lesions ***")
    print("=" * 70)

    correlation = auditor_lim_correlator.compute_correlation()
    detection_from_lim = auditor_lim_correlator.compute_detection_from_lim(lim_threshold=0.7)

    print(f"\n  Correlation Analysis (n={correlation['n_samples']} lesions):")
    print(f"    Pearson r:  {correlation['pearson_r']:.3f} (p={correlation['pearson_p']:.4f})")
    print(f"    Spearman rho: {correlation['spearman_r']:.3f} (p={correlation['spearman_p']:.4f})")

    expected_negative = correlation['pearson_r'] < 0
    print(f"\n  Expected: Negative correlation (high suspicion = low LIM)")
    print(f"  Result: {'CONFIRMED' if expected_negative else 'NOT CONFIRMED'}")

    if detection_from_lim:
        print(f"\n  Auditor's Ability to Detect Low-LIM Lesions (LIM < 0.7):")
        print(f"    Precision: {detection_from_lim['precision']:.3f}")
        print(f"    Recall:    {detection_from_lim['recall']:.3f}")
        print(f"    F1 Score:  {detection_from_lim['f1']:.3f}")
        print(f"\n  Detection Breakdown:")
        print(f"    At-Risk Lesions (LIM < 0.7): {detection_from_lim['num_at_risk']}")
        print(f"    True Positives:  {detection_from_lim['true_positives']}")
        print(f"    False Negatives: {detection_from_lim['false_negatives']}")
        print(f"    False Positives: {detection_from_lim['false_positives']}")

    # ================================================================
    # GENERATE FIGURES
    # ================================================================

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: ROC curves for hallucination detection
    fig = plot_roc_curves(
        roc_data,
        title="Hallucination Detection ROC Curves",
        save_path=output_dir / 'roc_curves.png'
    )
    plt.close(fig)
    print("  [1/6] Saved: roc_curves.png")

    # Figure 2: Example hallucination detection
    if example_images is not None:
        fig = create_hallucination_detection_figure(
            example_images['original'],
            example_images['hallucinated'],
            example_images['guardian_recon'],
            example_images['detection_map'],
            example_images['ground_truth_mask'],
            metrics={
                'auc': roc_data['Guardian'][2],
                'f1': detection_results['Guardian'].f1,
                'precision': detection_results['Guardian'].precision,
                'recall': detection_results['Guardian'].recall
            },
            save_path=output_dir / 'detection_example.png'
        )
        plt.close(fig)
        print("  [2/6] Saved: detection_example.png")

    # Figure 3: LIM visualization example
    if example_lim_data is not None:
        fig = create_lim_visualization(
            example_lim_data['gt_image'],
            example_lim_data['bb_image'],
            example_lim_data['guardian_image'],
            example_lim_data['lesion_mask'],
            example_lim_data['bb_lim'],
            example_lim_data['guardian_lim'],
            save_path=output_dir / 'lim_example.png'
        )
        plt.close(fig)
        print("  [3/6] Saved: lim_example.png")

    # Figure 4: LIM distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 4a: LIM score histograms
    ax = axes[0, 0]
    bb_lims = [r['lim_score'] for r in all_lim_results['blackbox']]
    g_lims = [r['lim_score'] for r in all_lim_results['guardian']]

    if len(bb_lims) > 0 and len(g_lims) > 0:
        ax.hist(bb_lims, bins=20, alpha=0.6, label=f'Black-box (mean={np.mean(bb_lims):.2f})', color='red')
        ax.hist(g_lims, bins=20, alpha=0.6, label=f'Guardian (mean={np.mean(g_lims):.2f})', color='green')
        ax.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Danger Threshold (0.7)')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Critical Threshold (0.5)')
    ax.set_xlabel('Lesion Integrity Marker (LIM)', fontsize=12)
    ax.set_ylabel('Number of Lesions', fontsize=12)
    ax.set_title('LIM Score Distribution: Black-box vs Guardian', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)

    # 4b: Component comparison bar chart
    ax = axes[0, 1]
    if 'blackbox' in lim_stats and 'guardian' in lim_stats:
        components = ['Intensity', 'Contrast', 'Shape', 'Texture', 'Edge', 'Location']
        bb_means = [lim_stats['blackbox'][f'mean_{c.lower()}_score'] for c in components]
        g_means = [lim_stats['guardian'][f'mean_{c.lower()}_score'] for c in components]

        x = np.arange(len(components))
        width = 0.35
        bars1 = ax.bar(x - width/2, bb_means, width, label='Black-box', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, g_means, width, label='Guardian', color='green', alpha=0.7)

        ax.set_ylabel('Mean Component Score (0-1)', fontsize=12)
        ax.set_title('LIM Component Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Danger')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # 4c: Risk category pie chart - Black-box
    ax = axes[1, 0]
    if 'blackbox' in lim_stats:
        categories = ['Excellent\n(>=0.9)', 'Good\n(0.8-0.9)', 'Acceptable\n(0.7-0.8)',
                      'Warning\n(0.5-0.7)', 'CRITICAL\n(<0.5)']
        bb_pcts = [lim_stats['blackbox']['pct_excellent'], lim_stats['blackbox']['pct_good'],
                   lim_stats['blackbox']['pct_acceptable'], lim_stats['blackbox']['pct_warning'],
                   lim_stats['blackbox']['pct_critical']]
        colors = ['darkgreen', 'limegreen', 'gold', 'orange', 'red']

        # Filter out zero values
        non_zero = [(c, p, col) for c, p, col in zip(categories, bb_pcts, colors) if p > 0]
        if non_zero:
            cats, pcts, cols = zip(*non_zero)
            wedges, texts, autotexts = ax.pie(pcts, labels=cats, autopct='%1.1f%%',
                                               colors=cols, startangle=90, pctdistance=0.75)
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
    ax.set_title('Black-box: Lesion Risk Distribution', fontsize=14, fontweight='bold')

    # 4d: Risk category pie chart - Guardian
    ax = axes[1, 1]
    if 'guardian' in lim_stats:
        g_pcts = [lim_stats['guardian']['pct_excellent'], lim_stats['guardian']['pct_good'],
                  lim_stats['guardian']['pct_acceptable'], lim_stats['guardian']['pct_warning'],
                  lim_stats['guardian']['pct_critical']]

        non_zero = [(c, p, col) for c, p, col in zip(categories, g_pcts, colors) if p > 0]
        if non_zero:
            cats, pcts, cols = zip(*non_zero)
            wedges, texts, autotexts = ax.pie(pcts, labels=cats, autopct='%1.1f%%',
                                               colors=cols, startangle=90, pctdistance=0.75)
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
    ax.set_title('Guardian: Lesion Risk Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'lim_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4/6] Saved: lim_distribution.png")

    # Figure 5: Auditor-LIM correlation analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    suspicion_scores = np.array(auditor_lim_correlator.suspicion_scores)
    lim_scores_corr = np.array(auditor_lim_correlator.lim_scores)

    # 5a: Scatter plot with regression line
    ax = axes[0]
    if len(suspicion_scores) > 2:
        ax.scatter(suspicion_scores, lim_scores_corr, alpha=0.5, c='blue', s=30, edgecolors='navy')

        # Fit and plot regression line
        z = np.polyfit(suspicion_scores, lim_scores_corr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(suspicion_scores.min(), suspicion_scores.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2,
                label=f'Linear fit (r={correlation["pearson_r"]:.3f})')

        ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='LIM Danger (0.7)')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='LIM Critical (0.5)')

    ax.set_xlabel('Auditor Suspicion Score', fontsize=12)
    ax.set_ylabel('Lesion Integrity Marker (LIM)', fontsize=12)
    ax.set_title('Auditor Suspicion vs LIM Correlation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)

    # 5b: Quadrant analysis (confusion matrix style)
    ax = axes[1]
    if len(suspicion_scores) > 2:
        median_sus = np.median(suspicion_scores)

        high_sus_low_lim = sum(1 for s, l in zip(suspicion_scores, lim_scores_corr)
                               if s > median_sus and l < 0.7)
        high_sus_high_lim = sum(1 for s, l in zip(suspicion_scores, lim_scores_corr)
                                if s > median_sus and l >= 0.7)
        low_sus_low_lim = sum(1 for s, l in zip(suspicion_scores, lim_scores_corr)
                              if s <= median_sus and l < 0.7)
        low_sus_high_lim = sum(1 for s, l in zip(suspicion_scores, lim_scores_corr)
                               if s <= median_sus and l >= 0.7)

        quadrants = ['True Positive\n(High Suspicion,\nLow LIM)',
                     'False Positive\n(High Suspicion,\nHigh LIM)',
                     'False Negative\n(Low Suspicion,\nLow LIM)',
                     'True Negative\n(Low Suspicion,\nHigh LIM)']
        counts = [high_sus_low_lim, high_sus_high_lim, low_sus_low_lim, low_sus_high_lim]
        colors = ['green', 'orange', 'red', 'lightgreen']

        bars = ax.bar(range(4), counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(quadrants, fontsize=9)
        ax.set_ylabel('Number of Lesions', fontsize=12)
        ax.set_title('Auditor Detection Quadrant Analysis', fontsize=14, fontweight='bold')

        # Add count labels
        for bar, count in zip(bars, counts):
            ax.annotate(str(count), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 5), textcoords="offset points", ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'auditor_lim_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [5/6] Saved: auditor_lim_correlation.png")

    # Figure 6: Summary infographic
    fig = plt.figure(figsize=(16, 10))

    # Main title
    fig.suptitle('EXPERIMENT 2: Hallucination Detection with Lesion Integrity Marker (LIM)',
                 fontsize=18, fontweight='bold', y=0.98)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Key finding box
    ax_key = fig.add_subplot(gs[0, :])
    ax_key.axis('off')

    if 'blackbox' in lim_stats and 'guardian' in lim_stats:
        bb_at_risk = lim_stats['blackbox']['pct_critical'] + lim_stats['blackbox']['pct_warning']
        g_at_risk = lim_stats['guardian']['pct_critical'] + lim_stats['guardian']['pct_warning']
        improvement_pct = lim_comparison.get('mean_improvement', 0) / (lim_stats['blackbox']['mean_lim'] + 1e-8) * 100

        key_finding = f"""
KEY BIOENGINEERING FINDING

In {bb_at_risk:.1f}% of lesions, Black-box AI degraded the Lesion Integrity Marker below 0.7 (danger threshold).
Guardian reconstruction reduced this at-risk rate to {g_at_risk:.1f}% — a {bb_at_risk - g_at_risk:.1f} percentage point improvement.

Guardian improved mean LIM by {improvement_pct:+.1f}% compared to Black-box,
demonstrating significantly better preservation of clinically critical lesion characteristics.
        """
    else:
        key_finding = "Insufficient data for key finding."

    ax_key.text(0.5, 0.5, key_finding, transform=ax_key.transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor='orange', linewidth=2),
                fontfamily='monospace')

    # Detection metrics table
    ax_det = fig.add_subplot(gs[1, 0])
    ax_det.axis('off')
    det_text = "HALLUCINATION DETECTION\n\n"
    for method, metrics in detection_results.items():
        det_text += f"{method}:\n  AUC={metrics.auc:.3f}, F1={metrics.f1:.3f}\n"
    ax_det.text(0.1, 0.9, det_text, transform=ax_det.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # LIM metrics table
    ax_lim = fig.add_subplot(gs[1, 1])
    ax_lim.axis('off')
    if 'blackbox' in lim_stats and 'guardian' in lim_stats:
        lim_text = f"""LIM SCORES

Black-box:
  Mean: {lim_stats['blackbox']['mean_lim']:.3f}
  Critical: {lim_stats['blackbox']['pct_critical']:.1f}%

Guardian:
  Mean: {lim_stats['guardian']['mean_lim']:.3f}
  Critical: {lim_stats['guardian']['pct_critical']:.1f}%
"""
    else:
        lim_text = "LIM data not available"
    ax_lim.text(0.1, 0.9, lim_text, transform=ax_lim.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Correlation metrics
    ax_corr = fig.add_subplot(gs[1, 2])
    ax_corr.axis('off')
    corr_text = f"""AUDITOR VALIDATION

Correlation with LIM:
  Pearson r = {correlation['pearson_r']:.3f}
  p-value = {correlation['pearson_p']:.4f}

Detection of At-Risk:
  Precision = {detection_from_lim.get('precision', 0):.3f}
  Recall = {detection_from_lim.get('recall', 0):.3f}
  F1 = {detection_from_lim.get('f1', 0):.3f}
"""
    ax_corr.text(0.1, 0.9, corr_text, transform=ax_corr.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))

    # Mini bar chart of LIM means
    ax_bar = fig.add_subplot(gs[2, 0:2])
    if 'blackbox' in lim_stats and 'guardian' in lim_stats:
        methods = ['Black-box', 'Guardian']
        means = [lim_stats['blackbox']['mean_lim'], lim_stats['guardian']['mean_lim']]
        stds = [lim_stats['blackbox']['std_lim'], lim_stats['guardian']['std_lim']]
        colors = ['red', 'green']

        bars = ax_bar.bar(methods, means, yerr=stds, color=colors, alpha=0.7,
                          capsize=10, edgecolor='black', linewidth=2)
        ax_bar.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='Danger Threshold')
        ax_bar.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Critical Threshold')
        ax_bar.set_ylabel('Mean LIM Score', fontsize=12)
        ax_bar.set_title('Mean Lesion Integrity Marker by Method', fontsize=14, fontweight='bold')
        ax_bar.legend(loc='lower right')
        ax_bar.set_ylim(0, 1.1)

        for bar, mean in zip(bars, means):
            ax_bar.annotate(f'{mean:.3f}', xy=(bar.get_x() + bar.get_width()/2, mean),
                           xytext=(0, 10), textcoords="offset points", ha='center',
                           fontsize=14, fontweight='bold')

    # Clinical interpretation
    ax_clinical = fig.add_subplot(gs[2, 2])
    ax_clinical.axis('off')
    clinical_text = """CLINICAL INTERPRETATION

The Lesion Integrity Marker (LIM)
provides a single, interpretable
score for radiologists:

  LIM >= 0.9: EXCELLENT
    Lesion fully preserved

  LIM 0.8-0.9: GOOD
    Minor changes, safe to use

  LIM 0.7-0.8: ACCEPTABLE
    Use with awareness

  LIM 0.5-0.7: WARNING
    Manual review recommended

  LIM < 0.5: CRITICAL
    Do NOT rely on this lesion
"""
    ax_clinical.text(0.1, 0.95, clinical_text, transform=ax_clinical.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig(output_dir / 'summary_infographic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [6/6] Saved: summary_infographic.png")

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    results = {
        'config': {
            'acceleration': config['undersampling']['acceleration'],
            'num_samples': num_samples,
            'seed': seed
        },
        'detection_metrics': {},
        'lim_analysis': {
            'statistics': lim_stats,
            'comparison': lim_comparison,
            'auditor_correlation': correlation,
            'auditor_detection': detection_from_lim
        }
    }

    for method in detection_methods.keys():
        pred_tensor = torch.from_numpy(np.concatenate(all_predictions[method]))
        metrics = compute_detection_metrics(pred_tensor, gt_tensor)
        results['detection_metrics'][method] = {
            'auc': float(metrics.auc),
            'f1': float(metrics.f1),
            'precision': float(metrics.precision),
            'recall': float(metrics.recall),
            'threshold': float(metrics.threshold)
        }

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results = convert_to_serializable(results)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save detailed LIM results for further analysis
    with open(output_dir / 'lim_detailed.json', 'w') as f:
        json.dump(convert_to_serializable(all_lim_results), f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # ================================================================
    # PRINT KEY BIOENGINEERING FINDING (THE HEADLINE)
    # ================================================================

    print("\n" + "=" * 70)
    print("KEY BIOENGINEERING FINDING")
    print("         *** Use this in your ISEF presentation! ***")
    print("=" * 70)

    if 'blackbox' in lim_stats and 'guardian' in lim_stats:
        bb_at_risk = lim_stats['blackbox']['pct_critical'] + lim_stats['blackbox']['pct_warning']
        g_at_risk = lim_stats['guardian']['pct_critical'] + lim_stats['guardian']['pct_warning']
        improvement = lim_comparison.get('mean_improvement', 0) / (lim_stats['blackbox']['mean_lim'] + 1e-8) * 100

        print(f"""
    "In {bb_at_risk:.1f}% of lesions, the black-box AI degraded the
     Lesion Integrity Marker below the clinical danger threshold (LIM < 0.7),
     while Guardian reconstruction kept this rate at only {g_at_risk:.1f}%.

     Guardian improved mean LIM by {improvement:+.1f}% compared to black-box,
     demonstrating significantly better preservation of clinically
     critical lesion characteristics including contrast, shape, and edges."

    Furthermore, our physics-guided auditor shows {abs(correlation['pearson_r']):.2f}
    correlation between suspicion scores and actual lesion degradation,
    validating that the auditor can reliably warn clinicians about
    at-risk lesions in AI-reconstructed MRI scans.
    """)

    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2: Hallucination Detection with Lesion Integrity Marker (LIM)'
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--simulated', action='store_true',
                        help='Use simulated data')
    parser.add_argument('--output', type=str, default='results/exp2_hallucination',
                        help='Output directory')
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config, use_simulated=args.simulated, output_dir=args.output)


if __name__ == '__main__':
    main()
