"""
Experiment 1: Reconstruction Quality Comparison

HYPOTHESIS H1:
Physics-guided generative reconstruction (Guardian) achieves higher
PSNR/SSIM than zero-filled FFT and UNet baselines.

PROTOCOL:
1. Load validation data with 4× acceleration
2. Reconstruct with each method: ZF-FFT, UNet, Guardian
3. Compute metrics: PSNR, SSIM, NRMSE, HFEN
4. Statistical significance testing
5. Generate comparison figures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse

# Import MRI-GUARDIAN modules
from mri_guardian.data.fastmri_loader import SliceDataset, SimulatedMRIDataset
from mri_guardian.data.transforms import MRIDataTransform
from mri_guardian.data.kspace_ops import ifft2c, channels_to_complex, complex_abs
from mri_guardian.models.unet import UNet
from mri_guardian.models.guardian import GuardianModel, GuardianConfig
from mri_guardian.metrics.image_quality import compute_all_image_metrics, MetricAggregator
from mri_guardian.metrics.statistical import paired_ttest, compute_summary_statistics, format_results_for_paper
from mri_guardian.visualization.plotting import plot_comparison, save_figure
from mri_guardian.visualization.comparison import create_comparison_figure


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def zero_filled_reconstruction(masked_kspace: torch.Tensor) -> torch.Tensor:
    """
    Zero-filled (ZF) reconstruction baseline.

    Simply inverse FFT the masked k-space.
    This is the simplest possible reconstruction.
    """
    # Convert 2-channel to complex
    kspace_complex = channels_to_complex(masked_kspace)
    # Inverse FFT
    image_complex = ifft2c(kspace_complex)
    # Take magnitude
    image_mag = complex_abs(image_complex)
    return image_mag.unsqueeze(1)


def run_experiment(
    config: dict,
    use_simulated: bool = False,
    output_dir: str = "results/exp1_reconstruction"
):
    """
    Run Experiment 1: Reconstruction Comparison.

    Args:
        config: Configuration dictionary
        use_simulated: Use simulated data (for testing without fastMRI)
        output_dir: Directory to save results
    """
    print("=" * 60)
    print("EXPERIMENT 1: RECONSTRUCTION QUALITY COMPARISON")
    print("=" * 60)

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
        print("Using SIMULATED data (no fastMRI required)")
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

    # UNet baseline
    unet_config = config['model']['unet']
    unet = UNet(
        in_channels=unet_config['in_channels'],
        out_channels=unet_config['out_channels'],
        base_channels=unet_config['base_channels'],
        num_levels=unet_config['num_levels'],
        use_residual=unet_config.get('use_residual', True),
        residual_learning=unet_config.get('residual_learning', True)
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
        learnable_dc=guardian_config_dict['learnable_dc'],
        use_attention=guardian_config_dict['use_attention']
    )
    guardian = GuardianModel(guardian_cfg).to(device)

    # Load pretrained weights if available
    unet_checkpoint = Path("checkpoints/unet_best.pt")
    guardian_checkpoint = Path("checkpoints/guardian_best.pt")

    if unet_checkpoint.exists():
        unet.load_state_dict(torch.load(unet_checkpoint, map_location=device)['model_state_dict'])
        print("Loaded UNet checkpoint")
    else:
        print("WARNING: No UNet checkpoint found, using random weights")

    if guardian_checkpoint.exists():
        guardian.load_state_dict(torch.load(guardian_checkpoint, map_location=device)['model_state_dict'])
        print("Loaded Guardian checkpoint")
    else:
        print("WARNING: No Guardian checkpoint found, using random weights")

    # Set to eval mode
    unet.eval()
    guardian.eval()

    # Metric aggregators
    metrics = {
        'ZF-FFT': MetricAggregator(),
        'UNet': MetricAggregator(),
        'Guardian': MetricAggregator()
    }

    # Store individual metrics for statistical tests
    raw_metrics = {method: {'psnr': [], 'ssim': [], 'nrmse': [], 'hfen': []}
                   for method in metrics.keys()}

    # Example images for visualization
    example_idx = [0, num_samples // 4, num_samples // 2, 3 * num_samples // 4]
    example_images = {i: {} for i in example_idx}

    # Evaluation loop
    print("\nRunning evaluation...")
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[i]

            masked_kspace = sample['masked_kspace'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            target = sample['target'].unsqueeze(0).to(device)
            zf_input = sample['zf_recon'].unsqueeze(0).to(device)

            # Denormalize target for metric computation
            mean = sample['mean']
            std = sample['std']
            target_denorm = target * std + mean

            # 1. Zero-filled reconstruction
            zf_recon = zero_filled_reconstruction(masked_kspace)
            zf_recon_denorm = zf_recon * std + mean

            # 2. UNet reconstruction
            unet_recon = unet(zf_input)
            unet_recon_denorm = unet_recon * std + mean

            # 3. Guardian reconstruction
            guardian_result = guardian(masked_kspace, mask)
            guardian_recon = guardian_result['output']
            guardian_recon_denorm = guardian_recon * std + mean

            # Compute metrics
            reconstructions = {
                'ZF-FFT': zf_recon_denorm,
                'UNet': unet_recon_denorm,
                'Guardian': guardian_recon_denorm
            }

            for method, recon in reconstructions.items():
                result = compute_all_image_metrics(recon, target_denorm)
                metrics[method].add(recon, target_denorm)

                raw_metrics[method]['psnr'].append(result.psnr)
                raw_metrics[method]['ssim'].append(result.ssim)
                raw_metrics[method]['nrmse'].append(result.nrmse)
                raw_metrics[method]['hfen'].append(result.hfen)

            # Save example images
            if i in example_idx:
                example_images[i] = {
                    'Ground Truth': target_denorm[0, 0].cpu(),
                    'ZF-FFT': zf_recon_denorm[0, 0].cpu(),
                    'UNet': unet_recon_denorm[0, 0].cpu(),
                    'Guardian': guardian_recon_denorm[0, 0].cpu()
                }

    # Compute summary statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    summaries = {}
    for method in metrics.keys():
        summary = metrics[method].get_summary()
        summaries[method] = summary
        print(f"\n{method}:")
        for metric, stats in summary.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Statistical tests: Compare Guardian vs baselines
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    for metric in ['psnr', 'ssim']:
        print(f"\n{metric.upper()}:")

        # Guardian vs ZF-FFT
        result = paired_ttest(
            np.array(raw_metrics['Guardian'][metric]),
            np.array(raw_metrics['ZF-FFT'][metric])
        )
        sig = "*" if result.significant else ""
        print(f"  Guardian vs ZF-FFT: p={result.p_value:.4f}{sig}, effect size d={result.effect_size:.2f}")

        # Guardian vs UNet
        result = paired_ttest(
            np.array(raw_metrics['Guardian'][metric]),
            np.array(raw_metrics['UNet'][metric])
        )
        sig = "*" if result.significant else ""
        print(f"  Guardian vs UNet: p={result.p_value:.4f}{sig}, effect size d={result.effect_size:.2f}")

    # Generate figures
    print("\nGenerating figures...")

    # Example comparison figures
    for i, imgs in example_images.items():
        if imgs:
            fig = plot_comparison(
                list(imgs.values()),
                list(imgs.keys()),
                main_title=f'Sample {i}: Reconstruction Comparison ({config["undersampling"]["acceleration"]}× acceleration)',
                save_path=output_dir / f'comparison_sample_{i}.png'
            )
            plt.close(fig)

    # Create publication figure with metrics
    if example_images[0]:
        imgs = example_images[0]
        fig = create_comparison_figure(
            imgs['Ground Truth'],
            {k: v for k, v in imgs.items() if k != 'Ground Truth'},
            error_maps=True,
            metrics={
                'ZF-FFT': {'psnr': summaries['ZF-FFT']['psnr']['mean'],
                          'ssim': summaries['ZF-FFT']['ssim']['mean']},
                'UNet': {'psnr': summaries['UNet']['psnr']['mean'],
                        'ssim': summaries['UNet']['ssim']['mean']},
                'Guardian': {'psnr': summaries['Guardian']['psnr']['mean'],
                            'ssim': summaries['Guardian']['ssim']['mean']},
            },
            save_path=output_dir / 'publication_figure.png'
        )
        plt.close(fig)

    # Save results to file
    results = {
        'config': {
            'acceleration': config['undersampling']['acceleration'],
            'num_samples': num_samples,
            'seed': seed
        },
        'metrics': {method: {m: {'mean': s['mean'], 'std': s['std']}
                             for m, s in summary.items()}
                   for method, summary in summaries.items()},
        'raw_metrics': raw_metrics
    }

    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Reconstruction Comparison')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--simulated', action='store_true',
                        help='Use simulated data instead of fastMRI')
    parser.add_argument('--output', type=str, default='results/exp1_reconstruction',
                        help='Output directory')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Import matplotlib here to avoid issues
    import matplotlib
    matplotlib.use('Agg')
    global plt
    import matplotlib.pyplot as plt

    # Run experiment
    run_experiment(config, use_simulated=args.simulated, output_dir=args.output)


if __name__ == '__main__':
    main()
