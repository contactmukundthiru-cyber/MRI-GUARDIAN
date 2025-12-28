"""
Experiment 3: Robustness Study

HYPOTHESIS H3:
Guardian auditor performance is maintained across different
acceleration factors (2×, 4×, 6×, 8×) and noise levels.

PROTOCOL:
1. Evaluate reconstruction and detection at multiple accelerations
2. Optionally add k-space noise
3. Plot performance curves
4. Analyze degradation patterns
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

# Import MRI-GUARDIAN modules
from mri_guardian.data.fastmri_loader import SliceDataset, SimulatedMRIDataset
from mri_guardian.data.transforms import MRIDataTransform
from mri_guardian.data.kspace_ops import ifft2c, channels_to_complex, complex_abs
from mri_guardian.models.unet import UNet
from mri_guardian.models.guardian import GuardianModel, GuardianConfig
from mri_guardian.models.blackbox import HallucinationInjector, HallucinationConfig
from mri_guardian.metrics.image_quality import compute_all_image_metrics
from mri_guardian.metrics.detection import compute_auc, compute_roc_curve
from mri_guardian.visualization.comparison import create_robustness_figure


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


def add_kspace_noise(
    kspace: torch.Tensor,
    noise_level: float
) -> torch.Tensor:
    """
    Add complex Gaussian noise to k-space.

    Args:
        kspace: K-space tensor (B, 2, H, W)
        noise_level: Standard deviation relative to signal

    Returns:
        Noisy k-space
    """
    if noise_level <= 0:
        return kspace

    signal_std = kspace.std()
    noise_std = noise_level * signal_std

    noise = torch.randn_like(kspace) * noise_std
    return kspace + noise


def run_experiment(
    config: dict,
    use_simulated: bool = False,
    output_dir: str = "results/exp3_robustness"
):
    """
    Run Experiment 3: Robustness Study.

    Args:
        config: Configuration dictionary
        use_simulated: Use simulated data
        output_dir: Directory to save results
    """
    print("=" * 60)
    print("EXPERIMENT 3: ROBUSTNESS STUDY")
    print("=" * 60)

    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config['experiment'].get('device', 'cuda')
                          if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Acceleration factors to test
    accelerations = [2, 4, 6, 8]

    # Noise levels to test (relative to signal)
    noise_levels = [0.0, 0.01, 0.02, 0.05]

    # Number of samples per configuration
    num_samples = min(config['evaluation'].get('num_samples', 50), 50)
    print(f"Evaluating {num_samples} samples per configuration")

    # Load models
    print("\nLoading models...")

    # UNet baseline
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
        dc_mode=guardian_config_dict['dc_mode']
    )
    guardian = GuardianModel(guardian_cfg).to(device)

    # Load checkpoints if available
    unet_ckpt = Path("checkpoints/unet_best.pt")
    guardian_ckpt = Path("checkpoints/guardian_best.pt")

    if unet_ckpt.exists():
        unet.load_state_dict(torch.load(unet_ckpt, map_location=device)['model_state_dict'])
    if guardian_ckpt.exists():
        guardian.load_state_dict(torch.load(guardian_ckpt, map_location=device)['model_state_dict'])

    unet.eval()
    guardian.eval()

    # Hallucination injector
    halluc_injector = HallucinationInjector()

    # Results storage
    recon_results = {}  # {accel: {method: {metric: mean}}}
    detection_results = {}  # {accel: {method: auc}}
    noise_results = {}  # {noise: {method: {metric: mean}}}

    # Part 1: Acceleration robustness
    print("\n" + "=" * 60)
    print("PART 1: ACCELERATION FACTOR ROBUSTNESS")
    print("=" * 60)

    for accel in accelerations:
        print(f"\nAcceleration: {accel}×")

        # Create transform for this acceleration
        transform = MRIDataTransform(
            mask_type=config['undersampling']['mask_type'],
            acceleration=accel,
            center_fraction=config['undersampling']['center_fraction'],
            crop_size=tuple(config['data'].get('crop_size', [320, 320]))
        )

        # Load data
        if use_simulated:
            dataset = SimulatedMRIDataset(
                num_samples=num_samples,
                image_size=(320, 320),
                transform=transform,
                seed=seed + accel
            )
        else:
            dataset = SliceDataset(
                root=config['data']['root'],
                challenge=config['data']['challenge'],
                split=config['data']['val_split'],
                transform=transform,
                sample_rate=num_samples / 10000
            )

        # Metric accumulators
        metrics_accum = {
            'ZF-FFT': {'psnr': [], 'ssim': []},
            'UNet': {'psnr': [], 'ssim': []},
            'Guardian': {'psnr': [], 'ssim': []}
        }

        # Detection accumulators
        all_predictions = {'ZF': [], 'Guardian': []}
        all_gt = []

        with torch.no_grad():
            for i in tqdm(range(min(num_samples, len(dataset))), desc=f"{accel}× eval"):
                sample = dataset[i]

                masked_kspace = sample['masked_kspace'].unsqueeze(0).to(device)
                mask = sample['mask'].unsqueeze(0).to(device)
                target = sample['target'].unsqueeze(0).to(device)
                zf_input = sample['zf_recon'].unsqueeze(0).to(device)

                mean, std = sample['mean'], sample['std']
                target_denorm = target * std + mean

                # Reconstructions
                zf_recon = zero_filled_reconstruction(masked_kspace) * std + mean
                unet_recon = unet(zf_input) * std + mean
                guardian_result = guardian(masked_kspace, mask)
                guardian_recon = guardian_result['output'] * std + mean

                # Compute reconstruction metrics
                for method, recon in [('ZF-FFT', zf_recon), ('UNet', unet_recon), ('Guardian', guardian_recon)]:
                    result = compute_all_image_metrics(recon, target_denorm)
                    metrics_accum[method]['psnr'].append(result.psnr)
                    metrics_accum[method]['ssim'].append(result.ssim)

                # Detection test
                blackbox_recon = unet_recon
                halluc_result = halluc_injector(blackbox_recon / std, ["lesion"])
                hallucinated = halluc_result['output'] * std
                gt_mask = halluc_result['mask']

                if gt_mask.sum() > 0:
                    zf_det = torch.abs(hallucinated - zf_recon)
                    guardian_det = torch.abs(hallucinated - guardian_recon)

                    all_predictions['ZF'].append(zf_det.cpu().numpy().flatten())
                    all_predictions['Guardian'].append(guardian_det.cpu().numpy().flatten())
                    all_gt.append(gt_mask.cpu().numpy().flatten())

        # Store reconstruction results
        recon_results[accel] = {
            method: {m: float(np.mean(v)) for m, v in vals.items()}
            for method, vals in metrics_accum.items()
        }

        # Compute and store detection AUC
        if len(all_gt) > 0:
            gt_concat = np.concatenate(all_gt)
            detection_results[accel] = {}
            for method in ['ZF', 'Guardian']:
                pred_concat = np.concatenate(all_predictions[method])
                auc = compute_auc(
                    torch.from_numpy(pred_concat),
                    torch.from_numpy(gt_concat)
                )
                detection_results[accel][method] = float(auc)

        # Print results for this acceleration
        print(f"  Reconstruction metrics:")
        for method, vals in recon_results[accel].items():
            print(f"    {method}: PSNR={vals['psnr']:.2f}, SSIM={vals['ssim']:.4f}")
        if accel in detection_results:
            print(f"  Detection AUC:")
            for method, auc in detection_results[accel].items():
                print(f"    {method}: {auc:.4f}")

    # Part 2: Noise robustness (at fixed 4× acceleration)
    print("\n" + "=" * 60)
    print("PART 2: NOISE ROBUSTNESS (at 4× acceleration)")
    print("=" * 60)

    transform = MRIDataTransform(
        mask_type=config['undersampling']['mask_type'],
        acceleration=4,
        center_fraction=config['undersampling']['center_fraction'],
        crop_size=tuple(config['data'].get('crop_size', [320, 320]))
    )

    if use_simulated:
        dataset = SimulatedMRIDataset(
            num_samples=num_samples,
            image_size=(320, 320),
            transform=transform,
            seed=seed
        )
    else:
        dataset = SliceDataset(
            root=config['data']['root'],
            challenge=config['data']['challenge'],
            split=config['data']['val_split'],
            transform=transform,
            sample_rate=num_samples / 10000
        )

    for noise in noise_levels:
        print(f"\nNoise level: {noise}")

        metrics_accum = {
            'ZF-FFT': {'psnr': [], 'ssim': []},
            'UNet': {'psnr': [], 'ssim': []},
            'Guardian': {'psnr': [], 'ssim': []}
        }

        with torch.no_grad():
            for i in tqdm(range(min(num_samples, len(dataset))), desc=f"noise={noise}"):
                sample = dataset[i]

                masked_kspace = sample['masked_kspace'].unsqueeze(0).to(device)

                # Add noise
                masked_kspace_noisy = add_kspace_noise(masked_kspace, noise)

                mask = sample['mask'].unsqueeze(0).to(device)
                target = sample['target'].unsqueeze(0).to(device)

                mean, std = sample['mean'], sample['std']
                target_denorm = target * std + mean

                # Reconstructions with noisy k-space
                zf_recon = zero_filled_reconstruction(masked_kspace_noisy) * std + mean
                zf_input = (zero_filled_reconstruction(masked_kspace_noisy) - mean) / std
                unet_recon = unet(zf_input) * std + mean
                guardian_result = guardian(masked_kspace_noisy, mask)
                guardian_recon = guardian_result['output'] * std + mean

                for method, recon in [('ZF-FFT', zf_recon), ('UNet', unet_recon), ('Guardian', guardian_recon)]:
                    result = compute_all_image_metrics(recon, target_denorm)
                    metrics_accum[method]['psnr'].append(result.psnr)
                    metrics_accum[method]['ssim'].append(result.ssim)

        noise_results[noise] = {
            method: {m: float(np.mean(v)) for m, v in vals.items()}
            for method, vals in metrics_accum.items()
        }

        print(f"  Results:")
        for method, vals in noise_results[noise].items():
            print(f"    {method}: PSNR={vals['psnr']:.2f}, SSIM={vals['ssim']:.4f}")

    # Generate figures
    print("\nGenerating figures...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Acceleration robustness figure
    fig = create_robustness_figure(
        accelerations,
        recon_results,
        metric_names=['psnr', 'ssim'],
        save_path=output_dir / 'acceleration_robustness.png'
    )
    plt.close(fig)

    # Noise robustness figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ['psnr', 'ssim']):
        for method in ['ZF-FFT', 'UNet', 'Guardian']:
            values = [noise_results[n][method][metric] for n in noise_levels]
            ax.plot(noise_levels, values, marker='o', linewidth=2, label=method)

        ax.set_xlabel('Noise Level (σ/signal)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'noise_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Detection robustness figure
    if detection_results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        for method in ['ZF', 'Guardian']:
            values = [detection_results[a].get(method, 0) for a in accelerations]
            ax.plot(accelerations, values, marker='o', linewidth=2, label=f'{method} Detector')

        ax.set_xlabel('Acceleration Factor')
        ax.set_ylabel('Detection AUC')
        ax.set_title('Hallucination Detection AUC vs Acceleration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(accelerations)

        plt.tight_layout()
        plt.savefig(output_dir / 'detection_robustness.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Save results
    results = {
        'reconstruction_vs_acceleration': recon_results,
        'detection_vs_acceleration': detection_results,
        'reconstruction_vs_noise': noise_results,
        'config': {
            'accelerations': accelerations,
            'noise_levels': noise_levels,
            'num_samples': num_samples
        }
    }

    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Robustness Study')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--simulated', action='store_true',
                        help='Use simulated data')
    parser.add_argument('--output', type=str, default='results/exp3_robustness',
                        help='Output directory')
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config, use_simulated=args.simulated, output_dir=args.output)


if __name__ == '__main__':
    main()
