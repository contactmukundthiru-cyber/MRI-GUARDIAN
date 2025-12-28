"""
Training Script for MRI-GUARDIAN Models

This script trains the Guardian model and UNet baseline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm

# Import MRI-GUARDIAN modules
from mri_guardian.data.fastmri_loader import SliceDataset, SimulatedMRIDataset, create_data_loaders
from mri_guardian.data.transforms import MRIDataTransform, AugmentationTransform, ComposeTransforms
from mri_guardian.data.kspace_ops import ifft2c, channels_to_complex, complex_abs
from mri_guardian.models.unet import UNet
from mri_guardian.models.guardian import GuardianModel, GuardianConfig, GuardianLoss
from mri_guardian.metrics.image_quality import compute_psnr, compute_ssim


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer based on config."""
    opt_name = config['training'].get('optimizer', 'adam').lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0)

    if opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def create_scheduler(optimizer: optim.Optimizer, config: dict, num_epochs: int):
    """Create learning rate scheduler."""
    scheduler_name = config['training'].get('scheduler', 'cosine')

    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.5)
    elif scheduler_name == 'none':
        return None
    else:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


def train_guardian(
    config: dict,
    use_simulated: bool = False,
    output_dir: str = "checkpoints"
):
    """
    Train the Guardian model.

    Args:
        config: Configuration dictionary
        use_simulated: Use simulated data
        output_dir: Directory to save checkpoints
    """
    print("=" * 60)
    print("TRAINING GUARDIAN MODEL")
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

    # Create transforms
    train_transform = ComposeTransforms([
        MRIDataTransform(
            mask_type=config['undersampling']['mask_type'],
            acceleration=config['undersampling']['acceleration'],
            center_fraction=config['undersampling']['center_fraction'],
            crop_size=tuple(config['data'].get('crop_size', [320, 320])),
            use_seed=False  # Random masks during training
        ),
        AugmentationTransform(flip_prob=0.3, rotate_prob=0.2)
    ])

    val_transform = MRIDataTransform(
        mask_type=config['undersampling']['mask_type'],
        acceleration=config['undersampling']['acceleration'],
        center_fraction=config['undersampling']['center_fraction'],
        crop_size=tuple(config['data'].get('crop_size', [320, 320])),
        use_seed=True  # Deterministic masks for validation
    )

    # Create datasets
    print("\nLoading data...")
    if use_simulated:
        print("Using SIMULATED data")
        train_dataset = SimulatedMRIDataset(
            num_samples=1000,
            image_size=(320, 320),
            transform=train_transform
        )
        val_dataset = SimulatedMRIDataset(
            num_samples=100,
            image_size=(320, 320),
            transform=val_transform,
            seed=123
        )
    else:
        train_dataset = SliceDataset(
            root=config['data']['root'],
            challenge=config['data']['challenge'],
            split=config['data']['train_split'],
            transform=train_transform,
            sample_rate=config['data'].get('sample_rate', 1.0)
        )
        val_dataset = SliceDataset(
            root=config['data']['root'],
            challenge=config['data']['challenge'],
            split=config['data']['val_split'],
            transform=val_transform,
            sample_rate=config['data'].get('sample_rate', 1.0)
        )

    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['experiment'].get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print("\nCreating Guardian model...")
    guardian_cfg = config['model']['guardian']
    model_config = GuardianConfig(
        num_iterations=guardian_cfg['num_iterations'],
        base_channels=guardian_cfg['base_channels'],
        num_levels=guardian_cfg['num_levels'],
        use_kspace_net=guardian_cfg['use_kspace_net'],
        use_image_net=guardian_cfg['use_image_net'],
        use_score_net=guardian_cfg['use_score_net'],
        dc_mode=guardian_cfg['dc_mode'],
        learnable_dc=guardian_cfg['learnable_dc'],
        dropout=guardian_cfg.get('dropout', 0.0),
        use_attention=guardian_cfg.get('use_attention', True),
        intermediate_supervision=True
    )

    model = GuardianModel(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    loss_config = config['training']['loss']
    criterion = GuardianLoss(
        lambda_l1=loss_config['l1_weight'],
        lambda_ssim=loss_config['ssim_weight'],
        lambda_dc=loss_config['dc_weight'],
        lambda_intermediate=loss_config['intermediate_weight']
    )

    # Optimizer and scheduler
    optimizer = create_optimizer(model, config)
    num_epochs = config['training']['epochs']
    scheduler = create_scheduler(optimizer, config, num_epochs)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_psnr': [],
        'val_ssim': []
    }

    best_val_psnr = 0
    grad_clip = config['training'].get('gradient_clip', 1.0)

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            masked_kspace = batch['masked_kspace'].to(device)
            mask = batch['mask'].to(device)
            target = batch['target'].to(device)

            # Forward
            result = model(masked_kspace, mask, return_intermediates=True)
            output = result['output']
            kspace_final = result['kspace_final']
            intermediates = result.get('intermediates', None)

            # Loss
            losses = criterion(output, target, kspace_final, masked_kspace, mask, intermediates)
            loss = losses['total']

            # Backward
            optimizer.zero_grad()
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []
        val_psnrs = []
        val_ssims = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                masked_kspace = batch['masked_kspace'].to(device)
                mask = batch['mask'].to(device)
                target = batch['target'].to(device)

                result = model(masked_kspace, mask)
                output = result['output']
                kspace_final = result['kspace_final']

                losses = criterion(output, target, kspace_final, masked_kspace, mask)
                val_losses.append(losses['total'].item())

                # Compute metrics
                for b in range(output.shape[0]):
                    psnr = compute_psnr(output[b], target[b])
                    ssim = compute_ssim(output[b], target[b])
                    val_psnrs.append(psnr)
                    val_ssims.append(ssim)

        avg_val_loss = np.mean(val_losses)
        avg_val_psnr = np.mean(val_psnrs)
        avg_val_ssim = np.mean(val_ssims)

        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(avg_val_psnr)
        history['val_ssim'].append(avg_val_ssim)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val PSNR: {avg_val_psnr:.2f}")
        print(f"  Val SSIM: {avg_val_ssim:.4f}")

        # Save best model
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim
            }
            torch.save(checkpoint, output_dir / 'guardian_best.pt')
            print(f"  Saved best model (PSNR: {avg_val_psnr:.2f})")

        # Save periodic checkpoint
        if (epoch + 1) % config['experiment'].get('save_frequency', 10) == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'history': history
            }
            torch.save(checkpoint, output_dir / f'guardian_epoch_{epoch+1}.pt')

    # Save final model
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config,
        'history': history
    }
    torch.save(checkpoint, output_dir / 'guardian_final.pt')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best Val PSNR: {best_val_psnr:.2f}")
    print(f"Models saved to {output_dir}")
    print("=" * 60)

    return history


def train_unet_baseline(
    config: dict,
    use_simulated: bool = False,
    output_dir: str = "checkpoints"
):
    """
    Train the UNet baseline model.

    Args:
        config: Configuration dictionary
        use_simulated: Use simulated data
        output_dir: Directory to save checkpoints
    """
    print("=" * 60)
    print("TRAINING UNET BASELINE")
    print("=" * 60)

    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config['experiment'].get('device', 'cuda')
                          if torch.cuda.is_available() else 'cpu')

    # Set seed
    torch.manual_seed(config['experiment'].get('seed', 42))

    # Create transforms
    train_transform = ComposeTransforms([
        MRIDataTransform(
            mask_type=config['undersampling']['mask_type'],
            acceleration=config['undersampling']['acceleration'],
            center_fraction=config['undersampling']['center_fraction'],
            crop_size=tuple(config['data'].get('crop_size', [320, 320])),
            use_seed=False
        ),
        AugmentationTransform(flip_prob=0.3, rotate_prob=0.2)
    ])

    val_transform = MRIDataTransform(
        mask_type=config['undersampling']['mask_type'],
        acceleration=config['undersampling']['acceleration'],
        center_fraction=config['undersampling']['center_fraction'],
        crop_size=tuple(config['data'].get('crop_size', [320, 320])),
        use_seed=True
    )

    # Create datasets
    if use_simulated:
        train_dataset = SimulatedMRIDataset(num_samples=1000, transform=train_transform)
        val_dataset = SimulatedMRIDataset(num_samples=100, transform=val_transform, seed=123)
    else:
        train_dataset = SliceDataset(
            root=config['data']['root'],
            challenge=config['data']['challenge'],
            split=config['data']['train_split'],
            transform=train_transform,
            sample_rate=config['data'].get('sample_rate', 1.0)
        )
        val_dataset = SliceDataset(
            root=config['data']['root'],
            challenge=config['data']['challenge'],
            split=config['data']['val_split'],
            transform=val_transform,
            sample_rate=config['data'].get('sample_rate', 1.0)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['experiment'].get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['experiment'].get('num_workers', 4)
    )

    # Create model
    unet_cfg = config['model']['unet']
    model = UNet(
        in_channels=unet_cfg['in_channels'],
        out_channels=unet_cfg['out_channels'],
        base_channels=unet_cfg['base_channels'],
        num_levels=unet_cfg['num_levels'],
        use_residual=unet_cfg.get('use_residual', True),
        residual_learning=unet_cfg.get('residual_learning', True)
    ).to(device)

    print(f"UNet parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, config['training']['epochs'])

    best_val_psnr = 0

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            zf_input = batch['zf_recon'].to(device)
            target = batch['target'].to(device)

            output = model(zf_input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_psnrs = []

        with torch.no_grad():
            for batch in val_loader:
                zf_input = batch['zf_recon'].to(device)
                target = batch['target'].to(device)

                output = model(zf_input)

                for b in range(output.shape[0]):
                    psnr = compute_psnr(output[b], target[b])
                    val_psnrs.append(psnr)

        avg_val_psnr = np.mean(val_psnrs)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f}, Val PSNR={avg_val_psnr:.2f}")

        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_psnr': avg_val_psnr
            }, output_dir / 'unet_best.pt')

    print(f"\nTraining complete. Best PSNR: {best_val_psnr:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train MRI-GUARDIAN models')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='guardian',
                        choices=['guardian', 'unet', 'both'],
                        help='Which model to train')
    parser.add_argument('--simulated', action='store_true',
                        help='Use simulated data')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model in ['guardian', 'both']:
        train_guardian(config, use_simulated=args.simulated, output_dir=args.output)

    if args.model in ['unet', 'both']:
        train_unet_baseline(config, use_simulated=args.simulated, output_dir=args.output)


if __name__ == '__main__':
    main()
