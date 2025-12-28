"""
Multi-GPU Training Script for MRI-GUARDIAN
===========================================

Optimized for Lambda Labs 4x A100 80GB instances.

Features:
- Distributed Data Parallel (DDP) for multi-GPU training
- Mixed precision (AMP) for 2x speedup
- Gradient checkpointing for memory efficiency
- Automatic checkpointing with crash recovery
- Wandb logging for monitoring

Usage:
    # Single GPU (testing)
    python cloud/train_multigpu.py --modality mri --data-dir /data/medical_imaging

    # Multi-GPU (production)
    torchrun --nproc_per_node=4 cloud/train_multigpu.py --modality mri --data-dir /data/medical_imaging

    # All modalities sequentially
    ./cloud/train_all.sh
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_main(msg):
    """Print only from main process."""
    if is_main_process():
        print(msg)


class CheckpointManager:
    """Manages checkpoints for crash recovery."""

    def __init__(self, checkpoint_dir: str, modality: str, keep_last: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.modality = modality
        self.keep_last = keep_last
        self.state_file = self.checkpoint_dir / f"{modality}_state.json"

    def save(self, model, optimizer, scheduler, scaler, epoch, global_step, metrics):
        """Save checkpoint."""
        if not is_main_process():
            return

        # Save model state (handle DDP wrapper)
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Save epoch checkpoint
        path = self.checkpoint_dir / f"{self.modality}_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        print_main(f"💾 Saved checkpoint: {path}")

        # Save latest
        latest_path = self.checkpoint_dir / f"{self.modality}_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best if applicable
        if metrics.get('is_best', False):
            best_path = self.checkpoint_dir / f"{self.modality}_best.pt"
            torch.save(checkpoint, best_path)
            print_main(f"⭐ New best model!")

        # Update state file
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'best_psnr': metrics.get('best_psnr', 0),
            'last_checkpoint': str(path)
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Cleanup old checkpoints
        self._cleanup()

    def load(self, checkpoint_path: Optional[str] = None):
        """Load checkpoint."""
        if checkpoint_path:
            path = Path(checkpoint_path)
        else:
            path = self.checkpoint_dir / f"{self.modality}_latest.pt"

        if path.exists():
            print_main(f"📂 Loading checkpoint: {path}")
            return torch.load(path, map_location='cuda')
        return None

    def _cleanup(self):
        """Keep only last N checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{self.modality}_epoch_*.pt"),
            key=lambda x: x.stat().st_mtime
        )
        for ckpt in checkpoints[:-self.keep_last]:
            ckpt.unlink()


def create_mri_model(config: Dict) -> nn.Module:
    """Create MRI reconstruction model."""
    from mri_guardian.models.guardian import GuardianReconstructor

    return GuardianReconstructor(
        in_channels=config.get('in_channels', 2),
        out_channels=config.get('out_channels', 2),
        num_iterations=config.get('num_iterations', 8),
        num_features=config.get('num_features', 64)
    )


def create_ct_model(config: Dict) -> nn.Module:
    """Create CT reconstruction model."""
    # Use similar architecture adapted for CT
    from mri_guardian.models.unet import UNet

    return UNet(
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512]
    )


def create_xray_model(config: Dict) -> nn.Module:
    """Create X-ray enhancement model."""
    from mri_guardian.models.unet import UNet

    return UNet(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256]
    )


def create_mri_dataloader(data_dir: str, config: Dict, split: str, world_size: int, rank: int):
    """Create MRI dataloader with distributed sampler."""
    from mri_guardian.data.fastmri_loader import FastMRIDataset

    dataset = FastMRIDataset(
        root_dir=data_dir,
        split=split,
        acceleration=config.get('acceleration', 4),
        center_fraction=config.get('center_fraction', 0.08)
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(split == 'train'))

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True,
        drop_last=(split == 'train')
    )

    return loader, sampler


def create_ct_dataloader(data_dir: str, config: Dict, split: str, world_size: int, rank: int):
    """Create CT dataloader."""
    from mri_guardian.data.ct_loader import CTDataset

    dataset = CTDataset(
        root_dir=data_dir,
        split=split
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(split == 'train'))

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return loader, sampler


def create_xray_dataloader(data_dir: str, config: Dict, split: str, world_size: int, rank: int):
    """Create X-ray dataloader."""
    from mri_guardian.data.xray_loader import XrayDataset

    dataset = XrayDataset(
        root_dir=data_dir,
        split=split
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(split == 'train'))

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return loader, sampler


def train_epoch(model, dataloader, sampler, optimizer, scaler, device, epoch, config, modality):
    """Train for one epoch."""
    model.train()
    sampler.set_epoch(epoch)

    total_loss = 0
    total_psnr = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())

    for batch in pbar:
        # Get data based on modality
        if modality == 'mri':
            inputs = batch['undersampled'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
        else:
            inputs = batch['input'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            mask = None

        optimizer.zero_grad()

        # Mixed precision forward
        with autocast():
            if modality == 'mri' and mask is not None:
                outputs = model(inputs, mask)
            else:
                outputs = model(inputs)

            loss = nn.functional.l1_loss(outputs, targets)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            mse = nn.functional.mse_loss(outputs, targets)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

        total_loss += loss.item()
        total_psnr += psnr.item()
        n_batches += 1

        pbar.set_postfix({'loss': f"{total_loss/n_batches:.4f}", 'psnr': f"{total_psnr/n_batches:.2f}"})

    return {
        'loss': total_loss / n_batches,
        'psnr': total_psnr / n_batches
    }


def validate(model, dataloader, device, modality):
    """Validate model."""
    model.eval()

    total_loss = 0
    total_psnr = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", disable=not is_main_process()):
            if modality == 'mri':
                inputs = batch['undersampled'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                mask = batch.get('mask')
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)
            else:
                inputs = batch['input'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                mask = None

            with autocast():
                if modality == 'mri' and mask is not None:
                    outputs = model(inputs, mask)
                else:
                    outputs = model(inputs)

                loss = nn.functional.l1_loss(outputs, targets)

            mse = nn.functional.mse_loss(outputs, targets)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

            total_loss += loss.item()
            total_psnr += psnr.item()
            n_batches += 1

    # Reduce across processes
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, total_psnr, n_batches], device=device)
        dist.all_reduce(metrics)
        total_loss, total_psnr, n_batches = metrics.tolist()

    return {
        'loss': total_loss / n_batches,
        'psnr': total_psnr / n_batches
    }


def train_modality(modality: str, data_dir: str, config: Dict, checkpoint_dir: str, resume: bool = False):
    """Train a specific modality."""

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device('cuda', local_rank)

    print_main(f"\n{'='*60}")
    print_main(f"Training {modality.upper()}")
    print_main(f"{'='*60}")
    print_main(f"World size: {world_size}")
    print_main(f"Device: {device}")

    # Create model
    if modality == 'mri':
        model = create_mri_model(config)
        train_loader, train_sampler = create_mri_dataloader(
            f"{data_dir}/mri/brain_multicoil_train", config, 'train', world_size, rank
        )
        val_loader, _ = create_mri_dataloader(
            f"{data_dir}/mri/brain_multicoil_val", config, 'val', world_size, rank
        )
    elif modality == 'ct':
        model = create_ct_model(config)
        train_loader, train_sampler = create_ct_dataloader(
            f"{data_dir}/ct", config, 'train', world_size, rank
        )
        val_loader, _ = create_ct_dataloader(
            f"{data_dir}/ct", config, 'val', world_size, rank
        )
    elif modality == 'xray':
        model = create_xray_model(config)
        train_loader, train_sampler = create_xray_dataloader(
            f"{data_dir}/xray", config, 'train', world_size, rank
        )
        val_loader, _ = create_xray_dataloader(
            f"{data_dir}/xray", config, 'val', world_size, rank
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    model = model.to(device)
    print_main(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler()

    # Checkpoint manager
    ckpt_manager = CheckpointManager(checkpoint_dir, modality)

    # Resume if requested
    start_epoch = 0
    best_psnr = 0

    if resume:
        checkpoint = ckpt_manager.load()
        if checkpoint:
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint['metrics'].get('best_psnr', 0)
            print_main(f"Resumed from epoch {checkpoint['epoch']}")

    # Initialize wandb
    if WANDB_AVAILABLE and is_main_process():
        wandb.init(
            project="mri-guardian",
            name=f"{modality}-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=config
        )

    # Training loop
    print_main(f"\nStarting training for {config['epochs']} epochs...")

    for epoch in range(start_epoch, config['epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, train_sampler, optimizer, scaler,
            device, epoch, config, modality
        )

        # Validate
        val_metrics = validate(model, val_loader, device, modality)

        # Update scheduler
        scheduler.step()

        # Check if best
        is_best = val_metrics['psnr'] > best_psnr
        if is_best:
            best_psnr = val_metrics['psnr']

        # Save checkpoint
        metrics = {
            'train_loss': train_metrics['loss'],
            'train_psnr': train_metrics['psnr'],
            'val_loss': val_metrics['loss'],
            'val_psnr': val_metrics['psnr'],
            'best_psnr': best_psnr,
            'is_best': is_best,
            'lr': scheduler.get_last_lr()[0]
        }

        ckpt_manager.save(model, optimizer, scheduler, scaler, epoch, 0, metrics)

        # Log
        print_main(f"\nEpoch {epoch}:")
        print_main(f"  Train - Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f} dB")
        print_main(f"  Val   - Loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.2f} dB")
        print_main(f"  Best PSNR: {best_psnr:.2f} dB")

        if WANDB_AVAILABLE and is_main_process():
            wandb.log(metrics, step=epoch)

    # Cleanup
    if WANDB_AVAILABLE and is_main_process():
        wandb.finish()

    cleanup_distributed()

    print_main(f"\n✅ Training complete for {modality.upper()}")
    print_main(f"   Best PSNR: {best_psnr:.2f} dB")

    return {'modality': modality, 'best_psnr': best_psnr}


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU training for MRI-GUARDIAN')
    parser.add_argument('--modality', type=str, required=True, choices=['mri', 'ct', 'xray'])
    parser.add_argument('--data-dir', type=str, default='/data/medical_imaging')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)  # Per GPU
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_workers': 4,
        'acceleration': 4,
        'center_fraction': 0.08,
        'in_channels': 2,
        'out_channels': 2,
        'num_iterations': 8,
        'num_features': 64
    }

    result = train_modality(
        modality=args.modality,
        data_dir=args.data_dir,
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume
    )

    print(f"\n{json.dumps(result, indent=2)}")


if __name__ == '__main__':
    main()
