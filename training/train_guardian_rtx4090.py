"""
MRI-GUARDIAN Training Script - Optimized for RTX 4090
======================================================

Features:
- Automatic checkpointing every epoch
- Crash recovery (resume from any checkpoint)
- Mixed precision training (2x speedup)
- Gradient accumulation for large effective batch sizes
- Progress logging to tensorboard and wandb
- Graceful shutdown on Ctrl+C

Usage:
    # Start fresh training
    python training/train_guardian_rtx4090.py --data-dir /path/to/fastmri

    # Resume from crash
    python training/train_guardian_rtx4090.py --data-dir /path/to/fastmri --resume

    # Resume from specific checkpoint
    python training/train_guardian_rtx4090.py --data-dir /path/to/fastmri --resume --checkpoint checkpoints/guardian_epoch_15.pt
"""

import os
import sys
import argparse
import signal
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import numpy as np

# Training state for crash recovery
class TrainingState:
    """Manages training state for crash recovery."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "training_state.json"

        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_psnr = 0.0
        self.training_started = None
        self.last_checkpoint = None

    def save_state(self):
        """Save training state to JSON."""
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'best_psnr': self.best_psnr,
            'training_started': self.training_started,
            'last_checkpoint': self.last_checkpoint,
            'last_saved': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> bool:
        """Load training state from JSON. Returns True if state exists."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.epoch = state.get('epoch', 0)
            self.global_step = state.get('global_step', 0)
            self.best_loss = state.get('best_loss', float('inf'))
            self.best_psnr = state.get('best_psnr', 0.0)
            self.training_started = state.get('training_started')
            self.last_checkpoint = state.get('last_checkpoint')
            return True
        return False


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup."""

    def __init__(self, checkpoint_dir: str = "checkpoints", keep_last: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        scaler: GradScaler,
        epoch: int,
        global_step: int,
        loss: float,
        psnr: float,
        is_best: bool = False
    ) -> str:
        """Save a checkpoint and clean up old ones."""

        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss,
            'psnr': psnr,
            'timestamp': datetime.now().isoformat()
        }

        # Save epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f"guardian_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "guardian_best.pt"
            torch.save(checkpoint, best_path)
            print(f"⭐ New best model saved: {best_path}")

        # Save latest (for quick resume)
        latest_path = self.checkpoint_dir / "guardian_latest.pt"
        torch.save(checkpoint, latest_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("guardian_epoch_*.pt"),
            key=lambda x: x.stat().st_mtime
        )

        # Keep best and latest always
        to_delete = checkpoints[:-self.keep_last] if len(checkpoints) > self.keep_last else []

        for ckpt in to_delete:
            ckpt.unlink()
            print(f"🗑️ Deleted old checkpoint: {ckpt.name}")

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """Load a checkpoint. If path not specified, load latest."""

        if checkpoint_path:
            path = Path(checkpoint_path)
        else:
            # Try latest first, then best
            path = self.checkpoint_dir / "guardian_latest.pt"
            if not path.exists():
                path = self.checkpoint_dir / "guardian_best.pt"

        if path.exists():
            print(f"📂 Loading checkpoint: {path}")
            return torch.load(path, map_location='cuda')
        return None

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest = self.checkpoint_dir / "guardian_latest.pt"
        if latest.exists():
            return str(latest)
        return None


class GracefulShutdown:
    """Handle graceful shutdown on Ctrl+C."""

    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        print("\n\n⚠️ Shutdown requested - finishing current batch and saving checkpoint...")
        self.should_stop = True


def create_model(config: Dict) -> nn.Module:
    """Create the Guardian model."""
    from mri_guardian.models.guardian import GuardianReconstructor

    model = GuardianReconstructor(
        in_channels=config.get('in_channels', 2),
        out_channels=config.get('out_channels', 2),
        num_iterations=config.get('num_iterations', 8),
        num_features=config.get('num_features', 64)
    )
    return model


def create_dataloader(data_dir: str, config: Dict, split: str = 'train') -> DataLoader:
    """Create dataloader for fastMRI data."""
    from mri_guardian.data.fastmri_loader import FastMRIDataset

    dataset = FastMRIDataset(
        root_dir=data_dir,
        split=split,
        acceleration=config.get('acceleration', 4),
        center_fraction=config.get('center_fraction', 0.08)
    )

    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True
    )

    return loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Dict,
    writer: SummaryWriter,
    training_state: TrainingState,
    shutdown: GracefulShutdown
) -> Dict[str, float]:
    """Train for one epoch with mixed precision."""

    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    grad_accum_steps = config.get('gradient_accumulation_steps', 1)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        if shutdown.should_stop:
            break

        # Move data to GPU
        undersampled = batch['undersampled'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        # Mixed precision forward pass
        with autocast():
            output = model(undersampled, mask)
            loss = nn.functional.l1_loss(output, target)
            loss = loss / grad_accum_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Calculate metrics
        with torch.no_grad():
            mse = nn.functional.mse_loss(output, target)
            psnr = 10 * torch.log10(1.0 / mse)

        total_loss += loss.item() * grad_accum_steps
        total_psnr += psnr.item()
        num_batches += 1
        training_state.global_step += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss/num_batches:.4f}",
            'psnr': f"{total_psnr/num_batches:.2f}dB"
        })

        # Log to tensorboard every 100 steps
        if training_state.global_step % 100 == 0:
            writer.add_scalar('train/loss', loss.item() * grad_accum_steps, training_state.global_step)
            writer.add_scalar('train/psnr', psnr.item(), training_state.global_step)

    return {
        'loss': total_loss / max(num_batches, 1),
        'psnr': total_psnr / max(num_batches, 1)
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate the model."""

    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation", leave=False):
            undersampled = batch['undersampled'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            with autocast():
                output = model(undersampled, mask)
                loss = nn.functional.l1_loss(output, target)

            mse = nn.functional.mse_loss(output, target)
            psnr = 10 * torch.log10(1.0 / mse)

            total_loss += loss.item()
            total_psnr += psnr.item()
            num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'psnr': total_psnr / max(num_batches, 1)
    }


def main():
    parser = argparse.ArgumentParser(description='Train MRI-GUARDIAN on RTX 4090')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to fastMRI data')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (4 optimal for 4090)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--half-dataset', action='store_true', help='Use only half the dataset')
    args = parser.parse_args()

    # Configuration optimized for RTX 4090
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'num_workers': 4,  # Good for external SSD
        'gradient_accumulation_steps': 4,  # Effective batch size = 16
        'acceleration': 4,
        'center_fraction': 0.08,
        'in_channels': 2,
        'out_channels': 2,
        'num_iterations': 8,
        'num_features': 64
    }

    print("=" * 70)
    print("MRI-GUARDIAN Training - RTX 4090 Optimized")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation_steps']})")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Half dataset: {args.half_dataset}")
    print("=" * 70)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize components
    shutdown = GracefulShutdown()
    training_state = TrainingState()
    checkpoint_manager = CheckpointManager(keep_last=5)

    # Create model
    model = create_model(config).to(device)
    print(f"\n📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler()

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint(args.checkpoint)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            training_state.load_state()
            print(f"✅ Resumed from epoch {checkpoint['epoch']}")
            print(f"   Previous loss: {checkpoint['loss']:.4f}")
            print(f"   Previous PSNR: {checkpoint['psnr']:.2f} dB")
        else:
            print("⚠️ No checkpoint found, starting fresh")

    # Create dataloaders
    print("\n📁 Loading datasets...")
    train_loader = create_dataloader(args.data_dir, config, 'train')
    val_loader = create_dataloader(args.data_dir, config, 'val')
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")

    # Tensorboard
    writer = SummaryWriter(log_dir='runs/guardian_training')

    # Training loop
    training_state.training_started = datetime.now().isoformat()
    print(f"\n🚀 Starting training at {training_state.training_started}")
    print("   Press Ctrl+C to gracefully stop and save checkpoint\n")

    try:
        for epoch in range(start_epoch, config['epochs']):
            if shutdown.should_stop:
                break

            training_state.epoch = epoch

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device,
                epoch, config, writer, training_state, shutdown
            )

            if shutdown.should_stop:
                break

            # Validate
            val_metrics = validate(model, val_loader, device, epoch)

            # Update scheduler
            scheduler.step()

            # Log metrics
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

            # Check if best
            is_best = val_metrics['psnr'] > training_state.best_psnr
            if is_best:
                training_state.best_psnr = val_metrics['psnr']
                training_state.best_loss = val_metrics['loss']

            # Save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, training_state.global_step,
                val_metrics['loss'], val_metrics['psnr'],
                is_best=is_best
            )
            training_state.last_checkpoint = checkpoint_path
            training_state.save_state()

            # Print epoch summary
            print(f"\n📈 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f} | PSNR: {train_metrics['psnr']:.2f} dB")
            print(f"   Val Loss:   {val_metrics['loss']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"   Best PSNR:  {training_state.best_psnr:.2f} dB")
            print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("   Saving emergency checkpoint...")
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, scaler,
            training_state.epoch, training_state.global_step,
            float('inf'), 0.0, is_best=False
        )
        training_state.save_state()
        raise

    finally:
        # Final save
        print("\n💾 Saving final checkpoint...")
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, scaler,
            training_state.epoch, training_state.global_step,
            val_metrics.get('loss', float('inf')),
            val_metrics.get('psnr', 0.0),
            is_best=False
        )
        training_state.save_state()
        writer.close()

    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print(f"   Best PSNR: {training_state.best_psnr:.2f} dB")
    print(f"   Checkpoints saved to: checkpoints/")
    print(f"   Logs saved to: runs/guardian_training/")
    print("=" * 70)


if __name__ == '__main__':
    main()
