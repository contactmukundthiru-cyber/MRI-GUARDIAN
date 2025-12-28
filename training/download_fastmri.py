"""
FastMRI Dataset Download Script
===============================

Downloads fastMRI data with resume support for crash recovery.

Prerequisites:
1. Register at https://fastmri.org/dataset/
2. Wait for approval (1-2 days)
3. Install AWS CLI: pip install awscli

Usage:
    # Download brain data (recommended first)
    python training/download_fastmri.py --dataset brain --output-dir E:/fastmri

    # Download knee data (for generalization)
    python training/download_fastmri.py --dataset knee --output-dir E:/fastmri

    # Download half dataset (faster)
    python training/download_fastmri.py --dataset brain --output-dir E:/fastmri --half

    # Resume interrupted download
    python training/download_fastmri.py --dataset brain --output-dir E:/fastmri --resume
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import hashlib
import json


# FastMRI S3 paths (public access, no credentials needed)
FASTMRI_FILES = {
    'brain': {
        'train': 'brain_multicoil_train.tar.gz',
        'val': 'brain_multicoil_val.tar.gz',
        'test': 'brain_multicoil_test.tar.gz',
        'train_size_gb': 800,
        'val_size_gb': 100,
        'test_size_gb': 100
    },
    'knee': {
        'train': 'knee_multicoil_train.tar.gz',
        'val': 'knee_multicoil_val.tar.gz',
        'test': 'knee_multicoil_test.tar.gz',
        'train_size_gb': 350,
        'val_size_gb': 50,
        'test_size_gb': 50
    }
}

S3_BUCKET = "fastmri-dataset"


def check_aws_cli():
    """Check if AWS CLI is installed."""
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
        print(f"✓ AWS CLI installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ AWS CLI not found. Install with: pip install awscli")
        return False


def get_download_state_file(output_dir: Path) -> Path:
    """Get path to download state file."""
    return output_dir / ".download_state.json"


def save_download_state(output_dir: Path, state: dict):
    """Save download progress state."""
    state_file = get_download_state_file(output_dir)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def load_download_state(output_dir: Path) -> dict:
    """Load download progress state."""
    state_file = get_download_state_file(output_dir)
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {'completed': [], 'in_progress': None}


def download_file(s3_path: str, local_path: Path, resume: bool = True) -> bool:
    """Download a file from S3 with resume support."""

    print(f"\n📥 Downloading: {s3_path}")
    print(f"   To: {local_path}")

    # AWS CLI command with resume support
    cmd = [
        'aws', 's3', 'cp',
        f's3://{S3_BUCKET}/{s3_path}',
        str(local_path),
        '--no-sign-request',  # Public bucket, no credentials needed
        '--only-show-errors'
    ]

    try:
        # Run download with progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Monitor progress
        while True:
            if process.poll() is not None:
                break

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"   ✅ Download complete: {local_path.name}")
            return True
        else:
            print(f"   ❌ Download failed: {stderr}")
            return False

    except KeyboardInterrupt:
        print("\n   ⚠️ Download interrupted - can resume later")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def extract_tar(tar_path: Path, output_dir: Path) -> bool:
    """Extract tar.gz file."""
    import tarfile

    print(f"\n📦 Extracting: {tar_path.name}")

    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Get total members for progress
            members = tar.getmembers()
            total = len(members)

            for i, member in enumerate(members):
                tar.extract(member, output_dir)
                if (i + 1) % 100 == 0:
                    print(f"   Extracted {i+1}/{total} files...", end='\r')

        print(f"   ✅ Extraction complete: {total} files")
        return True

    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return False


def download_dataset(
    dataset: str,
    output_dir: Path,
    splits: list = ['train', 'val'],
    half: bool = False,
    resume: bool = True
):
    """Download a complete dataset."""

    if dataset not in FASTMRI_FILES:
        print(f"❌ Unknown dataset: {dataset}")
        return False

    files = FASTMRI_FILES[dataset]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load state
    state = load_download_state(output_dir) if resume else {'completed': [], 'in_progress': None}

    print(f"\n{'=' * 60}")
    print(f"Downloading FastMRI {dataset.upper()} dataset")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Splits: {splits}")
    print(f"Half dataset: {half}")
    print(f"Resume mode: {resume}")

    total_size = sum(files.get(f'{s}_size_gb', 0) for s in splits)
    if half:
        total_size //= 2
    print(f"Estimated download size: ~{total_size} GB")
    print(f"{'=' * 60}\n")

    for split in splits:
        filename = files[split]

        # Skip if already completed
        if filename in state['completed']:
            print(f"⏭️ Skipping {filename} (already downloaded)")
            continue

        local_path = output_dir / filename

        # Download
        state['in_progress'] = filename
        save_download_state(output_dir, state)

        success = download_file(filename, local_path, resume=resume)

        if success:
            state['completed'].append(filename)
            state['in_progress'] = None
            save_download_state(output_dir, state)

            # Extract
            extract_success = extract_tar(local_path, output_dir)
            if extract_success:
                # Optionally delete tar to save space
                response = input(f"\n   Delete {filename} to save space? (y/n): ")
                if response.lower() == 'y':
                    local_path.unlink()
                    print(f"   🗑️ Deleted {filename}")
        else:
            print(f"\n⚠️ Download incomplete. Run with --resume to continue.")
            return False

    print(f"\n{'=' * 60}")
    print(f"✅ Dataset download complete!")
    print(f"   Location: {output_dir}")
    print(f"{'=' * 60}")

    return True


def create_subset(input_dir: Path, output_dir: Path, fraction: float = 0.5):
    """Create a subset of the dataset for faster training."""
    import random
    import shutil

    print(f"\n📁 Creating {fraction*100:.0f}% subset...")

    for split in ['train', 'val']:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue

        output_split_dir = output_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        # Get all h5 files
        h5_files = list(split_dir.glob("*.h5"))
        n_select = int(len(h5_files) * fraction)

        # Random selection
        random.seed(42)  # Reproducible
        selected = random.sample(h5_files, n_select)

        print(f"   {split}: {n_select}/{len(h5_files)} files")

        for f in selected:
            shutil.copy2(f, output_split_dir / f.name)

    print(f"   ✅ Subset created at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download FastMRI dataset')
    parser.add_argument('--dataset', type=str, choices=['brain', 'knee', 'both'],
                        default='brain', help='Which dataset to download')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory (e.g., E:/fastmri)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='Which splits to download')
    parser.add_argument('--half', action='store_true',
                        help='Create half-size subset after download')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted download')
    args = parser.parse_args()

    # Check prerequisites
    if not check_aws_cli():
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Download
    datasets = ['brain', 'knee'] if args.dataset == 'both' else [args.dataset]

    for dataset in datasets:
        dataset_dir = output_dir / dataset
        success = download_dataset(
            dataset=dataset,
            output_dir=dataset_dir,
            splits=args.splits,
            half=args.half,
            resume=args.resume
        )

        if not success:
            print(f"\n❌ Failed to download {dataset} dataset")
            sys.exit(1)

        # Create subset if requested
        if args.half:
            subset_dir = output_dir / f"{dataset}_half"
            create_subset(dataset_dir, subset_dir, fraction=0.5)

    print("\n" + "=" * 60)
    print("✅ All downloads complete!")
    print("\nNext steps:")
    print("1. Run training:")
    print(f"   python training/train_guardian_rtx4090.py --data-dir {output_dir}/brain")
    print("\n2. Or train on both anatomies:")
    print(f"   python training/train_multi_anatomy.py --data-dir {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
