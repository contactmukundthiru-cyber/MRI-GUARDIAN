"""
Multi-Modality Medical Imaging Dataset Download Script
=======================================================

Downloads public datasets for training MRI-GUARDIAN's universal imaging framework.

Available Datasets:
------------------
1. MRI (FastMRI) - NYU, ~1.5 TB
2. CT (LIDC-IDRI) - Lung cancer screening, ~125 GB
3. CT (COVID-CT) - COVID-19 CT scans, ~3 GB
4. X-ray (ChestX-ray14) - NIH chest X-rays, ~45 GB
5. X-ray (COVID-CXR) - COVID chest X-rays, ~2 GB
6. Mammography (CBIS-DDSM) - Breast cancer, ~165 GB
7. Retinal OCT - Kermany dataset, ~5 GB

Usage:
    # Download recommended starter pack (small, diverse)
    python training/download_multimodality.py --starter-pack --output-dir E:/medical_data

    # Download specific modalities
    python training/download_multimodality.py --datasets mri ct xray --output-dir E:/medical_data

    # Download everything
    python training/download_multimodality.py --all --output-dir E:/medical_data
"""

import os
import sys
import argparse
import subprocess
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import json
from tqdm import tqdm


# Dataset registry
DATASETS = {
    # MRI
    'mri_brain': {
        'name': 'FastMRI Brain',
        'modality': 'mri',
        'source': 'NYU/fastMRI',
        'size_gb': 1000,
        'format': 'h5',
        'download_method': 'aws_s3',
        'url': 's3://fastmri-dataset/',
        'registration': 'https://fastmri.org/dataset/',
        'description': 'Brain MRI multicoil data for reconstruction'
    },

    # CT - Lung
    'ct_lidc': {
        'name': 'LIDC-IDRI',
        'modality': 'ct',
        'source': 'TCIA',
        'size_gb': 125,
        'format': 'dicom',
        'download_method': 'tcia',
        'url': 'https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI',
        'registration': 'https://wiki.cancerimagingarchive.net/',
        'description': 'Lung CT with expert lesion annotations'
    },

    # CT - COVID (smaller, easier to download)
    'ct_covid': {
        'name': 'COVID-CT',
        'modality': 'ct',
        'source': 'GitHub/UCSD-AI4H',
        'size_gb': 3,
        'format': 'png',
        'download_method': 'github',
        'url': 'https://github.com/UCSD-AI4H/COVID-CT',
        'registration': None,
        'description': 'COVID-19 CT scans - small, easy to start'
    },

    # X-ray - NIH
    'xray_nih': {
        'name': 'ChestX-ray14',
        'modality': 'xray',
        'source': 'NIH Clinical Center',
        'size_gb': 45,
        'format': 'png',
        'download_method': 'box',
        'url': 'https://nihcc.app.box.com/v/ChestXray-NIHCC',
        'registration': None,
        'description': '112K chest X-rays with 14 disease labels'
    },

    # X-ray - COVID (smaller)
    'xray_covid': {
        'name': 'COVID-CXR',
        'modality': 'xray',
        'source': 'GitHub',
        'size_gb': 2,
        'format': 'png',
        'download_method': 'github',
        'url': 'https://github.com/ieee8023/covid-chestxray-dataset',
        'registration': None,
        'description': 'COVID chest X-rays - small, easy to start'
    },

    # Mammography
    'mammo_ddsm': {
        'name': 'CBIS-DDSM',
        'modality': 'mammography',
        'source': 'TCIA',
        'size_gb': 165,
        'format': 'dicom',
        'download_method': 'tcia',
        'url': 'https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM',
        'registration': 'https://wiki.cancerimagingarchive.net/',
        'description': 'Curated breast imaging with lesion annotations'
    },

    # OCT - Retinal
    'oct_retinal': {
        'name': 'Kermany OCT',
        'modality': 'oct',
        'source': 'Mendeley/Kermany',
        'size_gb': 5,
        'format': 'jpeg',
        'download_method': 'kaggle',
        'url': 'https://data.mendeley.com/datasets/rscbjbr9sj/2',
        'kaggle_dataset': 'paultimothymooney/kermany2018',
        'registration': None,
        'description': 'Retinal OCT images for disease classification'
    }
}


# Starter pack: small, diverse datasets for quick experimentation
STARTER_PACK = ['ct_covid', 'xray_covid', 'oct_retinal']

# Full training: comprehensive datasets
FULL_TRAINING = ['mri_brain', 'ct_lidc', 'xray_nih', 'mammo_ddsm']


def download_progress(url: str, filepath: Path, desc: str = None):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_github_repo(url: str, output_dir: Path, dataset_name: str):
    """Download dataset from GitHub."""
    print(f"\n📥 Downloading {dataset_name} from GitHub...")

    # Clone the repo
    repo_dir = output_dir / dataset_name
    if repo_dir.exists():
        print(f"   ⏭️ Already exists: {repo_dir}")
        return True

    cmd = ['git', 'clone', '--depth', '1', url, str(repo_dir)]

    try:
        subprocess.run(cmd, check=True)
        print(f"   ✅ Downloaded to: {repo_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        return False


def download_kaggle_dataset(dataset_slug: str, output_dir: Path, dataset_name: str):
    """Download dataset from Kaggle."""
    print(f"\n📥 Downloading {dataset_name} from Kaggle...")

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Check if kaggle CLI is installed
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ Kaggle CLI not found. Install with: pip install kaggle")
        print("   📝 Then configure API key: https://www.kaggle.com/docs/api")
        return False

    # Download dataset
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', str(dataset_dir), '--unzip']

    try:
        subprocess.run(cmd, check=True)
        print(f"   ✅ Downloaded to: {dataset_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        return False


def download_tcia_dataset(dataset_name: str, output_dir: Path):
    """Instructions for TCIA (The Cancer Imaging Archive) datasets."""
    print(f"\n📋 Instructions for downloading {dataset_name} from TCIA:")
    print("=" * 60)
    print("TCIA requires the NBIA Data Retriever tool:")
    print()
    print("1. Go to: https://wiki.cancerimagingarchive.net/")
    print("2. Search for the dataset")
    print("3. Download the NBIA Data Retriever")
    print("4. Use the retriever to download to:", output_dir)
    print()
    print("Alternative: Use TCIA REST API with Python")
    print("   pip install tciaclient")
    print("=" * 60)
    return False  # Manual download required


def download_fastmri(output_dir: Path):
    """FastMRI download instructions/automation."""
    print(f"\n📋 Instructions for downloading FastMRI:")
    print("=" * 60)
    print("1. Register at: https://fastmri.org/dataset/")
    print("2. Wait for approval (1-2 days)")
    print("3. Run: python training/download_fastmri.py --output-dir", output_dir)
    print("=" * 60)
    print("\nAlternatively, use AWS CLI directly:")
    print("   aws s3 cp s3://fastmri-dataset/brain_multicoil_train.tar.gz . --no-sign-request")
    return False


def download_dataset(dataset_id: str, output_dir: Path) -> bool:
    """Download a specific dataset."""

    if dataset_id not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_id}")
        return False

    info = DATASETS[dataset_id]
    dataset_dir = output_dir / info['modality'] / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {info['name']}")
    print(f"Modality: {info['modality'].upper()}")
    print(f"Size: ~{info['size_gb']} GB")
    print(f"Source: {info['source']}")
    print(f"{'=' * 60}")

    method = info['download_method']

    if method == 'github':
        return download_github_repo(info['url'], dataset_dir, dataset_id)

    elif method == 'kaggle':
        return download_kaggle_dataset(info['kaggle_dataset'], dataset_dir, dataset_id)

    elif method == 'tcia':
        return download_tcia_dataset(info['name'], dataset_dir)

    elif method == 'aws_s3':
        return download_fastmri(dataset_dir)

    elif method == 'box':
        print(f"\n📋 Manual download required from Box:")
        print(f"   URL: {info['url']}")
        print(f"   Save to: {dataset_dir}")
        return False

    else:
        print(f"❌ Unknown download method: {method}")
        return False


def create_dataset_manifest(output_dir: Path):
    """Create a manifest of downloaded datasets."""
    manifest = {
        'datasets': {},
        'total_samples': 0,
        'modalities': []
    }

    for modality_dir in output_dir.iterdir():
        if modality_dir.is_dir():
            modality = modality_dir.name
            if modality not in manifest['modalities']:
                manifest['modalities'].append(modality)

            for dataset_dir in modality_dir.iterdir():
                if dataset_dir.is_dir():
                    # Count files
                    n_files = sum(1 for _ in dataset_dir.rglob('*') if _.is_file())
                    manifest['datasets'][dataset_dir.name] = {
                        'modality': modality,
                        'path': str(dataset_dir),
                        'n_files': n_files
                    }
                    manifest['total_samples'] += n_files

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📋 Created manifest: {manifest_path}")
    return manifest


def print_dataset_info():
    """Print information about available datasets."""
    print("\n" + "=" * 70)
    print("Available Datasets for Multi-Modality Training")
    print("=" * 70)

    by_modality = {}
    for dataset_id, info in DATASETS.items():
        mod = info['modality']
        if mod not in by_modality:
            by_modality[mod] = []
        by_modality[mod].append((dataset_id, info))

    for modality, datasets in by_modality.items():
        print(f"\n📁 {modality.upper()}")
        for dataset_id, info in datasets:
            print(f"   • {dataset_id}: {info['name']} (~{info['size_gb']} GB)")
            print(f"     {info['description']}")

    print("\n" + "-" * 70)
    print("Recommended Starter Pack (small, diverse):")
    for ds in STARTER_PACK:
        print(f"   • {ds}: ~{DATASETS[ds]['size_gb']} GB")

    print("\nTotal starter pack: ~10 GB")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description='Download multi-modality medical imaging datasets')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to download')
    parser.add_argument('--starter-pack', action='store_true', help='Download small starter pack')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--modality', type=str, help='Download all datasets for a modality')
    args = parser.parse_args()

    if args.list:
        print_dataset_info()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to download
    if args.starter_pack:
        datasets = STARTER_PACK
        print("\n🚀 Downloading Starter Pack (small, diverse datasets)")
    elif args.all:
        datasets = list(DATASETS.keys())
        print("\n🚀 Downloading ALL datasets")
    elif args.modality:
        datasets = [d for d, info in DATASETS.items() if info['modality'] == args.modality]
        print(f"\n🚀 Downloading all {args.modality.upper()} datasets")
    elif args.datasets:
        datasets = args.datasets
    else:
        print("❌ Please specify --datasets, --starter-pack, --modality, or --all")
        print_dataset_info()
        return

    print(f"\n📁 Output directory: {output_dir}")
    print(f"📦 Datasets to download: {len(datasets)}")

    # Calculate total size
    total_size = sum(DATASETS[d]['size_gb'] for d in datasets if d in DATASETS)
    print(f"💾 Estimated total size: ~{total_size} GB")

    # Confirm
    response = input(f"\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    # Download each dataset
    results = {}
    for dataset_id in datasets:
        success = download_dataset(dataset_id, output_dir)
        results[dataset_id] = 'success' if success else 'manual_required'

    # Create manifest
    manifest = create_dataset_manifest(output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)

    for dataset_id, status in results.items():
        icon = "✅" if status == 'success' else "📋"
        print(f"   {icon} {dataset_id}: {status}")

    print(f"\n📊 Total files: {manifest['total_samples']}")
    print(f"📁 Modalities: {', '.join(manifest['modalities'])}")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Complete any manual downloads listed above")
    print("2. Run multi-modality training:")
    print(f"   python training/train_multimodality.py --data-dir {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
