"""
Results Packaging Script
========================

Packages all training results, checkpoints, and experiment outputs
into a single archive for easy download.

Usage:
    python cloud/package_results.py --output-dir results

This creates a ~1GB archive containing:
- Model checkpoints (best models only)
- All experiment results and figures
- Training logs
- Summary report
"""

import os
import sys
import argparse
import shutil
import tarfile
import json
from pathlib import Path
from datetime import datetime


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def package_results(
    project_dir: str,
    output_dir: str,
    include_all_checkpoints: bool = False
):
    """Package all results into a downloadable archive."""

    project_path = Path(project_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create staging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = output_path / f"mri_guardian_results_{timestamp}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MRI-GUARDIAN Results Packager")
    print("=" * 60)
    print(f"Project directory: {project_path}")
    print(f"Output directory: {output_path}")
    print()

    # 1. Copy best model checkpoints
    print("📦 Packaging model checkpoints...")
    checkpoints_dir = staging_dir / "checkpoints"
    checkpoints_dir.mkdir()

    src_checkpoints = project_path / "checkpoints"
    if src_checkpoints.exists():
        for modality in ['mri', 'ct', 'xray']:
            # Copy best checkpoint
            best_ckpt = src_checkpoints / f"{modality}_best.pt"
            if best_ckpt.exists():
                shutil.copy2(best_ckpt, checkpoints_dir / f"{modality}_best.pt")
                print(f"   ✓ {modality}_best.pt ({format_size(best_ckpt.stat().st_size)})")

            # Optionally copy all checkpoints
            if include_all_checkpoints:
                for ckpt in src_checkpoints.glob(f"{modality}_epoch_*.pt"):
                    shutil.copy2(ckpt, checkpoints_dir / ckpt.name)

    # 2. Copy experiment results
    print("\n📊 Packaging experiment results...")
    results_staging = staging_dir / "results"

    src_results = project_path / "results"
    if src_results.exists():
        shutil.copytree(src_results, results_staging)
        n_files = sum(1 for _ in results_staging.rglob('*') if _.is_file())
        print(f"   ✓ {n_files} result files")

    # 3. Copy training logs
    print("\n📝 Packaging training logs...")
    logs_staging = staging_dir / "logs"

    src_logs = project_path / "logs"
    if src_logs.exists():
        shutil.copytree(src_logs, logs_staging)
        print(f"   ✓ Training logs copied")

    # 4. Copy dashboard
    print("\n🖥️ Packaging dashboard...")
    dashboard_staging = staging_dir / "dashboard"

    src_dashboard = project_path / "dashboard"
    if src_dashboard.exists():
        shutil.copytree(src_dashboard, dashboard_staging)
        print(f"   ✓ Dashboard files copied")

    # 5. Copy key source files
    print("\n📁 Packaging source code...")
    src_staging = staging_dir / "mri_guardian"

    src_code = project_path / "mri_guardian"
    if src_code.exists():
        shutil.copytree(src_code, src_staging)
        print(f"   ✓ Source code copied")

    # 6. Create summary manifest
    print("\n📋 Creating manifest...")

    manifest = {
        "created": datetime.now().isoformat(),
        "project": "MRI-GUARDIAN",
        "description": "Physics-Guided MRI Reconstruction and Hallucination Auditor",
        "contents": {
            "checkpoints": [],
            "experiments": [],
            "modalities_trained": []
        },
        "metrics": {}
    }

    # List checkpoints
    for ckpt in checkpoints_dir.glob("*.pt"):
        manifest["contents"]["checkpoints"].append(ckpt.name)
        modality = ckpt.stem.split("_")[0]
        if modality not in manifest["contents"]["modalities_trained"]:
            manifest["contents"]["modalities_trained"].append(modality)

    # List experiments
    if results_staging.exists():
        for exp_dir in results_staging.iterdir():
            if exp_dir.is_dir():
                manifest["contents"]["experiments"].append(exp_dir.name)

    # Add metrics from summary report if available
    summary_report = results_staging / "summary_report.txt"
    if summary_report.exists():
        manifest["summary_report"] = "results/summary_report.txt"

    # Save manifest
    manifest_path = staging_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # 7. Create README
    readme_content = f"""# MRI-GUARDIAN Training Results

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents

- `checkpoints/` - Trained model weights (best checkpoints)
- `results/` - Experiment outputs and figures
- `logs/` - Training logs
- `dashboard/` - Streamlit dashboard
- `mri_guardian/` - Source code

## Modalities Trained

{chr(10).join(f"- {m.upper()}" for m in manifest["contents"]["modalities_trained"])}

## Quick Start

1. View results:
   ```
   streamlit run dashboard/app.py
   ```

2. Load a trained model:
   ```python
   import torch
   from mri_guardian.models.guardian import GuardianReconstructor

   model = GuardianReconstructor()
   checkpoint = torch.load("checkpoints/mri_best.pt")
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

## Experiments

{chr(10).join(f"- {exp}" for exp in manifest["contents"]["experiments"])}

## Novel Contributions

1. **Minimum Detectable Size (MDS)**: MDS = k × √R
2. **Lesion Integrity Marker (LIM)**: 14-feature preservation score
3. **Biological Plausibility Score (BPS)**: 6 biological priors
4. **Virtual Clinical Trial (VCT)**: Regulatory-grade validation

---
MRI-GUARDIAN | ISEF Bioengineering Project
"""

    readme_path = staging_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    # 8. Create tar archive
    print("\n📦 Creating archive...")
    archive_name = f"mri_guardian_results_{timestamp}.tar.gz"
    archive_path = output_path / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(staging_dir, arcname=staging_dir.name)

    # 9. Cleanup staging
    shutil.rmtree(staging_dir)

    # 10. Summary
    archive_size = archive_path.stat().st_size

    print("\n" + "=" * 60)
    print("✅ PACKAGING COMPLETE")
    print("=" * 60)
    print(f"\nArchive: {archive_path}")
    print(f"Size: {format_size(archive_size)}")
    print(f"\nModalities: {', '.join(manifest['contents']['modalities_trained'])}")
    print(f"Experiments: {len(manifest['contents']['experiments'])}")
    print(f"\nTo download to your local machine:")
    print(f"  scp ubuntu@<instance-ip>:{archive_path} .")
    print()

    return str(archive_path)


def main():
    parser = argparse.ArgumentParser(description='Package MRI-GUARDIAN results')
    parser.add_argument('--project-dir', type=str, default='.', help='Project directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--all-checkpoints', action='store_true', help='Include all checkpoints, not just best')
    args = parser.parse_args()

    archive_path = package_results(
        project_dir=args.project_dir,
        output_dir=args.output_dir,
        include_all_checkpoints=args.all_checkpoints
    )

    print(f"Archive created: {archive_path}")


if __name__ == '__main__':
    main()
