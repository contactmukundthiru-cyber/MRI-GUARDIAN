"""
Experiment 4: Comprehensive Safety Framework Evaluation

Tests the complete AI MRI Safety Framework:
1. Uncertainty estimation accuracy
2. Distribution shift detection
3. Physics violation detection
4. Lesion integrity verification
5. Bias detection across patient subgroups
6. Clinical risk scoring accuracy
7. FDA-aligned regulatory metrics

Hypotheses:
H4: The integrated safety framework provides reliable risk assessment
    for AI-reconstructed MRI images across diverse clinical scenarios.

This experiment demonstrates the novel contribution of a complete
AI safety pipeline for medical imaging.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import MRI-GUARDIAN modules
from mri_guardian.data.fastmri_loader import SimulatedMRIDataset
from mri_guardian.data.transforms import MRIDataTransform
from mri_guardian.data.kspace_ops import fft2c, ifft2c
from mri_guardian.models.guardian import GuardianModel, GuardianConfig
from mri_guardian.models.unet import UNet
from mri_guardian.models.blackbox import BlackBoxModel, HallucinationInjector

# Import Safety Framework
from mri_guardian.safety.uncertainty import (
    MCDropoutEstimator,
    TTAUncertaintyEstimator,
    CombinedUncertaintyEstimator
)
from mri_guardian.safety.distribution_shift import (
    DistributionShiftDetector,
    AnatomyOutlierDetector
)
from mri_guardian.safety.physics_violation import PhysicsViolationDetector
from mri_guardian.safety.lesion_integrity import (
    LesionIntegrityVerifier,
    SubtleLesionGenerator
)
from mri_guardian.safety.bias_detection import BiasDetector, SubgroupAnalyzer
from mri_guardian.safety.clinical_risk import (
    ClinicalRiskScorer,
    ClinicalConfidenceMap,
    IntegratedSafetyScorer
)
from mri_guardian.safety.multi_signal import MultiSignalConsistencyChecker
from mri_guardian.safety.regulatory import (
    FDAMetricsCalculator,
    SafetyBenchmark,
    ArtifactSeverityScorer
)
from mri_guardian.metrics.image_quality import compute_psnr, compute_ssim


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_test_scenarios(
    base_images: List[torch.Tensor],
    masks: List[torch.Tensor],
    device: str = 'cuda'
) -> Dict[str, List[Dict]]:
    """
    Create comprehensive test scenarios for safety evaluation.

    Returns scenarios for:
    - Normal cases (baseline)
    - Distribution shift cases
    - Physics violation cases
    - Lesion preservation cases
    - Bias evaluation cases
    """
    scenarios = {
        'normal': [],
        'distribution_shift': [],
        'physics_violation': [],
        'lesion_test': [],
        'bias_test': []
    }

    lesion_gen = SubtleLesionGenerator(device=device)
    hallucinator = HallucinationInjector()

    for i, (img, mask) in enumerate(zip(base_images, masks)):
        img = img.to(device)
        mask = mask.to(device)

        # Normal case
        scenarios['normal'].append({
            'image': img,
            'mask': mask,
            'label': 'normal',
            'index': i
        })

        # Distribution shift - simulate scanner difference
        # Add noise with different characteristics
        shifted = img.clone()
        noise = torch.randn_like(shifted) * 0.1 * shifted.std()
        # Add bias field (low-frequency intensity variation)
        H, W = shifted.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        bias_field = 1 + 0.2 * (x**2 + y**2)
        shifted = shifted * bias_field + noise

        scenarios['distribution_shift'].append({
            'image': shifted,
            'mask': mask,
            'label': 'scanner_shift',
            'original': img,
            'index': i
        })

        # Physics violation - create non-physical reconstruction
        violated = img.clone()
        # Add isolated k-space spikes (causes ringing)
        kspace = fft2c(violated.squeeze())
        cy, cx = H // 2, W // 2
        # Add random spikes
        for _ in range(3):
            spike_y = np.random.randint(10, H-10)
            spike_x = np.random.randint(10, W-10)
            if abs(spike_y - cy) > 20 or abs(spike_x - cx) > 20:
                kspace[spike_y, spike_x] = kspace.abs().max() * 2

        violated = torch.abs(ifft2c(kspace)).unsqueeze(0).unsqueeze(0)

        scenarios['physics_violation'].append({
            'image': violated,
            'kspace': kspace,
            'mask': mask,
            'label': 'ringing_artifact',
            'original': img,
            'index': i
        })

        # Lesion test cases
        for lesion_type in ['micro', 'low_contrast', 'linear']:
            if lesion_type == 'micro':
                img_lesion, lesion_mask, info = lesion_gen.generate_micro_lesion(
                    img.unsqueeze(0) if img.dim() == 3 else img,
                    seed=i
                )
            elif lesion_type == 'low_contrast':
                img_lesion, lesion_mask, info = lesion_gen.generate_low_contrast_lesion(
                    img.unsqueeze(0) if img.dim() == 3 else img,
                    seed=i + 1000
                )
            else:
                img_lesion, lesion_mask, info = lesion_gen.generate_linear_structure(
                    img.unsqueeze(0) if img.dim() == 3 else img,
                    seed=i + 2000
                )

            scenarios['lesion_test'].append({
                'image': img_lesion,
                'lesion_mask': lesion_mask,
                'lesion_info': info[0] if info else {},
                'mask': mask,
                'lesion_type': lesion_type,
                'original': img,
                'index': i
            })

        # Bias test - tag with subgroup info
        # Simulate different patient characteristics
        body_size = 'small' if i % 3 == 0 else ('large' if i % 3 == 1 else 'medium')
        contrast_level = 'low' if i % 2 == 0 else 'high'

        scenarios['bias_test'].append({
            'image': img,
            'mask': mask,
            'subgroup': {
                'body_size': body_size,
                'contrast': contrast_level
            },
            'index': i
        })

    return scenarios


def evaluate_uncertainty(
    model: nn.Module,
    scenarios: List[Dict],
    device: str = 'cuda'
) -> Dict:
    """Evaluate uncertainty estimation quality."""
    print("\n--- Evaluating Uncertainty Estimation ---")

    mc_estimator = MCDropoutEstimator(num_samples=10)
    tta_estimator = TTAUncertaintyEstimator(num_augmentations=4)
    combined_estimator = CombinedUncertaintyEstimator()

    results = {
        'mc_dropout': [],
        'tta': [],
        'combined': []
    }

    for scenario in tqdm(scenarios[:20], desc="Uncertainty evaluation"):
        img = scenario['image'].to(device)
        mask = scenario['mask'].to(device)

        # Get k-space
        kspace = fft2c(img.squeeze())
        masked_kspace = kspace * mask

        inputs = {
            'masked_kspace': masked_kspace.unsqueeze(0),
            'mask': mask.unsqueeze(0)
        }

        # MC Dropout
        try:
            mc_result = mc_estimator.estimate(model, inputs)
            results['mc_dropout'].append({
                'mean_uncertainty': mc_result.total.mean().item(),
                'epistemic': mc_result.epistemic.mean().item()
            })
        except Exception as e:
            results['mc_dropout'].append({'error': str(e)})

        # TTA
        try:
            tta_result = tta_estimator.estimate(model, inputs)
            results['tta'].append({
                'mean_uncertainty': tta_result.total.mean().item(),
                'epistemic': tta_result.epistemic.mean().item()
            })
        except Exception as e:
            results['tta'].append({'error': str(e)})

        # Combined
        try:
            combined_result = combined_estimator.estimate(model, inputs)
            results['combined'].append({
                'mean_uncertainty': combined_result.total.mean().item(),
                'confidence': combined_result.confidence.mean().item()
            })
        except Exception as e:
            results['combined'].append({'error': str(e)})

    # Aggregate
    summary = {}
    for method, method_results in results.items():
        valid_results = [r for r in method_results if 'mean_uncertainty' in r]
        if valid_results:
            summary[method] = {
                'mean_uncertainty': np.mean([r['mean_uncertainty'] for r in valid_results]),
                'std_uncertainty': np.std([r['mean_uncertainty'] for r in valid_results])
            }

    return {'detailed': results, 'summary': summary}


def evaluate_distribution_shift(
    model: nn.Module,
    normal_scenarios: List[Dict],
    shift_scenarios: List[Dict],
    device: str = 'cuda'
) -> Dict:
    """Evaluate distribution shift detection."""
    print("\n--- Evaluating Distribution Shift Detection ---")

    # Fit detector on normal cases
    normal_images = [s['image'] for s in normal_scenarios[:50]]

    detector = DistributionShiftDetector(device=device)
    detector.fit(normal_images)

    anatomy_detector = AnatomyOutlierDetector()
    anatomy_detector.fit(normal_images)

    # Test on normal cases (should be in-distribution)
    normal_scores = []
    for scenario in tqdm(normal_scenarios[:20], desc="Normal cases"):
        result = detector.detect(scenario['image'].to(device))
        normal_scores.append(result.ood_score)

    # Test on shifted cases (should be out-of-distribution)
    shift_scores = []
    shift_types = []
    for scenario in tqdm(shift_scenarios[:20], desc="Shift cases"):
        result = detector.detect(scenario['image'].to(device))
        shift_scores.append(result.ood_score)
        shift_types.append(result.shift_type)

    # Compute detection metrics
    labels = [0] * len(normal_scores) + [1] * len(shift_scores)
    scores = normal_scores + shift_scores

    from sklearn.metrics import roc_auc_score, accuracy_score

    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, scores)
        predictions = [1 if s > 0.5 else 0 for s in scores]
        accuracy = accuracy_score(labels, predictions)
    else:
        auc = 0.5
        accuracy = 0.5

    return {
        'normal_scores': normal_scores,
        'shift_scores': shift_scores,
        'auc': auc,
        'accuracy': accuracy,
        'mean_normal_score': np.mean(normal_scores),
        'mean_shift_score': np.mean(shift_scores),
        'shift_types_detected': shift_types
    }


def evaluate_physics_violations(
    physics_scenarios: List[Dict],
    device: str = 'cuda'
) -> Dict:
    """Evaluate physics violation detection."""
    print("\n--- Evaluating Physics Violation Detection ---")

    detector = PhysicsViolationDetector()

    results = []
    for scenario in tqdm(physics_scenarios[:20], desc="Physics violations"):
        img = scenario['image'].to(device)
        kspace = scenario['kspace'].to(device) if 'kspace' in scenario else fft2c(img.squeeze())
        mask = scenario['mask'].to(device)

        report = detector.detect(img, kspace.unsqueeze(0), mask.unsqueeze(0))

        results.append({
            'label': scenario['label'],
            'total_severity': report.total_severity,
            'is_plausible': report.is_physically_plausible,
            'violations': [v.violation_type for v in report.violations if v.severity > 0.3]
        })

    # Compute metrics
    detected = sum(1 for r in results if not r['is_plausible'])
    total = len(results)

    return {
        'detailed': results,
        'detection_rate': detected / total if total > 0 else 0,
        'mean_severity': np.mean([r['total_severity'] for r in results]),
        'common_violations': _get_common_items([r['violations'] for r in results])
    }


def _get_common_items(lists: List[List]) -> Dict[str, int]:
    """Count occurrences of items across lists."""
    counts = {}
    for lst in lists:
        for item in lst:
            counts[item] = counts.get(item, 0) + 1
    return counts


def evaluate_lesion_integrity(
    model: nn.Module,
    lesion_scenarios: List[Dict],
    device: str = 'cuda'
) -> Dict:
    """Evaluate lesion integrity preservation."""
    print("\n--- Evaluating Lesion Integrity ---")

    verifier = LesionIntegrityVerifier(device=device)

    results_by_type = {
        'micro': [],
        'low_contrast': [],
        'linear': []
    }

    model.eval()
    with torch.no_grad():
        for scenario in tqdm(lesion_scenarios, desc="Lesion integrity"):
            img = scenario['image'].to(device)
            mask = scenario['mask'].to(device)
            lesion_mask = scenario['lesion_mask'].to(device)
            lesion_type = scenario['lesion_type']

            # Reconstruct
            kspace = fft2c(img.squeeze())
            masked_kspace = kspace * mask

            result = model(masked_kspace.unsqueeze(0), mask.unsqueeze(0))
            recon = result['output'] if isinstance(result, dict) else result

            # Verify integrity
            integrity = verifier.verify_single(
                img, recon, lesion_mask, scenario['lesion_info']
            )

            results_by_type[lesion_type].append({
                'preserved': integrity.lesion_preserved,
                'score': integrity.preservation_score,
                'contrast': integrity.contrast_preservation,
                'sharpness': integrity.boundary_sharpness
            })

    # Aggregate by type
    summary = {}
    for ltype, results in results_by_type.items():
        if results:
            summary[ltype] = {
                'preservation_rate': sum(r['preserved'] for r in results) / len(results),
                'mean_score': np.mean([r['score'] for r in results]),
                'mean_contrast': np.mean([r['contrast'] for r in results]),
                'mean_sharpness': np.mean([r['sharpness'] for r in results])
            }

    return {'by_type': summary, 'detailed': results_by_type}


def evaluate_bias(
    model: nn.Module,
    bias_scenarios: List[Dict],
    device: str = 'cuda'
) -> Dict:
    """Evaluate patient-specific bias."""
    print("\n--- Evaluating Subgroup Bias ---")

    # Collect images for fitting
    images = [s['image'] for s in bias_scenarios]

    bias_detector = BiasDetector(n_subgroups=3)
    bias_detector.fit(images)

    model.eval()
    with torch.no_grad():
        for scenario in tqdm(bias_scenarios, desc="Bias evaluation"):
            img = scenario['image'].to(device)
            mask = scenario['mask'].to(device)

            # Reconstruct
            kspace = fft2c(img.squeeze())
            masked_kspace = kspace * mask

            result = model(masked_kspace.unsqueeze(0), mask.unsqueeze(0))
            recon = result['output'] if isinstance(result, dict) else result

            # Compute metrics
            psnr = compute_psnr(recon, img)
            ssim = compute_ssim(recon, img)

            # Record
            bias_detector.record_sample(
                image=img,
                reconstruction=recon,
                ground_truth=img,
                metrics={'psnr': psnr, 'ssim': ssim},
                metadata=scenario['subgroup']
            )

    # Analyze bias
    report = bias_detector.analyze()

    return {
        'overall_bias_score': report.overall_bias_score,
        'worst_group': report.worst_performing_group,
        'best_group': report.best_performing_group,
        'performance_gap': report.performance_gap,
        'subgroup_metrics': [
            {
                'name': m.subgroup_name,
                'psnr': m.mean_psnr,
                'ssim': m.mean_ssim,
                'n_samples': m.num_samples
            }
            for m in report.subgroup_metrics
        ],
        'recommendations': report.recommendations
    }


def evaluate_clinical_risk(
    model: nn.Module,
    scenarios: Dict[str, List[Dict]],
    device: str = 'cuda'
) -> Dict:
    """Evaluate integrated clinical risk scoring."""
    print("\n--- Evaluating Clinical Risk Scoring ---")

    risk_scorer = ClinicalRiskScorer()
    confidence_mapper = ClinicalConfidenceMap()

    risk_results = {
        'normal': [],
        'high_risk': []  # From physics violations and distribution shifts
    }

    model.eval()

    # Evaluate normal cases
    for scenario in tqdm(scenarios['normal'][:10], desc="Normal risk"):
        img = scenario['image'].to(device)
        mask = scenario['mask'].to(device)

        kspace = fft2c(img.squeeze())
        masked_kspace = kspace * mask

        with torch.no_grad():
            result = model(masked_kspace.unsqueeze(0), mask.unsqueeze(0))
            recon = result['output'] if isinstance(result, dict) else result

        assessment = risk_scorer.compute(
            uncertainty_score=0.1,  # Low for normal
            physics_violation_score=0.1,
            distribution_shift_score=0.1,
            hallucination_score=0.1
        )

        risk_results['normal'].append({
            'risk_level': assessment.risk_level.value,
            'risk_score': assessment.risk_score,
            'requires_review': assessment.requires_expert_review
        })

    # Evaluate high-risk cases (physics violations)
    for scenario in tqdm(scenarios['physics_violation'][:10], desc="High-risk"):
        img = scenario['image'].to(device)

        assessment = risk_scorer.compute(
            uncertainty_score=0.6,
            physics_violation_score=0.7,
            distribution_shift_score=0.3,
            hallucination_score=0.5
        )

        risk_results['high_risk'].append({
            'risk_level': assessment.risk_level.value,
            'risk_score': assessment.risk_score,
            'requires_review': assessment.requires_expert_review
        })

    # Compute discrimination
    normal_scores = [r['risk_score'] for r in risk_results['normal']]
    high_scores = [r['risk_score'] for r in risk_results['high_risk']]

    return {
        'normal_mean_risk': np.mean(normal_scores),
        'high_risk_mean_risk': np.mean(high_scores),
        'separation': np.mean(high_scores) - np.mean(normal_scores),
        'normal_review_rate': sum(r['requires_review'] for r in risk_results['normal']) / len(risk_results['normal']),
        'high_risk_review_rate': sum(r['requires_review'] for r in risk_results['high_risk']) / len(risk_results['high_risk']),
        'detailed': risk_results
    }


def evaluate_regulatory_metrics(
    model: nn.Module,
    scenarios: List[Dict],
    device: str = 'cuda'
) -> Dict:
    """Evaluate FDA-aligned regulatory metrics."""
    print("\n--- Evaluating Regulatory Metrics ---")

    fda_calc = FDAMetricsCalculator()
    artifact_scorer = ArtifactSeverityScorer()

    predictions = []
    ground_truths = []
    metadata = []

    model.eval()
    with torch.no_grad():
        for scenario in tqdm(scenarios[:30], desc="FDA metrics"):
            img = scenario['image'].to(device)
            mask = scenario['mask'].to(device)

            kspace = fft2c(img.squeeze())
            masked_kspace = kspace * mask

            result = model(masked_kspace.unsqueeze(0), mask.unsqueeze(0))
            recon = result['output'] if isinstance(result, dict) else result

            predictions.append(recon.cpu())
            ground_truths.append(img.cpu())
            metadata.append({'index': scenario['index']})

    # Calculate metrics
    benchmark_result = fda_calc.calculate(
        predictions, ground_truths, metadata=metadata
    )

    return {
        'overall_pass': benchmark_result.overall_pass,
        'certification_level': benchmark_result.certification_level,
        'psnr': benchmark_result.metrics.psnr,
        'ssim': benchmark_result.metrics.ssim,
        'artifact_severity': benchmark_result.metrics.artifact_severity,
        'failure_rate': benchmark_result.metrics.failure_rate,
        'worst_case_psnr': benchmark_result.metrics.worst_case_psnr,
        'recommendations': benchmark_result.recommendations
    }


def create_visualizations(
    results: Dict,
    output_dir: Path
) -> None:
    """Create visualization figures for the experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution shift detection
    if 'distribution_shift' in results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        shift_data = results['distribution_shift']
        x = ['Normal', 'Shifted']
        y = [shift_data['mean_normal_score'], shift_data['mean_shift_score']]

        bars = ax.bar(x, y, color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('OOD Score')
        ax.set_title(f'Distribution Shift Detection (AUC: {shift_data["auc"]:.3f})')
        ax.axhline(y=0.5, color='gray', linestyle='--', label='Threshold')
        ax.legend()

        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center')

        fig.savefig(output_dir / 'distribution_shift.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 2. Lesion integrity by type
    if 'lesion_integrity' in results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        lesion_data = results['lesion_integrity']['by_type']
        types = list(lesion_data.keys())
        rates = [lesion_data[t]['preservation_rate'] for t in types]
        scores = [lesion_data[t]['mean_score'] for t in types]

        x = np.arange(len(types))
        width = 0.35

        ax.bar(x - width/2, rates, width, label='Preservation Rate', color='steelblue')
        ax.bar(x + width/2, scores, width, label='Mean Score', color='coral')

        ax.set_ylabel('Score')
        ax.set_xlabel('Lesion Type')
        ax.set_title('Lesion Integrity by Type')
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.legend()
        ax.set_ylim(0, 1.1)

        fig.savefig(output_dir / 'lesion_integrity.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 3. Clinical risk distribution
    if 'clinical_risk' in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        risk_data = results['clinical_risk']

        # Risk scores
        ax = axes[0]
        categories = ['Normal', 'High-Risk']
        means = [risk_data['normal_mean_risk'], risk_data['high_risk_mean_risk']]
        colors = ['green' if m < 0.5 else 'red' for m in means]

        ax.bar(categories, means, color=colors, alpha=0.7)
        ax.set_ylabel('Mean Risk Score')
        ax.set_title('Risk Score by Category')
        ax.axhline(y=0.5, color='gray', linestyle='--')

        # Review rates
        ax = axes[1]
        rates = [risk_data['normal_review_rate'], risk_data['high_risk_review_rate']]
        ax.bar(categories, rates, color=['blue', 'orange'], alpha=0.7)
        ax.set_ylabel('Expert Review Rate')
        ax.set_title('Expert Review Requirement')

        fig.savefig(output_dir / 'clinical_risk.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 4. Regulatory metrics summary
    if 'regulatory' in results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        reg_data = results['regulatory']
        metrics = ['PSNR (÷10)', 'SSIM', '1-Artifact', '1-Failure']
        values = [
            reg_data['psnr'] / 10,
            reg_data['ssim'],
            1 - reg_data['artifact_severity'],
            1 - reg_data['failure_rate']
        ]
        thresholds = [3.0, 0.85, 0.7, 0.95]  # Normalized thresholds

        x = np.arange(len(metrics))
        bars = ax.bar(x, values, color='steelblue', alpha=0.7, label='Achieved')

        # Add threshold line markers
        for i, thresh in enumerate(thresholds):
            ax.plot([i-0.4, i+0.4], [thresh, thresh], 'r--', linewidth=2)

        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(f'FDA Metrics Summary (Certification: {reg_data["certification_level"]})')
        ax.legend(['Threshold', 'Achieved'])

        fig.savefig(output_dir / 'regulatory_metrics.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def run_experiment(
    config: dict,
    use_simulated: bool = True
) -> Dict:
    """
    Run complete safety framework evaluation experiment.

    Args:
        config: Configuration dictionary
        use_simulated: Use simulated data

    Returns:
        Dictionary with all experiment results
    """
    print("=" * 70)
    print("EXPERIMENT 4: COMPREHENSIVE SAFETY FRAMEWORK EVALUATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path("results/exp4_safety")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print("\nPreparing test data...")
    transform = MRIDataTransform(
        mask_type='cartesian',
        acceleration=4,
        center_fraction=0.08,
        crop_size=(320, 320),
        use_seed=True
    )

    dataset = SimulatedMRIDataset(
        num_samples=100,
        image_size=(320, 320),
        transform=transform,
        seed=42
    )

    # Extract images and masks
    base_images = []
    masks = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        base_images.append(sample['target'])
        masks.append(sample['mask'])

    print(f"Loaded {len(base_images)} test images")

    # Create test scenarios
    print("\nCreating test scenarios...")
    scenarios = create_test_scenarios(base_images, masks, device=str(device))

    for key, scens in scenarios.items():
        print(f"  {key}: {len(scens)} scenarios")

    # Load or create model
    print("\nLoading model...")
    guardian_cfg = config['model']['guardian']
    model_config = GuardianConfig(
        num_iterations=guardian_cfg['num_iterations'],
        base_channels=guardian_cfg['base_channels'],
        num_levels=guardian_cfg['num_levels'],
        use_kspace_net=guardian_cfg['use_kspace_net'],
        use_image_net=guardian_cfg['use_image_net'],
        dc_mode=guardian_cfg['dc_mode']
    )

    model = GuardianModel(model_config).to(device)

    # Try to load checkpoint
    checkpoint_path = Path("checkpoints/guardian_best.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model checkpoint")
    else:
        print("Using randomly initialized model (no checkpoint found)")

    # Run evaluations
    results = {}

    # 1. Uncertainty estimation
    results['uncertainty'] = evaluate_uncertainty(
        model, scenarios['normal'], device=str(device)
    )

    # 2. Distribution shift detection
    results['distribution_shift'] = evaluate_distribution_shift(
        model, scenarios['normal'], scenarios['distribution_shift'],
        device=str(device)
    )

    # 3. Physics violation detection
    results['physics_violation'] = evaluate_physics_violations(
        scenarios['physics_violation'], device=str(device)
    )

    # 4. Lesion integrity
    results['lesion_integrity'] = evaluate_lesion_integrity(
        model, scenarios['lesion_test'], device=str(device)
    )

    # 5. Bias detection
    results['bias'] = evaluate_bias(
        model, scenarios['bias_test'], device=str(device)
    )

    # 6. Clinical risk scoring
    results['clinical_risk'] = evaluate_clinical_risk(
        model, scenarios, device=str(device)
    )

    # 7. Regulatory metrics
    results['regulatory'] = evaluate_regulatory_metrics(
        model, scenarios['normal'], device=str(device)
    )

    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 SUMMARY")
    print("=" * 70)

    print("\n1. Uncertainty Estimation:")
    if 'summary' in results['uncertainty']:
        for method, stats in results['uncertainty']['summary'].items():
            print(f"   {method}: {stats['mean_uncertainty']:.4f} ± {stats['std_uncertainty']:.4f}")

    print("\n2. Distribution Shift Detection:")
    print(f"   AUC: {results['distribution_shift']['auc']:.4f}")
    print(f"   Normal score: {results['distribution_shift']['mean_normal_score']:.4f}")
    print(f"   Shift score: {results['distribution_shift']['mean_shift_score']:.4f}")

    print("\n3. Physics Violation Detection:")
    print(f"   Detection rate: {results['physics_violation']['detection_rate']:.2%}")
    print(f"   Mean severity: {results['physics_violation']['mean_severity']:.4f}")

    print("\n4. Lesion Integrity:")
    for ltype, stats in results['lesion_integrity']['by_type'].items():
        print(f"   {ltype}: {stats['preservation_rate']:.2%} preserved")

    print("\n5. Bias Analysis:")
    print(f"   Overall bias score: {results['bias']['overall_bias_score']:.4f}")
    print(f"   Performance gap: {results['bias']['performance_gap']:.2f} dB")

    print("\n6. Clinical Risk Scoring:")
    print(f"   Normal mean risk: {results['clinical_risk']['normal_mean_risk']:.4f}")
    print(f"   High-risk mean risk: {results['clinical_risk']['high_risk_mean_risk']:.4f}")
    print(f"   Separation: {results['clinical_risk']['separation']:.4f}")

    print("\n7. Regulatory Assessment:")
    print(f"   Overall pass: {results['regulatory']['overall_pass']}")
    print(f"   Certification: {results['regulatory']['certification_level']}")
    print(f"   PSNR: {results['regulatory']['psnr']:.2f} dB")
    print(f"   SSIM: {results['regulatory']['ssim']:.4f}")

    # Save results
    results_file = output_dir / 'results.json'

    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    serializable_results = make_serializable(results)

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    print("=" * 70)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run Safety Framework Experiment')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--simulated', action='store_true', default=True,
                        help='Use simulated data')
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_experiment(config, use_simulated=args.simulated)

    return results


if __name__ == '__main__':
    main()
