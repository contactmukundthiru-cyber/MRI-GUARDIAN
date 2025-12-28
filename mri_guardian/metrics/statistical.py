"""
Statistical Analysis for MRI-GUARDIAN

Provides statistical tests and confidence intervals
for rigorous hypothesis testing.

IMPORTANT FOR ISEF:
==================
Results need statistical significance!
- p < 0.05 is "significant"
- p < 0.01 is "highly significant"
- Always report confidence intervals
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from scipy import stats


@dataclass
class StatisticalResults:
    """Container for statistical test results."""
    statistic: float
    p_value: float
    significant: bool  # p < 0.05
    confidence_interval: Tuple[float, float]
    effect_size: Optional[float] = None
    test_name: str = ""


def paired_ttest(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alpha: float = 0.05
) -> StatisticalResults:
    """
    Paired t-test for comparing two methods on the same data.

    Use when: Comparing Guardian vs UNet on the SAME test images.

    Null hypothesis: Mean difference = 0
    Alternative: Mean difference ≠ 0

    Args:
        sample1: Metrics from method 1 (array of values)
        sample2: Metrics from method 2 (same samples)
        alpha: Significance level

    Returns:
        StatisticalResults with test outcome
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    # Paired t-test
    statistic, p_value = stats.ttest_rel(sample1, sample2)

    # Effect size (Cohen's d for paired samples)
    diff = sample1 - sample2
    effect_size = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

    # Confidence interval for the difference
    n = len(sample1)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)
    ci = (np.mean(diff) - t_crit * se, np.mean(diff) + t_crit * se)

    return StatisticalResults(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        confidence_interval=ci,
        effect_size=float(effect_size),
        test_name="Paired t-test"
    )


def wilcoxon_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alpha: float = 0.05
) -> StatisticalResults:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Use when: Data might not be normally distributed.

    More robust than t-test but less powerful.

    Args:
        sample1: Metrics from method 1
        sample2: Metrics from method 2
        alpha: Significance level

    Returns:
        StatisticalResults
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    statistic, p_value = stats.wilcoxon(sample1, sample2)

    # Effect size (rank-biserial correlation)
    n = len(sample1)
    effect_size = 1 - (2 * statistic) / (n * (n + 1))

    # CI via bootstrap for Wilcoxon
    ci = bootstrap_ci(sample1 - sample2, alpha=alpha)

    return StatisticalResults(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        confidence_interval=ci,
        effect_size=float(effect_size),
        test_name="Wilcoxon signed-rank test"
    )


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for the mean.

    Args:
        data: Sample data
        confidence: Confidence level (default 95%)

    Returns:
        (lower, upper) bounds of CI
    """
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval.

    Non-parametric CI estimation via resampling.
    Works for any statistic, not just the mean.

    Args:
        data: Sample data
        statistic: "mean", "median", or "std"
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (0.05 for 95% CI)

    Returns:
        (lower, upper) bounds of CI
    """
    data = np.asarray(data)
    n = len(data)

    # Bootstrap samples
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        if statistic == "mean":
            boot_stats.append(np.mean(sample))
        elif statistic == "median":
            boot_stats.append(np.median(sample))
        elif statistic == "std":
            boot_stats.append(np.std(sample, ddof=1))
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    boot_stats = np.array(boot_stats)

    # Percentile method
    lower = np.percentile(boot_stats, alpha/2 * 100)
    upper = np.percentile(boot_stats, (1 - alpha/2) * 100)

    return (float(lower), float(upper))


def anova_test(
    *groups: np.ndarray,
    alpha: float = 0.05
) -> StatisticalResults:
    """
    One-way ANOVA for comparing multiple methods.

    Use when: Comparing ZF-FFT vs UNet vs Guardian (3+ methods).

    Args:
        *groups: Multiple arrays of metrics
        alpha: Significance level

    Returns:
        StatisticalResults
    """
    statistic, p_value = stats.f_oneway(*groups)

    # Effect size (eta-squared)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    effect_size = ss_between / (ss_total + 1e-8)

    return StatisticalResults(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        confidence_interval=(0, 0),  # ANOVA doesn't have simple CI
        effect_size=float(effect_size),
        test_name="One-way ANOVA"
    )


def compare_methods(
    method_results: Dict[str, np.ndarray],
    reference_method: str,
    alpha: float = 0.05
) -> Dict[str, StatisticalResults]:
    """
    Compare multiple methods against a reference.

    Args:
        method_results: Dict mapping method names to metric arrays
        reference_method: Name of reference method
        alpha: Significance level

    Returns:
        Dict of StatisticalResults for each comparison
    """
    reference = method_results[reference_method]
    comparisons = {}

    for method_name, results in method_results.items():
        if method_name == reference_method:
            continue

        # Use Wilcoxon for robustness
        comparison = wilcoxon_test(results, reference, alpha)
        comparisons[f"{method_name}_vs_{reference_method}"] = comparison

    return comparisons


def compute_summary_statistics(
    data: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive summary statistics.

    Args:
        data: Sample data

    Returns:
        Dict with statistics
    """
    data = np.asarray(data)

    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)),
        'median': float(np.median(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q1': float(np.percentile(data, 25)),
        'q3': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
        'n': len(data),
        'ci_lower': compute_confidence_interval(data)[0],
        'ci_upper': compute_confidence_interval(data)[1],
    }


def format_results_for_paper(
    metric_name: str,
    method_stats: Dict[str, Dict[str, float]],
    comparison_results: Dict[str, StatisticalResults]
) -> str:
    """
    Format results for inclusion in paper/poster.

    Args:
        metric_name: Name of the metric
        method_stats: Summary statistics per method
        comparison_results: Statistical comparison results

    Returns:
        Formatted string for publication
    """
    lines = [f"\n{metric_name.upper()} Results:", "=" * 50]

    # Summary table
    lines.append("\nMethod           Mean ± Std        95% CI")
    lines.append("-" * 50)

    for method, stats in method_stats.items():
        ci = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
        lines.append(f"{method:15} {stats['mean']:.3f} ± {stats['std']:.3f}   {ci}")

    # Statistical comparisons
    lines.append("\nStatistical Comparisons:")
    lines.append("-" * 50)

    for comp_name, result in comparison_results.items():
        sig = "*" if result.significant else ""
        lines.append(f"{comp_name}: p={result.p_value:.4f}{sig}, d={result.effect_size:.2f}")

    lines.append("\n* p < 0.05")

    return "\n".join(lines)


def check_normality(
    data: np.ndarray,
    alpha: float = 0.05
) -> Tuple[bool, float]:
    """
    Check if data is normally distributed (Shapiro-Wilk test).

    Important for deciding between t-test vs Wilcoxon.

    Args:
        data: Sample data
        alpha: Significance level

    Returns:
        is_normal: Whether data passes normality test
        p_value: P-value of the test
    """
    data = np.asarray(data)

    if len(data) < 3:
        return True, 1.0  # Can't test with < 3 samples

    statistic, p_value = stats.shapiro(data)
    is_normal = p_value > alpha

    return is_normal, float(p_value)
