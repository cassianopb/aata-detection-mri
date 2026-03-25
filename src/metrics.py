from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute a standard set of binary classification performance metrics.

    Returns accuracy, AUC, F1-score, sensitivity (recall), specificity,
    positive predictive value (PPV), negative predictive value (NPV),
    positive likelihood ratio (PLR), and negative likelihood ratio (NLR).
    """
    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_score = 0.0
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    plr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else 0.0
    nlr = (1 - sensitivity) / specificity if specificity > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "AUC": auc_score,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "PLR": plr,
        "NLR": nlr,
    }


def bootstrap_metrics(
    y_true,
    y_pred,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute 95% bootstrap confidence intervals for all metrics in compute_metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        n_bootstraps: Number of bootstrap resampling iterations.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping each metric name to a (lower_bound, upper_bound) tuple.
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    samples = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        samples.append(compute_metrics(y_true[idx], y_pred[idx]))

    ci = {}
    for key in samples[0]:
        values = np.array([s[key] for s in samples])
        ci[key] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
    return ci


def print_metrics_table(point_estimates: dict, confidence_intervals: dict) -> None:
    """Print a formatted table of point estimates with 95% confidence intervals."""
    header = f"{'Metric':<14} {'Estimate':>10}   {'95% CI':>20}"
    print(header)
    print("-" * len(header))
    for key, value in point_estimates.items():
        lo, hi = confidence_intervals[key]
        print(f"{key:<14} {value:>10.4f}   [{lo:.4f}, {hi:.4f}]")


def bootstrap_roc(
    y_true,
    y_score,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple:
    """
    Estimate a mean ROC curve with 95% bootstrap confidence interval.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities for the positive class.
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (base_fpr, mean_tpr, tpr_lower, tpr_upper, mean_auc, std_auc).
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n = len(y_true)
    base_fpr = np.linspace(0, 1, 101)
    tprs, aucs = [], []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[idx], y_score[idx])
        aucs.append(auc(fpr, tpr))
        interp_tpr = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    tprs = np.array(tprs)
    return (
        base_fpr,
        tprs.mean(axis=0),
        np.percentile(tprs, 2.5, axis=0),
        np.percentile(tprs, 97.5, axis=0),
        float(np.mean(aucs)),
        float(np.std(aucs)),
    )


def plot_roc(
    y_true,
    y_score,
    title: str = "ROC Curve",
    zoom: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a ROC curve with 95% bootstrap confidence band.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities for the positive class.
        title: Figure title.
        zoom: If True, restricts axes to the high-performance region
              (FPR in [0, 0.2], TPR in [0.8, 1.0]).
        save_path: Optional path to save the figure (300 dpi).
    """
    base_fpr, mean_tpr, tpr_lower, tpr_upper, mean_auc, _ = bootstrap_roc(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(base_fpr, mean_tpr, color="steelblue", lw=2, label=f"AUC = {mean_auc:.4f}")
    ax.fill_between(base_fpr, tpr_lower, tpr_upper, color="steelblue", alpha=0.15, label="95% CI")

    if zoom:
        ax.set_xlim([0, 0.2])
        ax.set_ylim([0.8, 1.005])
    else:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.01])

    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
