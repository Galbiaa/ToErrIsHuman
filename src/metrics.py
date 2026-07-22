"""Binary classification metrics, calibration, and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from methodology_guards import majority_class_baseline

# Probabilities are model scores, not automatically "clinical percentages".
CALIBRATION_INTERPRETATION_NOTE = (
    "Calibration plots and slopes describe agreement between predicted scores and "
    "observed frequencies. Do not treat raw probabilities as clinically reliable "
    "percentages without assessing calibration."
)


def safe_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}
    metrics["n"] = int(len(y_true))
    metrics["prevalence"] = float(np.mean(y_true)) if len(y_true) else np.nan
    metrics["threshold"] = float(threshold)
    metrics["brier"] = float(brier_score_loss(y_true, y_prob)) if len(y_true) else np.nan
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan
    metrics["balanced_accuracy"] = (
        float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else np.nan
    )
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    maj = majority_class_baseline(y_true)
    metrics["majority_class"] = float(maj["majority_class"]) if maj["n"] else np.nan
    metrics["majority_accuracy"] = float(maj["majority_accuracy"]) if maj["n"] else np.nan

    if len(y_true) and len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        # clip for numerical stability in log-loss
        p_clip = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
        metrics["log_loss"] = float(log_loss(y_true, p_clip, labels=[0, 1]))
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan
        metrics["log_loss"] = np.nan

    if len(y_true):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    else:
        metrics.update({"tn": 0, "fp": 0, "fn": 0, "tp": 0, "specificity": np.nan})
    return metrics


def calibration_slope_intercept(y_true, y_prob) -> Dict[str, float]:
    """
    Fit observed labels ~ predicted probabilities (linear).
    Ideal calibration: slope≈1, intercept≈0.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1, 1)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return {"calibration_slope": np.nan, "calibration_intercept": np.nan}
    reg = LinearRegression()
    reg.fit(y_prob, y_true)
    return {
        "calibration_slope": float(reg.coef_[0]),
        "calibration_intercept": float(reg.intercept_),
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE over equal-width probability bins (interpret with caution on small n)."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def binary_metrics_with_calibration(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Convenience: discrimination metrics + calibration summaries."""
    metrics = safe_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics.update(calibration_slope_intercept(y_true, y_prob))
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    return metrics


def save_calibration_plot(
    y_true,
    y_prob,
    path: str | Path,
    n_bins: int = 10,
    title: str = "Calibration plot",
) -> None:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True)
    xs, ys = [], []
    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        if np.any(mask):
            xs.append(float(np.mean(y_prob[mask])))
            ys.append(float(np.mean(y_true[mask])))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    plt.scatter(xs, ys, label="Binned")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def append_metrics_row(path: str | Path, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
