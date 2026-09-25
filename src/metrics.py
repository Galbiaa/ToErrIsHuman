"""Binary classification metrics, calibration, and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import matplotlib

# Non-interactive backend: avoids GUI crashes on Windows / headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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


def brier_skill_score(y_true, y_prob) -> float:
    """Brier Skill Score relative to the prevalence-only (climatological) baseline.

    BSS = 1 - Brier_model / Brier_reference
    where Brier_reference = mean(y) * (1 - mean(y)) is the Brier score of the
    constant prediction p = prevalence. BSS > 0 means better than predicting the
    prevalence for every case; BSS <= 0 means worse or no better.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    brier_model = brier_score_loss(y_true, y_prob)
    prevalence = float(np.mean(y_true)) if len(y_true) else np.nan
    brier_ref = prevalence * (1.0 - prevalence)
    if not np.isfinite(brier_ref) or brier_ref == 0.0:
        return float("nan")
    return float(1.0 - (brier_model / brier_ref))


def binary_metrics_with_calibration(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Convenience: discrimination metrics + calibration summaries."""
    metrics = safe_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics.update(calibration_slope_intercept(y_true, y_prob))
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    return metrics


def confusion_counts(y_true, y_pred) -> Dict[str, int]:
    """TN/FP/FN/TP for binary labels."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def binary_metrics_with_rowwise_threshold(
    y_true,
    y_prob,
    thresholds: Sequence[float],
    *,
    threshold_note: str = "rowwise_validation_youden_per_test_fold",
) -> Dict[str, float]:
    """
    Pooled metrics when each row uses its own threshold (e.g. OOF @ fold validation Youden).
    Threshold-free metrics (AUROC, Brier, …) use probabilities; hard-cut metrics use row-wise preds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thr = np.asarray(thresholds, dtype=float)
    if len(y_true) != len(y_prob) or len(y_true) != len(thr):
        raise ValueError("y_true, y_prob, and thresholds must have the same length")
    y_pred = (y_prob >= thr).astype(int)

    metrics: Dict[str, float] = {}
    metrics["n"] = int(len(y_true))
    metrics["prevalence"] = float(np.mean(y_true)) if len(y_true) else np.nan
    metrics["threshold"] = float(np.mean(thr)) if len(thr) else np.nan
    metrics["threshold_note"] = threshold_note
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
        p_clip = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
        metrics["log_loss"] = float(log_loss(y_true, p_clip, labels=[0, 1]))
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan
        metrics["log_loss"] = np.nan

    counts = confusion_counts(y_true, y_pred)
    metrics.update(counts)
    tn, fp = counts["tn"], counts["fp"]
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    metrics.update(calibration_slope_intercept(y_true, y_prob))
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    return metrics


def save_confusion_matrix_plot(
    y_true,
    y_prob,
    path: str | Path,
    *,
    threshold: float = 0.5,
    thresholds: Optional[Sequence[float]] = None,
    title: str = "Confusion matrix",
) -> Dict[str, int]:
    """Save a 2×2 confusion-matrix figure; returns TN/FP/FN/TP."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if thresholds is not None:
        thr = np.asarray(thresholds, dtype=float)
        y_pred = (y_prob >= thr).astype(int)
        subtitle = f"row-wise validation Youden (mean thr={float(np.mean(thr)):.3f})"
    else:
        y_pred = (y_prob >= threshold).astype(int)
        subtitle = f"threshold = {threshold:.3f}"

    counts = confusion_counts(y_true, y_pred)
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=int)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.8))
    im = plt.imshow(matrix, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_labels = ["Pred 0", "Pred 1"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (row, col), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "black"
        plt.text(col, row, str(value), ha="center", va="center", color=color, fontsize=14)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{title}\n{subtitle}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return counts


def save_confusion_matrix_csv(
    rows: Iterable[Dict[str, Union[str, int, float]]],
    path: str | Path,
) -> None:
    """Write one or more confusion-matrix summaries (split, threshold, TN/FP/FN/TP, …)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def write_metrics_table(path: str | Path, rows: Iterable[Dict]) -> None:
    """Overwrite a metrics CSV with the given rows (stable column order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if len(df):
        df.to_csv(path, index=False)


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


def save_roc_curve(y_true, y_prob, path: str | Path, title: str = "ROC curve") -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = float(roc_auc_score(y_true, y_prob))
        plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    else:
        plt.text(0.5, 0.5, "ROC undefined (single class)", ha="center", va="center")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_pr_curve(y_true, y_prob, path: str | Path, title: str = "Precision–Recall curve") -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    prevalence = float(np.mean(y_true)) if len(y_true) else 0.0
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = float(average_precision_score(y_true, y_prob))
        plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    else:
        plt.text(0.5, 0.5, "PR undefined (single class)", ha="center", va="center")
    plt.hlines(prevalence, 0, 1, linestyles="--", colors="gray", label=f"Prevalence = {prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
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


def reset_metrics_file(path: str | Path) -> None:
    """Truncate an append-based metrics CSV at the start of a full run.

    append_metrics_row() accumulates rows across runs; call this once at the
    start of a script's full run so every re-run produces a clean, fresh table
    (no duplicated rows from previous runs).
    """
    path = Path(path)
    if path.is_file():
        path.unlink()
