from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def safe_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}
    metrics["threshold"] = float(threshold)
    metrics["n"] = int(len(y_true))
    metrics["prevalence"] = float(np.mean(y_true)) if len(y_true) else np.nan
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics.update(
        {
            "specificity": specificity,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )
    return metrics


def soft_target_metrics(y_true_soft, y_prob) -> Dict[str, float]:
    y_true_soft = np.asarray(y_true_soft).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    err = y_prob - y_true_soft
    return {
        "n": int(len(y_true_soft)),
        "target_mean": float(np.mean(y_true_soft)),
        "prediction_mean": float(np.mean(y_prob)),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "brier_soft": float(np.mean(err ** 2)),
    }


def save_calibration_plot(y_true, y_prob, path: str | Path, n_bins: int = 10) -> None:
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
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(xs, ys)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed error rate")
    plt.title("Calibration plot")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_roc_curve(y_true, y_prob, path: str | Path) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_pr_curve(y_true, y_prob, path: str | Path) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix(
    y_true,
    y_prob,
    path_prefix: str | Path,
    threshold: float = 0.5,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        path_prefix.with_suffix(".csv")
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, ["Pred 0", "Pred 1"])
    plt.yticks(ticks, ["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path_prefix.with_suffix(".png"), dpi=160)
    plt.close()


def append_metrics_row(path: str | Path, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def summarize_metrics(metrics_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "threshold",
        "n",
        "prevalence",
        "brier",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "specificity",
        "validation_loss",
    ]
    numeric_cols = [col for col in metric_cols if col in metrics_df.columns]
    summary_rows = []
    for keys, group in metrics_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in numeric_cols:
            values = group[col].dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)
