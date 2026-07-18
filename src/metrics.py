from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # non-interactive; avoid Tk crashes when only saving figures
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
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}
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
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
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


def aggregate_case_level_predictions(
    pred_df: pd.DataFrame,
    case_col: str = "case_id",
    target_col: str = "error_rating",
    prob_col: str = "predicted_probability_error",
) -> pd.DataFrame:
    """Average decision-level labels and probabilities within each case_id."""
    return (
        pred_df.groupby(case_col, as_index=False)
        .agg(
            n_ratings=(target_col, "size"),
            target_mean_error=(target_col, "mean"),
            predicted_probability_mean_error=(prob_col, "mean"),
        )
        .sort_values(case_col)
        .reset_index(drop=True)
    )


def case_level_soft_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    """Secondary metrics after aggregating replicated decisions by case_id."""
    case_df = aggregate_case_level_predictions(pred_df)
    metrics = soft_target_metrics(
        case_df["target_mean_error"],
        case_df["predicted_probability_mean_error"],
    )
    metrics["n_cases"] = metrics.pop("n")
    return metrics


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


def append_metrics_row(path: str | Path, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
