from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from metrics import (
    append_metrics_row,
    safe_binary_metrics,
    save_calibration_plot,
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
    summarize_metrics,
)


def get_classification_threshold(config: dict) -> float:
    return float(config.get("evaluation", {}).get("classification_threshold", 0.5))


def get_random_seed(config: dict) -> int:
    return int(config.get("evaluation", {}).get("random_seed", config.get("validation", {}).get("random_state", 42)))


def save_config_snapshot(config_path: str | Path, output_root: str | Path) -> Path:
    config_path = Path(config_path)
    destination = Path(output_root) / "config_used.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination)
    return destination


def write_run_metadata(
    output_root: str | Path,
    config: dict,
    experiment_name: str,
    extra: Optional[Dict] = None,
) -> Path:
    metadata = {
        "experiment": experiment_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": get_random_seed(config),
        "classification_threshold": get_classification_threshold(config),
        "n_splits": int(config.get("validation", {}).get("n_splits", 5)),
        "folds_file": config.get("validation", {}).get("folds_file", "outputs/folds/fold_assignments.csv"),
    }
    if extra:
        metadata.update(extra)

    path = Path(output_root) / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def build_oof_dataframe(
    case_id: Iterable[int],
    rater_id: Iterable[int],
    fold: int,
    y_true: Iterable[int],
    y_probability: Iterable[float],
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_prob = np.asarray(y_probability, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=int)
    return pd.DataFrame(
        {
            "case_id": np.asarray(case_id, dtype=int),
            "rater_id": np.asarray(rater_id, dtype=int),
            "fold": int(fold),
            "y_true": y_true_arr,
            "y_probability": y_prob,
            "y_pred": (y_prob >= threshold).astype(int),
        }
    )


def save_oof_predictions(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["case_id", "rater_id", "fold", "y_true", "y_probability", "y_pred"]
    df[columns].to_csv(path, index=False)


def evaluate_binary_fold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    plot_prefix: str | Path,
    threshold: float = 0.5,
) -> Dict[str, float]:
    plot_prefix = Path(plot_prefix)
    metrics = safe_binary_metrics(y_true, y_prob, threshold=threshold)
    save_calibration_plot(y_true, y_prob, plot_prefix.with_name(f"{plot_prefix.name}_calibration.png"))
    save_roc_curve(y_true, y_prob, plot_prefix.with_name(f"{plot_prefix.name}_roc.png"))
    save_pr_curve(y_true, y_prob, plot_prefix.with_name(f"{plot_prefix.name}_pr.png"))
    save_confusion_matrix(y_true, y_prob, plot_prefix.with_name(f"{plot_prefix.name}_confusion"), threshold=threshold)
    return metrics


def write_metrics_summary(metrics_df: pd.DataFrame, path: str | Path, group_cols: List[str]) -> pd.DataFrame:
    summary = summarize_metrics(metrics_df, group_cols=group_cols)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def init_experiment_directory(output_root: str | Path) -> Dict[str, Path]:
    output_root = Path(output_root)
    paths = {
        "root": output_root,
        "predictions": output_root / "predictions",
        "oof": output_root / "oof",
        "models": output_root / "models",
        "plots": output_root / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
