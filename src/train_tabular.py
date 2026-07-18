from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load_config, load_training_dataframe, project_path
from metrics import append_metrics_row, safe_binary_metrics, save_calibration_plot


def maybe_make_xgboost(scale_pos_weight: float):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        print(f"XGBoost not available; skipping. Import error: {exc}")
        return None

    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )


def make_models(y_train: np.ndarray) -> Dict[str, object]:
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = (n_neg / max(n_pos, 1)) if n_pos else 1.0

    models: Dict[str, object] = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "small_mlp": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    xgb = maybe_make_xgboost(scale_pos_weight=scale_pos_weight)
    if xgb is not None:
        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", xgb),
            ]
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    base_dir = Path(args.config).resolve().parent
    cfg = load_config(args.config)
    df = load_training_dataframe(cfg, base_dir=base_dir)

    feature_cols = cfg["data"]["numeric_features"]
    target_col = cfg["data"]["target_column"]
    n_splits = int(cfg["validation"]["n_splits"])

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)
    groups = df["case_id"].to_numpy(dtype=int)

    output_root = project_path(cfg["outputs"]["root_dir"], base_dir=base_dir) / "tabular"
    pred_dir = output_root / "predictions"
    model_dir = output_root / "models"
    plot_dir = output_root / "plots"
    for p in (pred_dir, model_dir, plot_dir):
        p.mkdir(parents=True, exist_ok=True)

    metrics_path = output_root / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    splitter = GroupKFold(n_splits=n_splits)
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        print(f"\nFold {fold}/{n_splits}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        models = make_models(y_train)

        for model_name, model in models.items():
            print(f"  Training {model_name}")
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = safe_binary_metrics(y_test, y_prob)
            row = {"fold": fold, "model": model_name, **metrics}
            append_metrics_row(metrics_path, row)

            fold_pred = df.iloc[test_idx][["id", "case_id", "rater_id", target_col]].copy()
            fold_pred["model"] = model_name
            fold_pred["fold"] = fold
            fold_pred["predicted_probability"] = y_prob
            fold_pred.to_csv(pred_dir / f"{model_name}_fold{fold}_predictions.csv", index=False)
            all_predictions.append(fold_pred)

            joblib.dump(model, model_dir / f"{model_name}_fold{fold}.joblib")
            save_calibration_plot(y_test, y_prob, plot_dir / f"{model_name}_fold{fold}_calibration.png")

    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    all_pred_df.to_csv(output_root / "all_predictions.csv", index=False)

    print(f"\nSaved tabular outputs to: {output_root}")
    print(f"Metrics file: {metrics_path}")


if __name__ == "__main__":
    main()
