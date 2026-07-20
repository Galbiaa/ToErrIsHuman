from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import (
    attach_fold_column,
    iter_group_fold_splits,
    load_config,
    load_fold_assignments,
    load_training_dataframe,
    prepare_fold_feature_frames,
    project_path,
    resolve_folds_path,
)
from experiment_output import (
    build_oof_dataframe,
    evaluate_binary_fold,
    get_classification_threshold,
    init_experiment_directory,
    save_config_snapshot,
    save_oof_predictions,
    write_metrics_summary,
    write_run_metadata,
)
from metrics import append_metrics_row


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
    df = load_training_dataframe(cfg, base_dir=base_dir).reset_index(drop=True)
    folds_path = resolve_folds_path(cfg, base_dir=base_dir)
    fold_assignments = load_fold_assignments(folds_path)
    attach_fold_column(df, fold_assignments)

    feature_cols = cfg["data"]["numeric_features"]
    target_col = cfg["data"]["target_column"]
    n_splits = int(cfg["validation"]["n_splits"])
    threshold = get_classification_threshold(cfg)

    output_root = project_path(cfg["outputs"]["root_dir"], base_dir=base_dir) / "tabular"
    paths = init_experiment_directory(output_root)
    save_config_snapshot(args.config, output_root)
    write_run_metadata(output_root, cfg, experiment_name="tabular")

    fold_stats_path = project_path("outputs/folds/fold_stats.csv", base_dir=base_dir)
    if fold_stats_path.exists():
        shutil.copy2(fold_stats_path, output_root / "fold_stats.csv")

    metrics_path = output_root / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    oof_by_model: Dict[str, list[pd.DataFrame]] = {}

    for fold, train_idx, test_idx in iter_group_fold_splits(df, fold_assignments, n_splits):
        print(f"\nFold {fold}/{n_splits}")
        train_df, test_df = prepare_fold_feature_frames(df, train_idx, test_idx, cfg)
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        X_test = test_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=int)
        y_test = test_df[target_col].to_numpy(dtype=int)
        models = make_models(y_train)

        for model_name, model in models.items():
            print(f"  Training {model_name}")
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            plot_prefix = paths["plots"] / f"{model_name}_fold{fold}"
            metrics = evaluate_binary_fold(y_test, y_prob, plot_prefix=plot_prefix, threshold=threshold)
            append_metrics_row(metrics_path, {"fold": fold, "model": model_name, **metrics})

            oof_df = build_oof_dataframe(
                case_id=test_df["case_id"],
                rater_id=test_df["rater_id"],
                fold=fold,
                y_true=y_test,
                y_probability=y_prob,
                threshold=threshold,
            )
            oof_by_model.setdefault(model_name, []).append(oof_df)
            oof_df.to_csv(paths["predictions"] / f"{model_name}_fold{fold}_predictions.csv", index=False)

            joblib.dump(model, paths["models"] / f"{model_name}_fold{fold}.joblib")

    metrics_df = pd.read_csv(metrics_path)
    write_metrics_summary(metrics_df, output_root / "metrics_summary.csv", group_cols=["model"])

    for model_name, fold_frames in oof_by_model.items():
        model_oof = pd.concat(fold_frames, ignore_index=True)
        save_oof_predictions(model_oof, paths["oof"] / f"{model_name}_predictions.csv")

    print(f"\nSaved tabular outputs to: {output_root}")
    print(f"Metrics file: {metrics_path}")
    print(f"Classification threshold: {threshold}")


if __name__ == "__main__":
    main()
