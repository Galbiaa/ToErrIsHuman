from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from data import load_config, project_path
from experiment_output import get_classification_threshold, get_random_seed

REQUIRED_COLS = ("case_id", "rater_id", "fold", "y_true", "y_probability", "y_pred")
BOOTSTRAP_METRICS = ("auroc", "auprc", "f1", "brier", "balanced_accuracy", "specificity")
TABULAR_CANDIDATES = ("xgboost", "logistic_regression", "random_forest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clustered (case_id) bootstrap comparison of OOF predictions."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outputs-root", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _read_oof(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF file for {model_name}: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df = df[list(REQUIRED_COLS)].copy()
    df["model"] = model_name
    df["case_id"] = df["case_id"].astype(int)
    df["rater_id"] = df["rater_id"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["y_true"] = df["y_true"].astype(int)
    df["y_probability"] = df["y_probability"].astype(float)
    return df.sort_values(["case_id", "rater_id", "fold"]).reset_index(drop=True)


def discover_models(outputs_root: Path) -> Dict[str, pd.DataFrame]:
    models: Dict[str, pd.DataFrame] = {}
    tabular_dir = outputs_root / "tabular" / "oof"
    for name in TABULAR_CANDIDATES:
        path = tabular_dir / f"{name}_predictions.csv"
        if path.exists():
            models[name] = _read_oof(path, name)

    image_path = outputs_root / "image_only" / "oof" / "predictions.csv"
    if image_path.exists():
        models["image_only"] = _read_oof(image_path, "image_only")

    multi_path = outputs_root / "multimodal" / "oof" / "predictions.csv"
    if multi_path.exists():
        models["multimodal"] = _read_oof(multi_path, "multimodal")

    reg_path = outputs_root / "image_only_regression" / "oof" / "predictions.csv"
    if reg_path.exists():
        models["image_only_regression"] = _read_oof(reg_path, "image_only_regression")

    if not models:
        raise SystemExit(f"No OOF prediction files found under {outputs_root}")
    return models


def binary_metrics_np(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, float] = {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return out


def point_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    return binary_metrics_np(df["y_true"].to_numpy(), df["y_probability"].to_numpy(), threshold)


def select_best_tabular(models: Dict[str, pd.DataFrame], threshold: float) -> Optional[str]:
    best_name = None
    best_auroc = -np.inf
    for name in TABULAR_CANDIDATES:
        if name not in models:
            continue
        auroc = point_metrics(models[name], threshold)["auroc"]
        if np.isfinite(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_name = name
    return best_name


def align_keys(models: Dict[str, pd.DataFrame], names: Sequence[str]) -> None:
    ref = None
    ref_name = None
    for name in names:
        df = models[name]
        key = list(zip(df["case_id"].tolist(), df["rater_id"].tolist()))
        if ref is None:
            ref = key
            ref_name = name
            continue
        if key != ref:
            raise ValueError(
                f"OOF rows for '{name}' are not aligned with '{ref_name}' "
                f"(case_id, rater_id order/content differ)."
            )


def build_case_index(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Return unique case_ids and mapping case_id -> row indices into df."""
    case_ids = df["case_id"].to_numpy(dtype=int)
    unique = np.array(sorted(pd.unique(case_ids)), dtype=int)
    index: Dict[int, np.ndarray] = {}
    for cid in unique:
        index[int(cid)] = np.flatnonzero(case_ids == cid)
    return unique, index


def metrics_from_indices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    case_index: Dict[int, np.ndarray],
    sampled_cases: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    # Concatenate row indices for sampled cases (with replacement → duplicate rows OK).
    parts = [case_index[int(cid)] for cid in sampled_cases]
    idx = np.concatenate(parts)
    return binary_metrics_np(y_true[idx], y_prob[idx], threshold)


def clustered_bootstrap(
    models: Dict[str, pd.DataFrame],
    model_names: Sequence[str],
    pairs: Sequence[Tuple[str, str]],
    n_bootstrap: int,
    threshold: float,
    seed: int,
    ci: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    align_keys(models, model_names)
    ref_df = models[model_names[0]]
    case_ids, _ = build_case_index(ref_df)
    n_cases = len(case_ids)
    rng = np.random.default_rng(seed)

    arrays = {}
    indices = {}
    for name in model_names:
        df = models[name]
        arrays[name] = (
            df["y_true"].to_numpy(dtype=int),
            df["y_probability"].to_numpy(dtype=float),
        )
        _, indices[name] = build_case_index(df)

    metric_store = {name: {m: np.empty(n_bootstrap, dtype=float) for m in BOOTSTRAP_METRICS} for name in model_names}
    delta_store = {pair: {m: np.empty(n_bootstrap, dtype=float) for m in BOOTSTRAP_METRICS} for pair in pairs}

    for b in range(n_bootstrap):
        sampled = rng.choice(case_ids, size=n_cases, replace=True)
        per_model: Dict[str, Dict[str, float]] = {}
        for name in model_names:
            y_true, y_prob = arrays[name]
            mets = metrics_from_indices(y_true, y_prob, indices[name], sampled, threshold)
            per_model[name] = mets
            for m in BOOTSTRAP_METRICS:
                metric_store[name][m][b] = mets[m]

        for a, b_name in pairs:
            if a not in per_model or b_name not in per_model:
                continue
            for m in BOOTSTRAP_METRICS:
                delta_store[(a, b_name)][m][b] = per_model[a][m] - per_model[b_name][m]

    alpha = (1.0 - ci) / 2.0
    boot_rows: List[Dict] = []
    for name in model_names:
        for m in BOOTSTRAP_METRICS:
            arr = metric_store[name][m]
            boot_rows.append(
                {
                    "model": name,
                    "metric": m,
                    "n_bootstrap": n_bootstrap,
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr, ddof=1)) if n_bootstrap > 1 else 0.0,
                    "ci_low": float(np.nanquantile(arr, alpha)),
                    "ci_high": float(np.nanquantile(arr, 1.0 - alpha)),
                }
            )

    delta_rows: List[Dict] = []
    for a, b_name in pairs:
        for m in BOOTSTRAP_METRICS:
            arr = delta_store[(a, b_name)][m]
            prop_le = float(np.mean(arr <= 0))
            prop_ge = float(np.mean(arr >= 0))
            p_value = float(min(1.0, 2.0 * min(prop_le, prop_ge)))
            delta_rows.append(
                {
                    "model_a": a,
                    "model_b": b_name,
                    "metric": m,
                    "delta_mean": float(np.nanmean(arr)),
                    "delta_std": float(np.nanstd(arr, ddof=1)) if n_bootstrap > 1 else 0.0,
                    "ci_low": float(np.nanquantile(arr, alpha)),
                    "ci_high": float(np.nanquantile(arr, 1.0 - alpha)),
                    "p_boot": p_value,
                    "n_bootstrap": n_bootstrap,
                }
            )

    return pd.DataFrame(boot_rows), pd.DataFrame(delta_rows)


def build_comparison_pairs(
    models: Dict[str, pd.DataFrame],
    best_tabular: Optional[str],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if best_tabular and "image_only" in models:
        pairs.append((best_tabular, "image_only"))
    if "image_only" in models and "multimodal" in models:
        pairs.append(("image_only", "multimodal"))
    if best_tabular and "multimodal" in models:
        pairs.append((best_tabular, "multimodal"))
    if "image_only" in models and "image_only_regression" in models:
        pairs.append(("image_only", "image_only_regression"))
    return pairs


def main() -> None:
    args = parse_args()
    base_dir = Path(args.config).resolve().parent
    cfg = load_config(args.config)
    outputs_root = (
        project_path(args.outputs_root, base_dir=base_dir)
        if args.outputs_root
        else project_path(cfg.get("outputs", {}).get("root_dir", "outputs"), base_dir=base_dir)
    )
    threshold = get_classification_threshold(cfg)
    seed = int(args.seed) if args.seed is not None else get_random_seed(cfg)
    n_boot = int(args.n_bootstrap)
    ci = float(args.ci)

    models = discover_models(outputs_root)
    best_tabular = select_best_tabular(models, threshold)

    point_rows = []
    for name, df in models.items():
        row = {"model": name, "n": len(df), "n_cases": int(df["case_id"].nunique())}
        row.update(point_metrics(df, threshold))
        row["is_best_tabular"] = bool(best_tabular is not None and name == best_tabular)
        point_rows.append(row)
    point_df = pd.DataFrame(point_rows).sort_values("auroc", ascending=False).reset_index(drop=True)

    pairs = build_comparison_pairs(models, best_tabular)
    model_names = []
    for name in list(TABULAR_CANDIDATES) + ["image_only", "multimodal", "image_only_regression"]:
        if name in models:
            model_names.append(name)

    print(f"Loaded models: {', '.join(sorted(models.keys()))}")
    if best_tabular:
        print(f"Best tabular (by OOF AUROC): {best_tabular}")
    print(f"Clustered bootstrap: n={n_boot}, cases={int(point_df['n_cases'].iloc[0])}, seed={seed}, CI={ci}")
    if "image_only_regression" not in models:
        print("Note: image_only_regression OOF not found - skipping that comparison.")

    boot_df, delta_df = clustered_bootstrap(
        models=models,
        model_names=model_names,
        pairs=pairs,
        n_bootstrap=n_boot,
        threshold=threshold,
        seed=seed,
        ci=ci,
    )

    out_dir = outputs_root / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    point_path = out_dir / "point_metrics.csv"
    boot_path = out_dir / "bootstrap_summary.csv"
    delta_path = out_dir / "paired_deltas.csv"
    meta_path = out_dir / "run_metadata.json"

    point_df.to_csv(point_path, index=False)
    boot_df.to_csv(boot_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_bootstrap": n_boot,
        "ci_level": ci,
        "random_seed": seed,
        "classification_threshold": threshold,
        "best_tabular": best_tabular,
        "models": sorted(models.keys()),
        "pairs": [{"model_a": a, "model_b": b} for a, b in pairs],
        "cluster_unit": "case_id",
        "outputs_root": str(outputs_root),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved: {point_path}")
    print(f"Saved: {boot_path}")
    print(f"Saved: {delta_path}")
    print(f"Saved: {meta_path}")

    print("\nPoint AUROC:")
    for _, row in point_df.iterrows():
        flag = " (best tabular)" if row.get("is_best_tabular") else ""
        print(f"  {row['model']}: {row['auroc']:.4f}{flag}")

    if not delta_df.empty:
        print("\nPaired AUROC deltas (A - B), clustered bootstrap CI:")
        for _, row in delta_df[delta_df["metric"] == "auroc"].iterrows():
            print(
                f"  {row['model_a']} - {row['model_b']}: "
                f"{row['delta_mean']:+.4f} "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
                f"p_boot={row['p_boot']:.3f}"
            )


if __name__ == "__main__":
    main()
