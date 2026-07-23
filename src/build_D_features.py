"""Fold-safe feature builder for solution D (no XGBoost fit here).

Builds the closed allowlist of D features from:
- USER decisions (without TARGET)
- case-level AI probabilities p_i with an explicit probability_source tag
- fold-safe rater-accuracy / rater-confidence (LOO on train; train-only on val/test)

The preprocessor (median impute + standardize) is defined here; fit/save per fold
happens in train_D.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from build_baseline import AI_FEATURE_COLS, compute_ai_decision_features, load_oof_predictions
from data import load_config, load_user_infos, project_path
from generate_folds import load_fold_assignment
from methodology_guards import assert_d_features, fold_safe_rater_stats, tag_probability_source
from reproducibility import bootstrap_run_reproducibility, set_global_seed

META_COLS = [
    "id",
    "case_id",
    "rater_id",
    "fold",
    "error-rating",
    "probability_source",
    "ai_decision_threshold",
]

USER_CONTEXT_COLS = [
    "rating-confidence",
    "case-difficulty",
    "rater-expertise",
    "rating",
]


def d_feature_allowlist(config: dict) -> List[str]:
    feats = list(config["solution_d"]["features"])
    forbidden = set(config["solution_d"].get("forbidden_features", ["TARGET"]))
    if any(f in forbidden or f.upper() == "TARGET" for f in feats):
        raise ValueError("D allowlist contains a forbidden feature name")
    return feats


def make_d_preprocessor() -> Pipeline:
    """Median imputer + standard scaler. Fit only on training rows (caller responsibility)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def fit_transform_d_features(
    preprocessor: Pipeline,
    X_train: np.ndarray | pd.DataFrame,
    X_other: np.ndarray | pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, Pipeline]:
    """Fit preprocessor on train; optionally transform another split with the same fit."""
    X_tr = np.asarray(X_train, dtype=float)
    preprocessor.fit(X_tr)
    X_tr_t = preprocessor.transform(X_tr)
    if X_other is None:
        return X_tr_t, None, preprocessor
    X_ot = np.asarray(X_other, dtype=float)
    return X_tr_t, preprocessor.transform(X_ot), preprocessor


def save_preprocessor(preprocessor: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(path: str | Path) -> Pipeline:
    return joblib.load(path)


def _strip_target(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c == "TARGET" or c.upper() == "TARGET"]
    if drop:
        return df.drop(columns=drop)
    return df


def build_d_feature_frame(
    user: pd.DataFrame,
    case_probabilities: pd.DataFrame,
    train_case_ids: Iterable[int],
    *,
    config: dict,
    probability_source: str,
    decision_threshold: float = 0.5,
    fold_col: str | None = "fold",
) -> pd.DataFrame:
    """
    Build one decision-level frame with meta + exactly the D allowlist features.

    ``case_probabilities`` must contain case_id and ai_probability_class_1
    (optionally fold). ``train_case_ids`` defines fold-safe rater stats.
    """
    source = tag_probability_source(probability_source)
    allow = d_feature_allowlist(config)
    forbidden = list(config["solution_d"].get("forbidden_features", ["TARGET"]))

    user_clean = _strip_target(user)
    if "TARGET" in user_clean.columns:
        raise ValueError("TARGET must not be present when building D features")

    required_user = {
        "id",
        "case_id",
        "rater_id",
        "error-rating",
        *USER_CONTEXT_COLS,
    }
    missing_u = required_user - set(user_clean.columns)
    if missing_u:
        raise ValueError(f"USER table missing columns for D features: {sorted(missing_u)}")

    probs = case_probabilities.copy()
    if "ai_probability_class_1" not in probs.columns or "case_id" not in probs.columns:
        raise ValueError("case_probabilities needs case_id and ai_probability_class_1")
    probs["case_id"] = pd.to_numeric(probs["case_id"], errors="raise").astype(int)
    probs["ai_probability_class_1"] = pd.to_numeric(
        probs["ai_probability_class_1"], errors="raise"
    ).astype(float)
    if probs["case_id"].duplicated().any():
        raise ValueError("case_probabilities has duplicate case_id")

    # Fold-safe rater stats on the full USER table (needs all train decisions).
    rated = fold_safe_rater_stats(user_clean, train_case_ids)
    rated = rated.copy()
    # Overwrite Excel globals with fold-safe values under allowlist names.
    rated["rater-accuracy"] = rated["rater_accuracy_foldsafe"]
    rated["rater-confidence"] = rated["rater_confidence_foldsafe"]

    merge_cols = ["case_id", "ai_probability_class_1"]
    if fold_col and fold_col in probs.columns:
        merge_cols.append(fold_col)
    merged = rated.merge(probs[merge_cols], on="case_id", how="inner", validate="many_to_one")
    if merged.empty:
        raise RuntimeError("Join USER×p_i produced zero rows")
    missing_cases = set(probs["case_id"].astype(int)) - set(merged["case_id"].astype(int))
    if missing_cases:
        raise RuntimeError(f"Some probability case_ids missing from USER: {sorted(missing_cases)[:20]}")
    # Note: merged may be a subset of USER when p_i covers only train or only test cases.

    ai = compute_ai_decision_features(
        merged["ai_probability_class_1"].to_numpy(),
        merged["rating"].to_numpy(),
        decision_threshold=decision_threshold,
    )
    # Prefer AI columns from compute_*; drop the temporary merge column name clash
    for c in AI_FEATURE_COLS:
        merged[c] = ai[c].to_numpy()

    if fold_col and fold_col not in merged.columns:
        merged[fold_col] = np.nan

    merged["probability_source"] = source
    merged["ai_decision_threshold"] = float(decision_threshold)

    feature_frame = merged[allow].copy()
    assert_d_features(feature_frame.columns, allowed=allow, forbidden=forbidden)

    meta_present = [c for c in META_COLS if c in merged.columns]
    out = pd.concat([merged[meta_present], feature_frame], axis=1)
    # Keep fold-safe audit columns for profiles (not D model inputs)
    out["rater_accuracy_foldsafe"] = merged["rater_accuracy_foldsafe"].to_numpy()
    out["rater_confidence_foldsafe"] = merged["rater_confidence_foldsafe"].to_numpy()
    out.attrs["d_feature_columns"] = allow
    out.attrs["probability_source"] = source
    out.attrs["rater_stats_are_fold_safe"] = True
    return out.sort_values(["case_id", "rater_id"]).reset_index(drop=True)


def extract_xy(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Return X (allowlist only), y (error-rating), feature names."""
    allow = d_feature_allowlist(config)
    assert_d_features(allow, allowed=allow, forbidden=config["solution_d"].get("forbidden_features"))
    missing = [c for c in allow if c not in frame.columns]
    if missing:
        raise ValueError(f"Feature frame missing D columns: {missing}")
    if "error-rating" not in frame.columns:
        raise ValueError("Feature frame missing error-rating label")
    X = frame[allow].copy()
    assert_d_features(X.columns, allowed=allow)
    y = frame["error-rating"].to_numpy(dtype=int)
    return X, y, allow


def rater_profile_table(frame: pd.DataFrame, train_case_ids: Iterable[int]) -> pd.DataFrame:
    """One row per rater: fold-safe stats as seen on train-pool aggregates (non-LOO summary)."""
    train_cases = set(int(x) for x in train_case_ids)
    train_rows = frame.loc[frame["case_id"].isin(train_cases)]
    if train_rows.empty:
        return pd.DataFrame(columns=["rater_id", "n_train", "rater-accuracy", "rater-confidence"])
    # For profiles, report the train-only (non-LOO) aggregate via groupby on train rows
    # using error-rating / rating-confidence — same formulas as val/test side.
    g = train_rows.groupby("rater_id", as_index=False).agg(
        n_train=("error-rating", "size"),
        mean_error=("error-rating", "mean"),
        mean_rating_confidence=("rating-confidence", "mean"),
    )
    g["rater-accuracy"] = 1.0 - g["mean_error"]
    g["rater-confidence"] = g["mean_rating_confidence"]
    return g[["rater_id", "n_train", "rater-accuracy", "rater-confidence"]].sort_values(
        "rater_id"
    )


def run_smoke_from_outer_oof(config: dict, base_dir: Path) -> dict:
    """
    Smoke-test the builder using outer OOF p_i.

    For each outer fold k: train_case_ids = cases with fold != k; build features for
    all 5551 rows under that definition; keep rows of fold k as the 'test' slice
    for export. Also writes a stacked smoke CSV and rater profiles.
    """
    out_dir = project_path(config["outputs"]["d_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_path = project_path(config["outputs"]["diagnostic_dir"], base_dir) / "oof_predictions.csv"
    folds_path = project_path(config["outputs"]["folds_dir"], base_dir) / "case_fold_assignment.csv"
    oof = load_oof_predictions(oof_path)
    folds = load_fold_assignment(folds_path)
    user = load_user_infos(config, base_dir=base_dir)
    user = _strip_target(user)

    n_splits = int(config["validation"]["n_splits"])
    allow = d_feature_allowlist(config)
    smoke_rows: List[pd.DataFrame] = []
    summary = {"folds": {}, "feature_allowlist": allow}

    for fold in range(1, n_splits + 1):
        test_cases = folds.loc[folds["fold"] == fold, "case_id"].astype(int).tolist()
        train_cases = folds.loc[folds["fold"] != fold, "case_id"].astype(int).tolist()
        if set(test_cases) & set(train_cases):
            raise RuntimeError(f"train/test case overlap in smoke fold {fold}")

        # Probabilities: use OOF p_i for all cases (smoke only; nested rebuild is Fase 7)
        frame = build_d_feature_frame(
            user,
            oof,
            train_cases,
            config=config,
            probability_source="nested_oof",
        )
        # Attach official fold from assignment (overwrite any from oof merge)
        frame = frame.drop(columns=["fold"], errors="ignore").merge(
            folds[["case_id", "fold"]], on="case_id", how="left", validate="many_to_one"
        )

        test_frame = frame.loc[frame["fold"] == fold].copy()
        X_test, y_test, _ = extract_xy(test_frame, config)
        if X_test.isna().any().any():
            raise RuntimeError(f"NaN in D features on smoke test fold {fold}")

        # Preprocessor fit on train rows only (definition check; not saved as final artefact)
        train_frame = frame.loc[frame["fold"] != fold]
        X_train, _, _ = extract_xy(train_frame, config)
        pre = make_d_preprocessor()
        X_tr_t, X_te_t, pre = fit_transform_d_features(pre, X_train, X_test)
        assert X_tr_t.shape[1] == len(allow)
        assert X_te_t is not None and X_te_t.shape[0] == len(test_frame)

        profiles = rater_profile_table(frame, train_cases)
        profiles_path = out_dir / f"rater_profiles_foldsafe_smoke_fold_{fold}.csv"
        profiles.to_csv(profiles_path, index=False)

        test_frame = test_frame.copy()
        test_frame["smoke_outer_fold"] = fold
        smoke_rows.append(test_frame)

        summary["folds"][str(fold)] = {
            "n_train_cases": len(train_cases),
            "n_test_cases": len(test_cases),
            "n_test_rows": int(len(test_frame)),
            "n_feature_nan_test": int(X_test.isna().sum().sum()),
            "profiles_path": str(profiles_path),
            "y_prevalence_test": float(y_test.mean()),
        }

    smoke = pd.concat(smoke_rows, ignore_index=True)
    if len(smoke) != len(user):
        raise RuntimeError(f"Smoke stacked rows {len(smoke)} != USER {len(user)}")
    # Exactly one row per decision across folds
    if smoke["id"].duplicated().any():
        raise RuntimeError("Duplicate decision id in smoke stack")

    smoke_path = out_dir / "feature_smoke_from_outer_oof.csv"
    # Export meta + allowlist features (drop audit-only extras optional keep)
    export_cols = [c for c in META_COLS if c in smoke.columns] + allow
    smoke[export_cols].to_csv(smoke_path, index=False)

    # Assert once more on exported feature block
    assert_d_features(smoke[allow].columns, allowed=allow)

    summary_path = out_dir / "feature_smoke_summary.json"
    summary["smoke_csv"] = str(smoke_path)
    summary["n_rows"] = int(len(smoke))
    summary["note"] = (
        "Smoke uses outer OOF p_i with fold-safe rater stats. "
        "Nested train_D must rebuild p_i (inner OOF / outer-test); do not treat this as final D matrix."
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fold-safe D features (no XGBoost).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run outer-OOF smoke build into outputs/D/",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    allow = d_feature_allowlist(cfg)
    print(f"D feature allowlist ({len(allow)}): {allow}")
    assert_d_features(allow, allowed=allow, forbidden=cfg["solution_d"].get("forbidden_features"))

    if not args.smoke:
        print("No action: pass --smoke to write outputs/D/feature_smoke_from_outer_oof.csv")
        print("Preprocessor helper available via make_d_preprocessor() for train_D.")
        return

    summary = run_smoke_from_outer_oof(cfg, base_dir)
    print(f"Wrote {summary['smoke_csv']} ({summary['n_rows']} rows)")
    for fold, info in summary["folds"].items():
        print(
            f"  fold {fold}: test_rows={info['n_test_rows']} "
            f"prev={info['y_prevalence_test']:.4f} nan={info['n_feature_nan_test']}"
        )
    print(summary["note"])


if __name__ == "__main__":
    main()
