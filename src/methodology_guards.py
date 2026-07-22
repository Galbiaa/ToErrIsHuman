"""Methodological guards used before / during training and comparison.

These helpers exist so later stages do not accidentally:
- put TARGET into solution D features;
- reuse Excel-wide rater-accuracy / rater-confidence in CV;
- pick thresholds on the test set;
- treat 5551 decision rows as independent in bootstrap;
- leave AI probability provenance undeclared (nested vs fast).
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ProbabilitySource = Literal["nested_oof", "nested_test", "fast_insample", "fast_test"]

ALLOWED_PROBABILITY_SOURCES: tuple[str, ...] = (
    "nested_oof",
    "nested_test",
    "fast_insample",
    "fast_test",
)


def assert_d_features(
    columns: Iterable[str],
    *,
    allowed: Sequence[str] | None = None,
    forbidden: Sequence[str] | None = None,
) -> List[str]:
    """
    Validate feature column names for solution D.
    Raises if TARGET (or other forbidden names) appear, or if an allowlist is violated.
    Returns the cleaned list of column names.
    """
    cols = [str(c) for c in columns]
    forbidden_set = {str(x) for x in (forbidden if forbidden is not None else ["TARGET"])}
    # Also block case-insensitive TARGET
    forbidden_set |= {x.upper() for x in forbidden_set}

    bad_forbidden = [c for c in cols if c in forbidden_set or c.upper() in forbidden_set]
    if bad_forbidden:
        raise ValueError(f"Forbidden feature columns for solution D: {bad_forbidden}")

    if allowed is not None:
        allowed_set = set(allowed)
        extra = [c for c in cols if c not in allowed_set]
        if extra:
            raise ValueError(f"Feature columns not in D allowlist: {extra}")
        missing = [c for c in allowed if c not in cols]
        if missing:
            raise ValueError(f"Required D feature columns missing: {missing}")

    return cols


def tag_probability_source(source: str) -> ProbabilitySource:
    """Require an explicit provenance tag for AI probabilities used by D."""
    if source not in ALLOWED_PROBABILITY_SOURCES:
        raise ValueError(
            f"Unknown probability_source={source!r}. "
            f"Allowed: {ALLOWED_PROBABILITY_SOURCES}"
        )
    return source  # type: ignore[return-value]


def select_threshold_on_validation(
    y_val: np.ndarray | Sequence,
    p_val: np.ndarray | Sequence,
    *,
    method: Literal["youden", "f1"] = "youden",
) -> float:
    """
    Choose an operating threshold using ONLY validation labels/scores.
    Do not pass test labels into this function.
    """
    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val).astype(float)
    if len(y) != len(p):
        raise ValueError("y_val and p_val must have the same length")
    if len(y) == 0:
        raise ValueError("Cannot select threshold on an empty validation set")
    if len(np.unique(y)) < 2:
        return 0.5

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y, p)
        # roc_curve thresholds include +inf as first value; skip non-finite
        j = tpr - fpr
        best_i = int(np.nanargmax(j))
        thr = float(thresholds[best_i])
        if not np.isfinite(thr):
            # Fall back to mid-point style default if only +inf is present
            return 0.5
        return float(np.clip(thr, 0.0, 1.0))

    if method == "f1":
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
        best_thr = 0.5
        best_f1 = -1.0
        from sklearn.metrics import f1_score

        for thr in candidates:
            pred = (p >= thr).astype(int)
            score = float(f1_score(y, pred, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)
        return best_thr

    raise ValueError(f"Unknown threshold method: {method}")


def majority_class_baseline(y: np.ndarray | Sequence) -> dict:
    """Accuracy of always predicting the majority class (sanity check)."""
    y_arr = np.asarray(y).astype(int)
    if len(y_arr) == 0:
        return {
            "majority_class": np.nan,
            "majority_accuracy": np.nan,
            "n": 0,
        }
    values, counts = np.unique(y_arr, return_counts=True)
    maj = int(values[int(np.argmax(counts))])
    acc = float(np.mean(y_arr == maj))
    return {
        "majority_class": maj,
        "majority_accuracy": acc,
        "n": int(len(y_arr)),
    }


def fold_safe_rater_stats(
    df: pd.DataFrame,
    train_case_ids: Iterable[int],
    *,
    case_col: str = "case_id",
    rater_col: str = "rater_id",
    error_col: str = "error-rating",
    confidence_col: str = "rating-confidence",
) -> pd.DataFrame:
    """
    Replace global Excel rater aggregates with fold-safe values.

    - Rows whose case is outside train_case_ids (val/test):
      stats use only that rater's training decisions.
    - Rows whose case is inside train_case_ids:
      leave-one-out stats (the row itself is excluded).

    Never uses the Excel columns rater-accuracy / rater-confidence as inputs.
    """
    required = {case_col, rater_col, error_col, confidence_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fold_safe_rater_stats missing columns: {sorted(missing)}")

    train_cases = set(int(x) for x in train_case_ids)
    out = df.copy()
    n = len(out)
    acc = np.full(n, np.nan, dtype=float)
    conf = np.full(n, np.nan, dtype=float)

    case_ids = out[case_col].to_numpy()
    rater_ids = out[rater_col].to_numpy()
    errors = out[error_col].to_numpy(dtype=float)
    confidences = out[confidence_col].to_numpy(dtype=float)
    in_train = np.array([int(c) in train_cases for c in case_ids], dtype=bool)

    for rater in np.unique(rater_ids):
        rater_mask = rater_ids == rater
        train_mask = rater_mask & in_train
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) == 0:
            # No training decisions for this rater in this fold — leave NaN
            continue

        train_err = errors[train_idx]
        train_conf = confidences[train_idx]
        sum_err = float(train_err.sum())
        sum_conf = float(train_conf.sum())
        n_train = float(len(train_idx))

        # Val/test rows for this rater: train-only aggregates
        eval_mask = rater_mask & ~in_train
        if np.any(eval_mask):
            acc[eval_mask] = 1.0 - (sum_err / n_train)
            conf[eval_mask] = sum_conf / n_train

        # Train rows: leave-one-out
        if n_train <= 1:
            # Cannot LOO with a single point; leave NaN for that train row
            continue
        for j in train_idx:
            loo_err = (sum_err - errors[j]) / (n_train - 1.0)
            loo_conf = (sum_conf - confidences[j]) / (n_train - 1.0)
            acc[j] = 1.0 - loo_err
            conf[j] = loo_conf

    out["rater_accuracy_foldsafe"] = acc
    out["rater_confidence_foldsafe"] = conf
    out.attrs["rater_stats_are_fold_safe"] = True
    out.attrs["used_excel_global_rater_stats"] = False
    return out


def case_bootstrap_indices(
    case_ids: np.ndarray | Sequence,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Iterator[np.ndarray]:
    """
    Yield row-index arrays for bootstrap replicates that resample case_id
    (with replacement), taking all decision rows of each drawn case.
    """
    case_ids_arr = np.asarray(case_ids)
    if case_ids_arr.ndim != 1:
        raise ValueError("case_ids must be 1-D and aligned with decision rows")

    unique_cases = np.unique(case_ids_arr)
    rng = np.random.default_rng(random_state)
    # Pre-index rows per case for speed
    rows_by_case = {int(c): np.flatnonzero(case_ids_arr == c) for c in unique_cases}

    for _ in range(int(n_bootstrap)):
        drawn = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        parts = [rows_by_case[int(c)] for c in drawn]
        yield np.concatenate(parts)


def assert_case_disjoint(train_case_ids: Iterable[int], test_case_ids: Iterable[int]) -> None:
    """Hard check: no case_id may appear in both train and test."""
    overlap = set(int(x) for x in train_case_ids) & set(int(x) for x in test_case_ids)
    if overlap:
        raise AssertionError(
            f"case_id overlap between train and test: {sorted(overlap)[:20]}"
        )
