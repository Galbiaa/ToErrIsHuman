"""Generate a single outer fold assignment at case level.

Units:
- diagnostic model: one case (n=427)
- solution D: case–rater decisions (n=5551), always grouped by case_id

Stratification uses TARGET and case mean error-rating (dichotomized at the median).
All later stages must read the saved assignment; do not invent new folds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data import load_config, load_ground_truth, load_user_infos, project_path
from reproducibility import bootstrap_run_reproducibility, set_global_seed


def build_case_fold_frame(config: dict, base_dir: str | Path | None = None) -> pd.DataFrame:
    """Build one row per case with stratification fields (no fold yet)."""
    gt = load_ground_truth(config, base_dir=base_dir)
    users = load_user_infos(config, base_dir=base_dir)

    mean_error = (
        users.groupby("case_id", as_index=False)["error-rating"]
        .mean()
        .rename(columns={"error-rating": "mean_error_rating"})
    )
    cases = gt.merge(mean_error, on="case_id", how="inner", validate="one_to_one")
    if len(cases) != len(gt):
        raise ValueError(
            f"Case count mismatch after merging mean error: gt={len(gt)}, merged={len(cases)}"
        )

    median_error = float(cases["mean_error_rating"].median())
    cases = cases.copy()
    cases["error_above_median"] = (cases["mean_error_rating"] >= median_error).astype(int)
    # Joint stratum for StratifiedKFold (4 cells: TARGET × error band)
    cases["stratum"] = cases["TARGET"].astype(str) + "_" + cases["error_above_median"].astype(str)
    cases.attrs["error_median"] = median_error
    return cases.sort_values("case_id").reset_index(drop=True)


def assign_outer_folds(cases: pd.DataFrame, n_splits: int, random_state: int) -> pd.DataFrame:
    """Assign each case to exactly one outer fold via stratified K-fold."""
    if cases["case_id"].duplicated().any():
        raise ValueError("case_id must be unique before fold assignment")

    y = cases["stratum"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold = np.full(len(cases), -1, dtype=int)

    for fold_idx, (_, test_pos) in enumerate(skf.split(cases["case_id"], y), start=1):
        fold[test_pos] = fold_idx

    if (fold < 1).any():
        raise RuntimeError("Some cases were not assigned to a fold")

    out = cases.copy()
    out["fold"] = fold
    return out


def assert_no_case_leakage(assignment: pd.DataFrame, n_splits: int) -> None:
    """Ensure train and test case sets are disjoint for every outer fold."""
    for fold_idx in range(1, n_splits + 1):
        test_ids = set(assignment.loc[assignment["fold"] == fold_idx, "case_id"])
        train_ids = set(assignment.loc[assignment["fold"] != fold_idx, "case_id"])
        overlap = test_ids & train_ids
        if overlap:
            raise AssertionError(f"case_id overlap train/test in fold {fold_idx}: {sorted(overlap)[:20]}")
        if not test_ids:
            raise AssertionError(f"Empty test set for fold {fold_idx}")


def load_fold_assignment(path: str | Path) -> pd.DataFrame:
    """Load the canonical case→fold CSV used by the rest of the pipeline."""
    df = pd.read_csv(path)
    required = {"case_id", "fold", "TARGET", "mean_error_rating", "error_above_median", "stratum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fold assignment missing columns: {sorted(missing)}")
    if df["case_id"].duplicated().any():
        raise ValueError("Fold assignment has duplicate case_id")
    return df


def build_fold_balance_stats(
    assignment: pd.DataFrame,
    n_raters_per_case: int = 13,
    error_median: float | None = None,
) -> pd.DataFrame:
    """
    One row per fold (+ overall) with counts and prevalences for balance checks.
    """
    rows: list[dict] = []

    def _row(label: str, part: pd.DataFrame) -> dict:
        n_cases = int(len(part))
        n_target1 = int((part["TARGET"] == 1).sum())
        n_target0 = int((part["TARGET"] == 0).sum())
        n_err_hi = int((part["error_above_median"] == 1).sum())
        n_err_lo = int((part["error_above_median"] == 0).sum())
        stratum_counts = part["stratum"].value_counts()
        out = {
            "fold": label,
            "n_cases": n_cases,
            "n_decisions": n_cases * n_raters_per_case,
            "n_target_0": n_target0,
            "n_target_1": n_target1,
            "target_prevalence": float(part["TARGET"].mean()) if n_cases else float("nan"),
            "mean_error_rating_mean": float(part["mean_error_rating"].mean()) if n_cases else float("nan"),
            "mean_error_rating_std": float(part["mean_error_rating"].std(ddof=0)) if n_cases else float("nan"),
            "n_error_below_median": n_err_lo,
            "n_error_above_median": n_err_hi,
            "error_above_median_prevalence": (
                float(part["error_above_median"].mean()) if n_cases else float("nan")
            ),
        }
        for stratum in sorted(assignment["stratum"].unique()):
            out[f"n_stratum_{stratum}"] = int(stratum_counts.get(stratum, 0))
        return out

    for fold_idx in sorted(assignment["fold"].unique()):
        rows.append(_row(str(int(fold_idx)), assignment.loc[assignment["fold"] == fold_idx]))
    overall = _row("overall", assignment)
    if error_median is not None:
        overall["error_median_threshold"] = float(error_median)
        for row in rows:
            row["error_median_threshold"] = float(error_median)
    else:
        for row in rows + [overall]:
            row["error_median_threshold"] = float("nan")
    rows.append(overall)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared outer fold assignment by case_id.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    seed = int(cfg.get("reproducibility", {}).get("seed", cfg["validation"]["random_state"]))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    n_splits = int(cfg["validation"]["n_splits"])
    random_state = int(cfg["validation"]["random_state"])

    cases = build_case_fold_frame(cfg, base_dir=base_dir)
    median_error = float(cases.attrs["error_median"])
    assignment = assign_outer_folds(cases, n_splits=n_splits, random_state=random_state)
    assert_no_case_leakage(assignment, n_splits=n_splits)

    out_dir = project_path(cfg["outputs"]["folds_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "case_fold_assignment.csv"

    save_cols = [
        "case_id",
        "TARGET",
        "mean_error_rating",
        "error_above_median",
        "stratum",
        "fold",
    ]
    assignment[save_cols].to_csv(out_path, index=False)

    balance = build_fold_balance_stats(
        assignment,
        n_raters_per_case=13,
        error_median=median_error,
    )
    balance_path = out_dir / "fold_balance_stats.csv"
    balance.to_csv(balance_path, index=False)

    print(f"Cases: {len(assignment)}")
    print(f"Outer folds: {n_splits}")
    print(f"Mean error-rating median (dichotomy threshold): {median_error:.6f}")
    print("\nFold balance stats:")
    print(balance.to_string(index=False))
    print("\nTrain/test case_id disjoint: OK for all folds")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {balance_path}")


if __name__ == "__main__":
    main()
